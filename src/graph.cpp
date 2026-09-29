#include "graph.h"

#include "ggml-alloc.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

constexpr size_t GRAPH_NODES = 8192;
constexpr size_t GRAPH_CACHE = 64; // cached graphs; ~0.4 MB of touched host memory each
// Metal uses its half-precision simdgroup GEMM when the activation operand has more than 8 columns
// (ggml_metal_op_mul_mat_use_mm: ne11 > 8); batches of <= 8 columns take the exact F32 mat-vec kernels.
constexpr int64_t CHUNK = 8;
// Fast mode pads n to a multiple of PAD_FAST: kernel_mul_mm takes its bounds-checked variants unless the column count is
// a multiple of 32. Flash attention with n_kv % 64 != 0 adds a K/V padding pass per call: stock ggml's pass copies whole
// strided rows and cannot skip the partial last KV block, so 64 is cheaper there; the patched ggml's pass
// (GGML_JULIA1_FA_KVPAD) costs about as much as 32 extra keys, so 32 saves the padded tokens. The logits are the same
// either way (pad keys are masked). Precise mode pads to 8 (CHUNK), so that the matmul batches have no tail.
#ifdef GGML_JULIA1_FA_KVPAD
constexpr int64_t PAD_FAST = 32;
#else
constexpr int64_t PAD_FAST = 64;
#endif
// Metal runs flash attention with fewer than 20 queries on its vec kernel (ggml_metal_op_flash_attn_ext_use_vec)
constexpr int64_t FA_VEC_MAX_Q = 20;

int64_t round_up(int64_t x, int64_t m) { return (x + m - 1) / m * m; }

struct Builder {
    ggml_context * ctx;
    bool precise;
    bool p32 = false;  // precise mode with the patched ggml: exact float-staged GEMM and flash attention kernels

    // W [K, M] x X [K, N] -> [M, N], or X [K, N, B] -> [M, N, B] (marker rows: Metal picks its kernel by N, so each
    // sequence's k_pad marker rows take the same kernel as at batch 1). Precise: view X as [K, 8, N/8] so each batch
    // has <= 8 columns (patched ggml: one exact float-staged GEMM instead).
    ggml_tensor * mm(ggml_tensor * w, ggml_tensor * x) const {
        const int64_t N = x->ne[1];
        if (p32) {
            ggml_tensor * y = ggml_mul_mat(ctx, w, x);
            ggml_prec_set_src(y, GGML_PREC_F32, 1);
            return y;
        }
        if (!precise || N <= CHUNK) return ggml_mul_mat(ctx, w, x);
        if (x->ne[2] > 1) { // N is a multiple of 8 here (k_pad), so no 8-column batch spans two sequences
            ggml_tensor * y = mm(w, ggml_reshape_2d(ctx, x, x->ne[0], N * x->ne[2]));
            return ggml_reshape_3d(ctx, y, y->ne[0], N, x->ne[2]);
        }
        const int64_t n_main = N - N % CHUNK;
        ggml_tensor * x3 = ggml_view_3d(ctx, x, x->ne[0], CHUNK, n_main / CHUNK, x->nb[1], CHUNK * x->nb[1], 0);
        ggml_tensor * y = ggml_mul_mat(ctx, w, x3);                                    // [M, 8, N/8]
        y = ggml_reshape_2d(ctx, y, y->ne[0], n_main);                                // column 8c+r preserved
        if (n_main < N) {
            ggml_tensor * xt = ggml_view_2d(ctx, x, x->ne[0], N - n_main, x->nb[1], n_main * x->nb[1]);
            y = ggml_concat(ctx, y, ggml_mul_mat(ctx, w, xt), 1);
        }
        return y;
    }

    // q: [head_dim, n_head, n_q, B]; k, v: [head_dim, n_head, n_kv, B]; mask: F16 [n_kv, n_q, 1, B] or null (B
    // sequences attend separately). Returns [n_embd, n_q*B]. Mirrors llama.cpp build_attn_mha.
    ggml_tensor * attention(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask, float scale) const {
        const int64_t hd = q->ne[0], nh = q->ne[1], n_q = q->ne[2], nb = q->ne[3];
        q = ggml_permute(ctx, q, 0, 2, 1, 3); // [head_dim, n_q, n_head]
        k = ggml_permute(ctx, k, 0, 2, 1, 3);
        v = ggml_permute(ctx, v, 0, 2, 1, 3);
        // K and V stay F32 strided views. KV blocks whose mask is all -inf are skipped, so the sliding layers cost
        // O(n * window). Metal rounds Q to half inside the kernel (K, V, softmax and accumulation stay F32); the CPU
        // kernel is F32 throughout. With the patched ggml, GGML_PREC_F32 makes Metal's non-vec kernel (n_q >= 20)
        // stage Q as float, which is exact; its vec kernel (n_q < 20) still rounds Q.
        const bool exact_q = p32 && n_q >= FA_VEC_MAX_Q;
        if (precise && !exact_q) {
            // Exact on Metal: split q = hi + lo with hi = f32(half(q)) and attend with [hi, lo] against [k, k]
            // (head dim 128). The kernel's rounding of Q to half is then exact for hi and costs lo at most
            // 2^-11 |lo| <= 2^-23 |q|, so Q.K is F32-grade. V is zero-padded to 128 (Metal has no dk 128 / dv 64
            // kernel) and the extra output dims are dropped.
            ggml_tensor * q_hi = ggml_cast(ctx, ggml_cast(ctx, q, GGML_TYPE_F16), GGML_TYPE_F32);
            q = ggml_concat(ctx, q_hi, ggml_sub(ctx, q, q_hi), 0);
            k = ggml_concat(ctx, k, k, 0);
            v = ggml_pad(ctx, v, (int) hd, 0, 0, 0);
        }
        ggml_tensor * o = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f); // [dv, n_head, n_q]
        if (precise) ggml_prec_set_acc(o, GGML_PREC_F32);
        if (precise && !exact_q) o = ggml_cont(ctx, ggml_view_4d(ctx, o, hd, nh, n_q, nb, o->nb[1], o->nb[2], o->nb[3], 0));
        return ggml_reshape_2d(ctx, o, hd * nh, n_q * nb);
    }

    ggml_tensor * layer_norm(ggml_tensor * x, float eps, ggml_tensor * w, ggml_tensor * b) const {
        ggml_tensor * y = ggml_mul(ctx, ggml_norm(ctx, x, eps), w);
        return b ? ggml_add(ctx, y, b) : y;
    }
};

} // namespace

Runtime::Runtime(const JuliaModel & model, std::vector<ggml_backend_t> backends, bool precise)
    : model_(model), backends_(std::move(backends)) {
    if (backends_.empty()) throw std::runtime_error("no backends");
    backend_ = backends_[0];
    const bool on_gpu = ggml_backend_dev_type(ggml_backend_get_device(backend_)) == GGML_BACKEND_DEVICE_TYPE_GPU;
    // the CPU (and BLAS) backends compute F32 matmuls exactly already; the exact kernels only matter for Metal
    precise_ = precise && on_gpu;
#ifdef GGML_JULIA1_EXACT_MM
    p32_ = precise_;
#endif
    pad_ = !on_gpu ? 1 : precise_ && !p32_ ? CHUNK : PAD_FAST;

    // Reserve the compute buffer once for a request of 1024 tokens, so that requests up to that size never re-plan or
    // grow it (longer requests grow it automatically). Every node runs on backend_ if it can; otherwise
    // (the BLAS backend runs only matmuls) a scheduler spreads the graph over all backends.
    const int64_t n_res = std::min<int64_t>(model_.n_ctx, 1024), k_res = padded_k(std::min<int64_t>(model_.max_options, n_res));
    ensure_inputs(n_res, k_res, 1);
    ggml_init_params ip = { ggml_tensor_overhead() * GRAPH_NODES + ggml_graph_overhead_custom(GRAPH_NODES, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) throw std::runtime_error("failed to create graph context");
    struct CtxGuard { ggml_context * c; ~CtxGuard() { ggml_free(c); } } guard{ ctx };
    ggml_cgraph * gf = build_graph(ctx, n_res, k_res, 1);
    bool single = true;
    for (int i = 0; i < ggml_graph_n_nodes(gf) && single; ++i) {
        ggml_tensor * t = ggml_graph_node(gf, i);
        single = ggml_backend_supports_op(backend_, t);
        if (!single && backends_.size() == 1) {
            throw std::runtime_error(std::string(ggml_backend_name(backend_)) + " does not support " + ggml_op_desc(t) + " (" + t->name + ")");
        }
    }
    if (!single) {
        sched_ = ggml_backend_sched_new(backends_.data(), nullptr, (int) backends_.size(), GRAPH_NODES, false, false);
        if (!sched_ || !ggml_backend_sched_reserve(sched_, gf)) throw std::runtime_error("failed to reserve the scheduler's compute buffers");
        return;
    }
    galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!galloc_) throw std::runtime_error("failed to create the graph allocator");
    if (!ggml_gallocr_reserve(galloc_, gf)) throw std::runtime_error("failed to reserve the compute buffer");
    galloc_size_ = ggml_gallocr_get_buffer_size(galloc_, 0);
}

Runtime::~Runtime() {
    if (pending_.logits) { // the device may still use this runtime's buffers
        if (sched_) ggml_backend_sched_synchronize(sched_);
        else ggml_backend_synchronize(backend_);
    }
    if (event_) ggml_backend_event_free(event_);
    clear_graphs();
    if (sched_) ggml_backend_sched_free(sched_);
    if (galloc_) ggml_gallocr_free(galloc_);
    if (in_buf_) ggml_backend_buffer_free(in_buf_);
    if (in_ctx_) ggml_free(in_ctx_);
}

void Runtime::clear_graphs() {
    for (auto & [key, g] : graphs_) ggml_free(g.ctx);
    graphs_.clear();
}

void Runtime::ensure_inputs(int64_t n, int64_t k, int64_t nb) {
    if (n <= n_cap_ && k <= k_cap_ && nb <= b_cap_) return;
    clear_graphs(); // they view the old input tensors
    const int64_t n_cap = std::max(n, n_cap_), k_cap = std::max(k, k_cap_), b_cap = std::max(nb, b_cap_);
    if (in_buf_) ggml_backend_buffer_free(in_buf_);
    if (in_ctx_) ggml_free(in_ctx_);
    in_buf_ = nullptr;
    ggml_init_params ip = { 8 * ggml_tensor_overhead(), nullptr, true };
    in_ctx_ = ggml_init(ip);
    if (!in_ctx_) throw std::runtime_error("failed to create the input context");
    in_ids_ = ggml_new_tensor_1d(in_ctx_, GGML_TYPE_I32, b_cap * n_cap);
    ggml_set_name(in_ids_, "ids");
    in_pos_ = ggml_new_tensor_1d(in_ctx_, GGML_TYPE_I32, n_cap);
    ggml_set_name(in_pos_, "pos");
    in_markers_ = ggml_new_tensor_1d(in_ctx_, GGML_TYPE_I32, b_cap * k_cap);
    ggml_set_name(in_markers_, "markers");
    in_qtype_ = ggml_new_tensor_1d(in_ctx_, GGML_TYPE_I32, b_cap);
    ggml_set_name(in_qtype_, "qtype");
    in_swa_mask_ = ggml_new_tensor_1d(in_ctx_, GGML_TYPE_F16, b_cap * n_cap * n_cap);
    ggml_set_name(in_swa_mask_, "swa_mask");
    const bool masked = pad_masks(b_cap);
    in_pad_mask_ = ggml_new_tensor_1d(in_ctx_, GGML_TYPE_F16, masked ? b_cap * n_cap * n_cap : 1);
    ggml_set_name(in_pad_mask_, "pad_mask");
    in_marker_mask_ = ggml_new_tensor_1d(in_ctx_, GGML_TYPE_F16, masked ? b_cap * k_cap * n_cap : 1);
    ggml_set_name(in_marker_mask_, "marker_mask");
    in_buf_ = ggml_backend_alloc_ctx_tensors(in_ctx_, backend_);
    if (!in_buf_) throw std::runtime_error("failed to allocate the input buffer");
    std::vector<int32_t> pos((size_t) n_cap);
    for (int64_t i = 0; i < n_cap; ++i) pos[i] = (int32_t) i;
    ggml_backend_tensor_set(in_pos_, pos.data(), 0, ggml_nbytes(in_pos_));
    n_cap_ = n_cap;
    k_cap_ = k_cap;
    b_cap_ = b_cap;
    mask_key_.clear();
}

int64_t Runtime::padded_k(int64_t k) const { return pad_ > 1 ? round_up(k, CHUNK) : k; }

// Masks for sequences of lengths[b] real tokens, each padded to n_pad, one block per sequence (row = query). Sliding:
// [n_pad, n_pad] with 0 where j < n and |i - j| <= swa/2. With pad_masks(): padding [n_pad, n_pad] and marker
// [n_pad, k_pad], 0 where j < n. Rows of pad queries (i >= n) attend to the last real key only: they stay finite (pad
// keys and values then contribute exact zeros to the real rows) and flash attention skips all their other key blocks.
void Runtime::upload_masks(const std::vector<int64_t> & lengths, int64_t n_pad, int64_t k_pad) {
    std::vector<int64_t> key(lengths);
    key.push_back(k_pad);
    if (key == mask_key_) return;
    const int64_t nb = (int64_t) lengths.size(), half_window = model_.swa / 2;
    const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f), neg_inf = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
    // writes nb blocks of `rows` rows: row i of block b is 0 on keys [j0, j1) = window(lengths[b], i), -inf elsewhere
    const auto fill = [&](int64_t rows, auto window) {
        mask_host_.resize((size_t) (nb * rows * n_pad));
        for (int64_t b = 0; b < nb; ++b) {
            for (int64_t i = 0; i < rows; ++i) {
                const auto [j0, j1] = window(lengths[b], i);
                ggml_fp16_t * r = mask_host_.data() + (b * rows + i) * n_pad;
                std::fill(r, r + j0, neg_inf);
                std::fill(r + j0, r + j1, zero);
                std::fill(r + j1, r + n_pad, neg_inf);
            }
        }
        return mask_host_.size() * sizeof(ggml_fp16_t);
    };
    using Keys = std::pair<int64_t, int64_t>;
    const auto real_keys = [](int64_t n, int64_t i) { return i < n ? Keys(0, n) : Keys(n - 1, n); };
    if (n_pad > half_window + 1) {
        const size_t bytes = fill(n_pad, [&](int64_t n, int64_t i) {
            return i < n ? Keys(std::max<int64_t>(0, i - half_window), std::min(n, i + half_window + 1)) : Keys(n - 1, n);
        });
        ggml_backend_tensor_set(in_swa_mask_, mask_host_.data(), 0, bytes);
    }
    if (pad_masks(nb)) {
        size_t bytes = fill(n_pad, real_keys); // before data(): fill() may reallocate
        ggml_backend_tensor_set(in_pad_mask_, mask_host_.data(), 0, bytes);
        bytes = fill(k_pad, [](int64_t n, int64_t) { return Keys(0, n); }); // every marker row is a real query
        ggml_backend_tensor_set(in_marker_mask_, mask_host_.data(), 0, bytes);
    }
    mask_key_ = std::move(key);
}

ggml_cgraph * Runtime::build_graph(ggml_context * ctx, int64_t n, int64_t k, int64_t nb) const {
    const JuliaModel & m = model_;
    const int64_t E = m.n_embd, H = m.n_head, D = E / H;
    const int64_t HH = m.head_n_head, HD = E / HH;
    const float scale = 1.0f / std::sqrt((float) D);
    const float head_scale = 1.0f / std::sqrt((float) HD);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, GRAPH_NODES, false);
    const Builder b{ ctx, precise_, p32_ };

    // tokens of all sequences side by side: [E, n*nb], sequence s in columns s*n .. s*n + n - 1
    ggml_tensor * t_ids = ggml_view_1d(ctx, in_ids_, n * nb, 0);
    ggml_tensor * t_pos = ggml_view_1d(ctx, in_pos_, n, 0);
    ggml_tensor * t_markers = ggml_view_1d(ctx, in_markers_, k * nb, 0); // global token indices
    ggml_tensor * t_qtype = ggml_view_1d(ctx, in_qtype_, nb, 0);
    const size_t row = n * sizeof(ggml_fp16_t);
    const auto mask = [&](ggml_tensor * t, int64_t rows) { return ggml_view_4d(ctx, t, n, rows, 1, nb, row, rows * row, rows * row, 0); };
    ggml_tensor * t_pad = pad_masks(nb) ? mask(in_pad_mask_, n) : nullptr;       // global layers, head
    ggml_tensor * t_pad_k = pad_masks(nb) ? mask(in_marker_mask_, k) : nullptr; // marker queries
    // the window covers every pair when n <= swa/2 + 1: the sliding layers need no more than the padding mask then
    ggml_tensor * t_swa = n > m.swa / 2 + 1 ? mask(in_swa_mask_, n) : t_pad;

    // the question-type rows first, so that the final norm's NORM, MUL and this ADD are consecutive (Metal fuses them)
    ggml_tensor * type_row = ggml_reshape_3d(ctx, ggml_get_rows(ctx, m.type_embd, t_qtype), E, 1, nb); // [E, 1, nb]
    ggml_build_forward_expand(gf, type_row);

    // ---- encoder ----
    ggml_tensor * x = ggml_get_rows(ctx, m.tok_embd, t_ids);                 // [E, n]
    x = b.layer_norm(x, m.eps, m.tok_norm, nullptr);
    for (int il = 0; il < m.n_layer; ++il) {
        const EncoderLayer & L = m.layers[il];
        ggml_tensor * h = L.attn_norm ? b.layer_norm(x, m.eps, L.attn_norm, nullptr) : x;
        ggml_tensor * qkv = b.mm(L.wqkv, h); // [3E, n*nb]
        const size_t es = ggml_element_size(qkv);
        // one RoPE over the 2H query and key heads (same per-head math as roping q and k separately)
        ggml_tensor * qk = ggml_view_4d(ctx, qkv, D, 2 * H, n, nb, D * es, qkv->nb[1], n * qkv->nb[1], 0);
        qk = ggml_rope_ext(ctx, qk, t_pos, nullptr, m.n_rot, GGML_ROPE_TYPE_NEOX, m.n_ctx, m.rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        ggml_tensor * q = ggml_view_4d(ctx, qk, D, H, n, nb, qk->nb[1], qk->nb[2], qk->nb[3], 0);
        ggml_tensor * kk = ggml_view_4d(ctx, qk, D, H, n, nb, qk->nb[1], qk->nb[2], qk->nb[3], H * qk->nb[1]);
        ggml_tensor * v = ggml_view_4d(ctx, qkv, D, H, n, nb, D * es, qkv->nb[1], n * qkv->nb[1], 2 * E * es);
        const bool global = (il % m.swa_pattern) == 0;
        ggml_tensor * a = b.attention(q, kk, v, global ? t_pad : t_swa, scale);
        x = ggml_add(ctx, x, b.mm(L.wo, a));
        h = b.layer_norm(x, m.eps, L.ffn_norm, nullptr);
        ggml_tensor * u = b.mm(L.ffn_up, h);                                 // [2F, n]: gelu branch | gate
        ggml_tensor * g = ggml_geglu_erf(ctx, u);                            // [F, n]
        x = ggml_add(ctx, x, b.mm(L.ffn_down, g));
    }
    ggml_tensor * hidden = b.layer_norm(ggml_reshape_3d(ctx, x, E, n, nb), m.eps, m.out_norm, nullptr); // [E, n, nb]
    ggml_set_name(hidden, "hidden");

    // ---- decision head ----
    ggml_tensor * hd = ggml_reshape_2d(ctx, ggml_add(ctx, hidden, type_row), E, n * nb); // type row over each sequence
    const int n_head_layers = m.head_layers;
    const auto marker_rows = [&](ggml_tensor * t) { return ggml_reshape_3d(ctx, ggml_get_rows(ctx, t, t_markers), t->ne[0], k, nb); };
    for (int j = 0; j < n_head_layers; ++j) {
        const HeadLayer & Hl = m.head[j];
        ggml_tensor * a = b.layer_norm(hd, m.head_eps, Hl.attn_norm_w, Hl.attn_norm_b);
        ggml_tensor * q, * kk, * v, * mask = t_pad;
        const bool last = j == n_head_layers - 1;
        if (!last) {
            ggml_tensor * qkv = ggml_add(ctx, b.mm(Hl.wqkv, a), Hl.bqkv);   // [3E, n*nb]
            const size_t es = ggml_element_size(qkv);
            q = ggml_view_4d(ctx, qkv, HD, HH, n, nb, HD * es, qkv->nb[1], n * qkv->nb[1], 0);
            kk = ggml_view_4d(ctx, qkv, HD, HH, n, nb, HD * es, qkv->nb[1], n * qkv->nb[1], E * es);
            v = ggml_view_4d(ctx, qkv, HD, HH, n, nb, HD * es, qkv->nb[1], n * qkv->nb[1], 2 * E * es);
        } else {
            // last layer, marker rows only (upstream _selected_head, same math): queries, residual, W_o and the FFN
            // at the k markers of each sequence ([E, k, nb]); keys and values over all its n tokens
            const size_t w1 = Hl.wqkv->nb[1], bs = ggml_element_size(Hl.bqkv);
            ggml_tensor * qm = ggml_add(ctx, b.mm(ggml_view_2d(ctx, Hl.wqkv, E, E, w1, 0), marker_rows(a)),
                                        ggml_view_1d(ctx, Hl.bqkv, E, 0));                     // [E, k, nb]
            ggml_tensor * kv = ggml_add(ctx, b.mm(ggml_view_2d(ctx, Hl.wqkv, E, 2 * E, w1, E * w1), a),
                                        ggml_view_1d(ctx, Hl.bqkv, 2 * E, E * bs));            // [2E, n*nb]
            const size_t es = ggml_element_size(kv);
            q = ggml_view_4d(ctx, qm, HD, HH, k, nb, HD * es, qm->nb[1], qm->nb[2], 0);
            kk = ggml_view_4d(ctx, kv, HD, HH, n, nb, HD * es, kv->nb[1], n * kv->nb[1], 0);
            v = ggml_view_4d(ctx, kv, HD, HH, n, nb, HD * es, kv->nb[1], n * kv->nb[1], E * es);
            hd = marker_rows(hd);                                                                  // [E, k, nb]
            mask = t_pad_k;
        }
        ggml_tensor * att = b.attention(q, kk, v, mask, head_scale);                              // [E, rows]
        if (last) att = ggml_reshape_3d(ctx, att, E, k, nb);
        hd = ggml_add(ctx, hd, ggml_add(ctx, b.mm(Hl.wo, att), Hl.bo));
        ggml_tensor * f = b.layer_norm(hd, m.head_eps, Hl.ffn_norm_w, Hl.ffn_norm_b);
        ggml_tensor * u = ggml_relu(ctx, ggml_add(ctx, b.mm(Hl.ffn_up, f), Hl.ffn_up_b));
        hd = ggml_add(ctx, hd, ggml_add(ctx, b.mm(Hl.ffn_down, u), Hl.ffn_down_b));
    }
    ggml_tensor * sel = n_head_layers > 0 ? hd : marker_rows(hd); // [E, k, nb]
    ggml_tensor * s = b.layer_norm(sel, m.head_eps, m.sc_norm_w, m.sc_norm_b);
    s = ggml_gelu_erf(ctx, ggml_add(ctx, b.mm(m.sc_up_w, s), m.sc_up_b));
    ggml_tensor * logits = ggml_add(ctx, b.mm(m.sc_out_w, s), m.sc_out_b);   // [1, k, nb]
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    return gf;
}

std::vector<std::vector<float>> Runtime::forward(const std::vector<Sequence> & batch) {
    submit(batch, false);
    return collect();
}

void Runtime::submit(const std::vector<Sequence> & batch, bool async) {
    if (pending_.logits) { // an uncollected graph (its collect() threw or was skipped): let it finish first
        if (sched_) ggml_backend_sched_synchronize(sched_);
        else ggml_backend_synchronize(backend_);
        pending_ = {};
    }
    const JuliaModel & m = model_;
    const int64_t nb = (int64_t) batch.size();
    if (nb < 1) throw std::runtime_error("empty batch");
    std::vector<int64_t> lengths((size_t) nb);
    int64_t n = 0, k = 0; // the longest sequence, the most markers
    for (int64_t s = 0; s < nb; ++s) {
        const std::vector<int32_t> & ids = *batch[s].ids, & markers = *batch[s].markers;
        const int64_t len = (int64_t) ids.size();
        if (len < 1 || len > m.n_ctx) throw std::runtime_error("sequence length " + std::to_string(len) + " outside 1.." + std::to_string(m.n_ctx));
        if (markers.empty()) throw std::runtime_error("at least one marker is required");
        for (int32_t p : markers) if (p < 0 || p >= len) throw std::runtime_error("marker position out of range");
        for (int32_t t : ids) if (t < 0 || t >= m.n_vocab) throw std::runtime_error("token id out of range");
        if (batch[s].qtype < 0 || batch[s].qtype >= (int32_t) m.type_embd->ne[1]) throw std::runtime_error("qtype out of range");
        lengths[s] = len;
        n = std::max(n, len);
        k = std::max(k, (int64_t) markers.size());
    }

    const int64_t n_pad = round_up(n, pad_), k_pad = padded_k(k);
    ensure_inputs(n_pad, k_pad, nb);
    // malloc'd, not reused: the pages beyond what the graph touches are never committed
    ggml_init_params ip = { ggml_tensor_overhead() * GRAPH_NODES + ggml_graph_overhead_custom(GRAPH_NODES, false), nullptr, true };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> request_ctx(nullptr, &ggml_free); // scheduler: this request's graph
    ggml_cgraph * gf = nullptr;
    auto it = graphs_.end();
    bool hit = false;
    if (sched_) {
        request_ctx.reset(ggml_init(ip));
        if (!request_ctx) throw std::runtime_error("failed to create graph context");
        gf = build_graph(request_ctx.get(), n_pad, k_pad, nb);
    } else {
        it = graphs_.find({ n_pad, k_pad, nb });
        hit = it != graphs_.end();
        if (!hit) {
            if (graphs_.size() >= GRAPH_CACHE) {
                auto lru = graphs_.begin();
                for (auto g = graphs_.begin(); g != graphs_.end(); ++g) if (g->second.last_use < lru->second.last_use) lru = g;
                ggml_free(lru->second.ctx);
                graphs_.erase(lru);
            }
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) throw std::runtime_error("failed to create graph context");
            it = graphs_.emplace(std::make_tuple(n_pad, k_pad, nb), CachedGraph{ ctx, nullptr, 0 }).first;
            it->second.gf = build_graph(ctx, n_pad, k_pad, nb);
        }
        it->second.last_use = ++use_clock_;
        gf = it->second.gf;
    }
    ggml_tensor * logits = ggml_graph_node(gf, -1);

    if (sched_) {
        ggml_backend_sched_reset(sched_);
        if (!ggml_backend_sched_alloc_graph(sched_, gf)) throw std::runtime_error("failed to allocate the compute graph");
    } else if (!hit) {
        if (!ggml_gallocr_alloc_graph(galloc_, gf)) {
            ggml_free(it->second.ctx);
            graphs_.erase(it);
            throw std::runtime_error("failed to allocate the compute graph");
        }
        const size_t size = ggml_gallocr_get_buffer_size(galloc_, 0);
        if (size != galloc_size_) { // the compute buffer was re-created: only this graph's tensors point into it
            for (auto g = graphs_.begin(); g != graphs_.end();) {
                if (g == it) { ++g; continue; }
                ggml_free(g->second.ctx);
                g = graphs_.erase(g);
            }
            galloc_size_ = size;
        }
    }
    if (pad_masks(nb) || n_pad > m.swa / 2 + 1) upload_masks(lengths, n_pad, k_pad); // only when the lengths changed
    ids_host_.assign((size_t) (nb * n_pad), 0); // pad tokens: id 0
    markers_host_.resize((size_t) (nb * k_pad));
    qtypes_host_.resize((size_t) nb);
    for (int64_t s = 0; s < nb; ++s) {
        const std::vector<int32_t> & ids = *batch[s].ids, & markers = *batch[s].markers;
        std::copy(ids.begin(), ids.end(), ids_host_.begin() + s * n_pad);
        for (int64_t i = 0; i < k_pad; ++i) { // pad rows repeat a real marker; their logits are dropped
            markers_host_[s * k_pad + i] = (int32_t) (s * n_pad) + markers[i < (int64_t) markers.size() ? i : 0];
        }
        qtypes_host_[s] = batch[s].qtype;
    }
    ggml_backend_tensor_set(in_ids_, ids_host_.data(), 0, ids_host_.size() * sizeof(int32_t));
    ggml_backend_tensor_set(in_markers_, markers_host_.data(), 0, markers_host_.size() * sizeof(int32_t));
    ggml_backend_tensor_set(in_qtype_, qtypes_host_.data(), 0, qtypes_host_.size() * sizeof(int32_t));

    // ggml_backend_graph_compute without the wait: collect() waits (on the event recorded here when async)
    if (async && !sched_ && !event_) {
        event_ = ggml_backend_event_new(ggml_backend_get_device(backend_));
        if (!event_) throw std::runtime_error("failed to create a backend event");
    }
    const ggml_status status = sched_ ? ggml_backend_sched_graph_compute_async(sched_, gf) : ggml_backend_graph_compute_async(backend_, gf);
    if (async && !sched_ && status == GGML_STATUS_SUCCESS) ggml_backend_event_record(event_, backend_);
    if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("graph compute failed");
    pending_ = { gf, logits, nb, k_pad, async && !sched_, {}, std::move(request_ctx) };
    for (const Sequence & q : batch) pending_.n_markers.push_back(q.markers->size());
}

std::vector<std::vector<float>> Runtime::collect() {
    if (!pending_.logits) throw std::runtime_error("collect() without submit()");
    const Pending p = std::move(pending_);
    pending_ = {};
    const int64_t nb = p.nb, k_pad = p.k_pad;
    if (p.event) ggml_backend_event_synchronize(event_);
    else if (sched_) ggml_backend_sched_synchronize(sched_);
    else ggml_backend_synchronize(backend_);
    ggml_tensor * logits = p.logits;

    std::vector<float> flat((size_t) (nb * k_pad));
    ggml_backend_tensor_get(logits, flat.data(), 0, flat.size() * sizeof(float));
    std::vector<std::vector<float>> out((size_t) nb);
    for (int64_t s = 0; s < nb; ++s) {
        out[s].assign(flat.begin() + s * k_pad, flat.begin() + s * k_pad + (int64_t) p.n_markers[s]);
        for (float v : out[s]) if (!std::isfinite(v)) throw std::runtime_error("Inference returned nonfinite logits");
    }
    return out;
}
