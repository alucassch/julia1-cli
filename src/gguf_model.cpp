#include "gguf_model.h"

#include "ggml-alloc.h"
#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Meta {
    gguf_context * g;

    int64_t key(const char * name, bool required = true) const {
        int64_t id = gguf_find_key(g, name);
        if (id < 0 && required) throw std::runtime_error(std::string("missing metadata key ") + name);
        return id;
    }
    uint32_t u32(const char * name) const {
        int64_t id = key(name);
        if (gguf_get_kv_type(g, id) != GGUF_TYPE_UINT32) throw std::runtime_error(std::string("expected u32 for ") + name);
        return gguf_get_val_u32(g, id);
    }
    float f32(const char * name) const {
        int64_t id = key(name);
        if (gguf_get_kv_type(g, id) != GGUF_TYPE_FLOAT32) throw std::runtime_error(std::string("expected f32 for ") + name);
        return gguf_get_val_f32(g, id);
    }
    std::string str(const char * name) const {
        int64_t id = key(name);
        if (gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) throw std::runtime_error(std::string("expected string for ") + name);
        return gguf_get_val_str(g, id);
    }
    std::vector<std::string> str_arr(const char * name) const {
        int64_t id = key(name);
        if (gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, id) != GGUF_TYPE_STRING) {
            throw std::runtime_error(std::string("expected string array for ") + name);
        }
        size_t n = gguf_get_arr_n(g, id);
        std::vector<std::string> out;
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) out.emplace_back(gguf_get_arr_str(g, id, i));
        return out;
    }
    std::vector<int32_t> i32_arr(const char * name) const {
        int64_t id = key(name);
        if (gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, id) != GGUF_TYPE_INT32) {
            throw std::runtime_error(std::string("expected i32 array for ") + name);
        }
        size_t n = gguf_get_arr_n(g, id);
        const int32_t * data = (const int32_t *) gguf_get_arr_data(g, id);
        return std::vector<int32_t>(data, data + n);
    }
};

// Fetch a tensor by name and check its PyTorch shape [d0, d1] == ggml ne = [d1, d0].
ggml_tensor * get_tensor(ggml_context * ctx, const std::string & name, std::vector<int64_t> shape, bool require_f32 = false) {
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t) throw std::runtime_error("missing tensor " + name);
    std::vector<int64_t> ne(shape.rbegin(), shape.rend());
    bool ok = true;
    for (size_t i = 0; ok && i < GGML_MAX_DIMS; ++i) ok = t->ne[i] == (i < ne.size() ? ne[i] : 1);
    if (!ok) {
        char buf[256];
        snprintf(buf, sizeof(buf), "tensor %s has ne=[%lld,%lld,%lld,%lld], expected PyTorch shape with %zu dims",
                 name.c_str(), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], shape.size());
        throw std::runtime_error(buf);
    }
    if (require_f32 && t->type != GGML_TYPE_F32) throw std::runtime_error("tensor " + name + " must be F32");
    return t;
}

} // namespace

JuliaModel::~JuliaModel() {
    if (buf) ggml_backend_buffer_free(buf);
    if (ctx) ggml_free(ctx);
    if (map_addr) munmap(map_addr, map_size);
}

void read_model(const std::string & path, JuliaModel & m, bool with_tokenizer) {
    ggml_context * ctx = nullptr;
    gguf_init_params params = { /*no_alloc*/ true, /*ctx*/ &ctx };
    gguf_context * g = gguf_init_from_file(path.c_str(), params);
    if (!g) throw std::runtime_error("cannot read GGUF file " + path);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> g_guard(g, gguf_free);
    m.ctx = ctx;
    Meta meta{ g };

    std::string arch = meta.str("general.architecture");
    if (arch != "julia1") {
        throw std::runtime_error("general.architecture is '" + arch + "', expected 'julia1' (a full Julia-1 GGUF written by "
                                 "tools/convert_julia1_to_gguf.py)");
    }
    m.name = gguf_find_key(g, "general.name") >= 0 ? meta.str("general.name") : "";
    if (gguf_find_key(g, "general.file_type") >= 0) {
        switch (meta.u32("general.file_type")) {
            case 0: m.file_type = "F32"; break;
            case 1: m.file_type = "F16"; break;
            case 2: m.file_type = "Q4_0"; break;
            case 7: m.file_type = "Q8_0"; break;
            case 8: m.file_type = "Q5_0"; break;
            case 32: m.file_type = "BF16"; break;
            default: m.file_type = "type" + std::to_string(meta.u32("general.file_type"));
        }
    }

    m.n_ctx       = meta.u32("julia1.context_length");
    m.n_embd      = meta.u32("julia1.embedding_length");
    m.n_layer     = meta.u32("julia1.block_count");
    m.n_ff        = meta.u32("julia1.feed_forward_length");
    m.n_head      = meta.u32("julia1.attention.head_count");
    m.eps         = meta.f32("julia1.attention.layer_norm_epsilon");
    m.swa         = meta.u32("julia1.attention.sliding_window");
    m.swa_pattern = meta.u32("julia1.attention.sliding_window_pattern");
    m.rope_base   = meta.f32("julia1.rope.freq_base");
    m.n_rot       = meta.u32("julia1.rope.dimension_count");
    m.n_vocab     = meta.u32("julia1.vocab_size");
    if (gguf_find_key(g, "julia1.rope.freq_base_swa") >= 0 && meta.f32("julia1.rope.freq_base_swa") != m.rope_base) {
        throw std::runtime_error("julia1.rope.freq_base_swa differs from julia1.rope.freq_base; not supported");
    }
    m.head_layers = meta.u32("julia1.head.block_count");
    m.head_n_head = meta.u32("julia1.head.attention.head_count");
    m.head_ff     = meta.u32("julia1.head.feed_forward_length");
    m.head_eps    = meta.f32("julia1.head.layer_norm_epsilon");
    if (m.n_embd % m.n_head != 0 || m.n_embd / m.n_head != m.n_rot) throw std::runtime_error("head size must equal rope dimension count");
    if (m.n_embd % m.head_n_head != 0) throw std::runtime_error("head_count does not divide embedding_length");

    m.cls_id              = meta.u32("julia1.encoding.cls_token_id");
    m.sep_id              = meta.u32("julia1.encoding.sep_token_id");
    m.marker_id           = meta.u32("julia1.encoding.marker_token_id");
    m.marker_text         = meta.str("julia1.encoding.marker_token_text");
    m.head_template       = meta.str("julia1.encoding.head_template");
    m.option_prefix       = meta.str("julia1.encoding.option_prefix");
    m.option_token_limit  = meta.u32("julia1.encoding.option_token_limit");
    m.min_options         = meta.u32("julia1.encoding.min_options");
    m.max_options         = meta.u32("julia1.encoding.max_options");
    m.default_max_length  = meta.u32("julia1.encoding.default_max_length");
    m.default_head_length = meta.u32("julia1.encoding.default_head_length");

    // tokenizer
    if (with_tokenizer) {
        std::vector<std::string> tokens = meta.str_arr("tokenizer.ggml.tokens");
        if ((int32_t) tokens.size() != m.n_vocab) throw std::runtime_error("tokenizer.ggml.tokens size != julia1.vocab_size");
        std::vector<std::string> merges = meta.str_arr("tokenizer.ggml.merges");
        std::vector<std::string> added = meta.str_arr("julia1.tokenizer.added_tokens");
        std::vector<int32_t> added_ids = meta.i32_arr("julia1.tokenizer.added_token_ids");
        int32_t unk = meta.u32("tokenizer.ggml.unknown_token_id");
        int32_t mask = meta.u32("tokenizer.ggml.mask_token_id");
        m.tokenizer.init(std::move(tokens), merges, added, added_ids, unk, mask);
    }

    // tensors (SPEC §2.2)
    const int64_t E = m.n_embd, F = m.n_ff, V = m.n_vocab, HF = m.head_ff;
    m.tok_embd = get_tensor(ctx, "token_embd.weight", { V, E });
    m.tok_norm = get_tensor(ctx, "token_embd_norm.weight", { E }, true);
    m.out_norm = get_tensor(ctx, "output_norm.weight", { E }, true);
    m.layers.resize(m.n_layer);
    for (int i = 0; i < m.n_layer; ++i) {
        std::string p = "blk." + std::to_string(i) + ".";
        EncoderLayer & L = m.layers[i];
        if (i != 0) L.attn_norm = get_tensor(ctx, p + "attn_norm.weight", { E }, true);
        L.wqkv     = get_tensor(ctx, p + "attn_qkv.weight", { 3 * E, E });
        L.wo       = get_tensor(ctx, p + "attn_output.weight", { E, E });
        L.ffn_norm = get_tensor(ctx, p + "ffn_norm.weight", { E }, true);
        L.ffn_up   = get_tensor(ctx, p + "ffn_up.weight", { 2 * F, E });
        L.ffn_down = get_tensor(ctx, p + "ffn_down.weight", { E, F });
    }
    m.type_embd = get_tensor(ctx, "julia1.type_embd.weight", { 3, E }, true);
    m.head.resize(m.head_layers);
    for (int j = 0; j < m.head_layers; ++j) {
        std::string p = "julia1.head." + std::to_string(j) + ".";
        HeadLayer & H = m.head[j];
        H.attn_norm_w = get_tensor(ctx, p + "attn_norm.weight", { E }, true);
        H.attn_norm_b = get_tensor(ctx, p + "attn_norm.bias", { E }, true);
        H.wqkv        = get_tensor(ctx, p + "attn_qkv.weight", { 3 * E, E }, true);
        H.bqkv        = get_tensor(ctx, p + "attn_qkv.bias", { 3 * E }, true);
        H.wo          = get_tensor(ctx, p + "attn_output.weight", { E, E }, true);
        H.bo          = get_tensor(ctx, p + "attn_output.bias", { E }, true);
        H.ffn_norm_w  = get_tensor(ctx, p + "ffn_norm.weight", { E }, true);
        H.ffn_norm_b  = get_tensor(ctx, p + "ffn_norm.bias", { E }, true);
        H.ffn_up      = get_tensor(ctx, p + "ffn_up.weight", { HF, E }, true);
        H.ffn_up_b    = get_tensor(ctx, p + "ffn_up.bias", { HF }, true);
        H.ffn_down    = get_tensor(ctx, p + "ffn_down.weight", { E, HF }, true);
        H.ffn_down_b  = get_tensor(ctx, p + "ffn_down.bias", { E }, true);
    }
    m.sc_norm_w = get_tensor(ctx, "julia1.scorer.norm.weight", { E }, true);
    m.sc_norm_b = get_tensor(ctx, "julia1.scorer.norm.bias", { E }, true);
    m.sc_up_w   = get_tensor(ctx, "julia1.scorer.up.weight", { E, E }, true);
    m.sc_up_b   = get_tensor(ctx, "julia1.scorer.up.bias", { E }, true);
    m.sc_out_w  = get_tensor(ctx, "julia1.scorer.out.weight", { 1, E }, true);
    m.sc_out_b  = get_tensor(ctx, "julia1.scorer.out.bias", { 1 }, true);

    const int64_t n_tensors = gguf_get_n_tensors(g);
    const int64_t expected = 3 + 6 * (int64_t) m.n_layer - 1 + 1 + 12 * (int64_t) m.head_layers + 6;
    if (n_tensors != expected) {
        throw std::runtime_error("file has " + std::to_string(n_tensors) + " tensors, expected " + std::to_string(expected));
    }

    m.data_offset = gguf_get_data_offset(g);
    for (int64_t i = 0; i < n_tensors; ++i) {
        m.tensor_offsets.emplace_back(ggml_get_tensor(ctx, gguf_get_tensor_name(g, i)), gguf_get_tensor_offset(g, i));
    }
}

void load_weights(const std::string & path, ggml_backend_t backend, JuliaModel & m) {
    // Map the file read-only and let the backend's device wrap the tensor data in place (no copy; on Apple silicon
    // Metal's weight buffers are shared host memory either way). Devices without that capability (or a file that
    // cannot be mapped) get a buffer and the bytes streamed from the file.
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    void * addr = MAP_FAILED;
    struct stat st;
    if (props.caps.buffer_from_host_ptr) {
        const int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path);
        if (fstat(fd, &st) == 0) addr = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
    }
    if (addr != MAP_FAILED) {
        m.map_addr = addr;
        m.map_size = (size_t) st.st_size;
        char * data = (char *) addr + m.data_offset;
        const size_t data_size = m.map_size - std::min(m.map_size, m.data_offset);
        size_t max_tensor = 0;
        for (const auto & [t, offset] : m.tensor_offsets) {
            if (offset + ggml_nbytes(t) > data_size) throw std::runtime_error(std::string("short read for tensor ") + t->name);
            max_tensor = std::max(max_tensor, ggml_nbytes(t));
        }
        m.buf = ggml_backend_dev_buffer_from_host_ptr(dev, data, data_size, max_tensor);
        if (!m.buf) throw std::runtime_error("failed to map the weights into a backend buffer");
        ggml_backend_buffer_set_usage(m.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (const auto & [t, offset] : m.tensor_offsets) {
            if (ggml_backend_tensor_alloc(m.buf, t, data + offset) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error(std::string("failed to place tensor ") + t->name);
            }
        }
        return;
    }

    // allocate the backend buffer and stream the tensor bytes from the file
    m.buf = ggml_backend_alloc_ctx_tensors(m.ctx, backend);
    if (!m.buf) throw std::runtime_error("failed to allocate weight buffer");
    ggml_backend_buffer_set_usage(m.buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::unique_ptr<FILE, decltype(&fclose)> f_guard(f, fclose);
    std::vector<char> chunk(64u << 20);
    for (const auto & [t, offset] : m.tensor_offsets) {
        const size_t nbytes = ggml_nbytes(t);
        if (fseeko(f, (off_t) (m.data_offset + offset), SEEK_SET) != 0) {
            throw std::runtime_error(std::string("seek failed for tensor ") + t->name);
        }
        for (size_t done = 0; done < nbytes;) {
            size_t want = std::min(chunk.size(), nbytes - done);
            if (fread(chunk.data(), 1, want, f) != want) throw std::runtime_error(std::string("short read for tensor ") + t->name);
            ggml_backend_tensor_set(t, chunk.data(), done, want);
            done += want;
        }
    }
}
