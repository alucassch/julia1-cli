// SPEC §6: builds and runs the ggml graph for one request (encoder + decision head + scorer).
#pragma once

#include "gguf_model.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cstdint>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

class Runtime {
public:
    // backends: priority order (e.g. {metal, cpu}, {blas, cpu} or {cpu}); the model weights must already live on
    // backends[0], which runs the whole graph when it supports every op; otherwise (BLAS) a ggml_backend_sched assigns
    // each op to the first backend that supports it.
    // precise: exact F32 arithmetic on a GPU backend. Built against the patched ggml (patches/ggml-julia1.patch), the
    // matmuls and flash attention are flagged GGML_PREC_F32, which Metal runs with float-staged kernels; with stock
    // ggml every matmul's activation operand is viewed as batches of <= 8 columns so that Metal takes its exact F32
    // mat-vec kernels instead of the half-precision simdgroup GEMM. No effect when backends[0] is not a GPU (CPU, BLAS).
    Runtime(const JuliaModel & model, std::vector<ggml_backend_t> backends, bool precise);
    ~Runtime();
    Runtime(const Runtime &) = delete;
    Runtime & operator=(const Runtime &) = delete;

    struct Sequence {
        const std::vector<int32_t> * ids;
        const std::vector<int32_t> * markers;
        int32_t qtype;
    };

    // Evaluates the sequences in one graph: each is padded to the longest one (rounded up to the padding multiple) and
    // masked separately, and the matmuls run over all their tokens at once. Returns one logit per marker for each
    // sequence, in order. Throws std::runtime_error on invalid inputs or compute failure.
    std::vector<std::vector<float>> forward(const std::vector<Sequence> & batch);
    // forward() in two halves: submit() validates, prepares and enqueues the graph without waiting for it; collect()
    // waits for it and returns its logits. With async, submit() also records an event that collect() waits on instead
    // of the whole backend, so that another Runtime (own inputs and compute buffer) on the same GPU can submit its next
    // graph in between (the batch path overlaps the host work of one graph with the GPU work of the previous one).
    void submit(const std::vector<Sequence> & batch, bool async);
    std::vector<std::vector<float>> collect();

    int64_t padded_length(int64_t n) const { return (n + pad_ - 1) / pad_ * pad_; } // tokens the graph runs for n

private:
    // n, k: padded tokens and markers per sequence; nb: sequences
    ggml_cgraph * build_graph(ggml_context * ctx, int64_t n, int64_t k, int64_t nb) const;
    // (re)allocates the persistent input tensors for >= nb sequences of >= n tokens and >= k markers
    void ensure_inputs(int64_t n, int64_t k, int64_t nb);
    void upload_masks(const std::vector<int64_t> & lengths, int64_t n_pad, int64_t k_pad);
    bool pad_masks(int64_t nb) const { return pad_ > 1 || nb > 1; } // padding masks needed (else no pad keys exist)
    int64_t padded_k(int64_t k) const;

    const JuliaModel & model_;
    std::vector<ggml_backend_t> backends_;
    ggml_backend_t backend_ = nullptr; // backends_[0]
    bool precise_ = false;
    // Metal: n is padded up to a multiple of pad_ (and k to a multiple of 8): PAD_FAST (aligned GEMM and flash
    // attention blocks) in fast mode and with the exact kernels (p32_), 8 in precise mode without them (no matmul
    // batch tail). Pad tokens are masked out as keys everywhere; their own rows are computed (finite) and ignored.
    // 1 = no padding (CPU).
    int64_t pad_ = 1;
    // precise mode built against the patched ggml (GGML_JULIA1_EXACT_MM): plain matmuls and flash attention flagged
    // GGML_PREC_F32, which Metal runs with float-staged kernels (exact); fast-mode padding
    bool p32_ = false;
    ggml_gallocr_t galloc_ = nullptr;  // compute buffer of backend_, reserved for the largest expected graph
    size_t galloc_size_ = 0;           // its size: a change means the buffer was re-created (cached graphs invalid)
    // instead of galloc_ when backend_ cannot run every op: reserved like galloc_; graphs are built and allocated per
    // request (the scheduler's allocation belongs to its last graph), so the graph cache is not used
    ggml_backend_sched_t sched_ = nullptr;

    // Built and allocated graphs by (n_pad, k_pad, sequences), least recently used evicted. A graph only depends on
    // the padded sizes (the real lengths live in the masks and ids), so a hit skips graph build and allocation.
    struct CachedGraph { ggml_context * ctx = nullptr; ggml_cgraph * gf = nullptr; uint64_t last_use = 0; };
    std::map<std::tuple<int64_t, int64_t, int64_t>, CachedGraph> graphs_;
    uint64_t use_clock_ = 0;
    void clear_graphs();

    // Persistent graph inputs in a buffer of backend_ (no per-request host->device copy by a scheduler); graphs take
    // views of their first entries. Sequence b's tokens are ids[b*n_pad ..], its markers are global token indices
    // b*n_pad + m. pos is 0..cap-1, written once; the masks are rewritten only when the lengths change.
    ggml_context * in_ctx_ = nullptr;
    ggml_backend_buffer_t in_buf_ = nullptr;
    ggml_tensor * in_ids_ = nullptr, * in_pos_ = nullptr, * in_markers_ = nullptr, * in_qtype_ = nullptr;
    // F16 masks, one [n_pad (keys), rows (queries)] block per sequence, flat: 0 = attend, -inf = masked
    ggml_tensor * in_swa_mask_ = nullptr;    // n_pad rows: real keys inside the sliding window
    ggml_tensor * in_pad_mask_ = nullptr;    // n_pad rows: real keys (only when pad_masks())
    ggml_tensor * in_marker_mask_ = nullptr; // k_pad rows: real keys, for the marker queries (only when pad_masks())
    int64_t n_cap_ = 0, k_cap_ = 0, b_cap_ = 0;
    std::vector<int64_t> mask_key_; // lengths..., k_pad of the uploaded masks
    std::vector<ggml_fp16_t> mask_host_;
    std::vector<int32_t> ids_host_, markers_host_, qtypes_host_;

    // the submitted, not yet collected graph (logits == nullptr: none)
    struct Pending {
        ggml_cgraph * gf = nullptr;
        ggml_tensor * logits = nullptr;
        int64_t nb = 0, k_pad = 0;
        bool event = false; // collect() waits on event_ (else on the backend)
        std::vector<size_t> n_markers;
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ nullptr, &ggml_free }; // scheduler: the graph's context
    };
    Pending pending_;
    ggml_backend_event_t event_ = nullptr; // created by the first async submit()
};
