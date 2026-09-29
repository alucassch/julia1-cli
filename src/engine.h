// Engine: one loaded Julia-1 GGUF model on a device, with the request encoding (SPEC §5), batching and the decision
// APIs of upstream julia (SPEC §7): logits(rows), legacy predict(rows), named predict(state=..., questions=...).
//
// Thread compatibility: one Engine holds one Runtime (graph cache, persistent inputs), one CPU threadpool and shared
// host scratch buffers, so calls that run the model (forward, replay, logits, predict_rows, predict_typed) must be
// serialised by the caller. encode() and tokenize() only read the tokenizer and may run concurrently with each other
// and with a model call.
#pragma once

#include "encoding.h"
#include "gguf_model.h"
#include "graph.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

struct EngineOptions {
    std::string model;          // path of a full Julia-1-<TYPE>.gguf
    std::string device = "auto"; // auto (Metal when it initialises, else CPU) | cpu | metal
    int threads = 4;            // CPU device threads and encode workers; <= 0: min(4, hardware threads)
    bool precise = true;        // Metal: exact F32 kernels (false: half-precision GEMM/attention operands); no effect on CPU
    std::optional<int> max_length = 8192; // upstream julia.load_model defaults; must be 1..julia1.context_length;
                                          // nullopt: julia1.context_length (upstream max_length=None)
    int head_length = 256;
    bool strict = false;        // strict (lossless) encoding
    int batch = 1;              // window size of replay/logits/predict_*: rows per length-grouped set of graphs
    bool tokenizer = true;      // false: replay/forward only (skips the tokenizer build, ~40% of the load)
    bool runtime = true;        // false: encode/tokenize only (no compute buffers, no graphs)
};

class Engine {
public:
    // Loads the model; throws std::runtime_error (also for max_length outside 1..julia1.context_length, as upstream).
    explicit Engine(const EngineOptions & options);
    ~Engine();
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;

    // {name, file_type, model, device, gpu, mode (precise|fast|cpu), threads, n_ctx, max_length, head_length, strict,
    //  batch, defaults {max_length, head_length} (the file's recommended policy), load_ms, version}
    nlohmann::ordered_json info() const;

    // Validates and encodes requests ({state, question, options, type}; julia/data.py validate_row + sequence).
    // Messages are julia/data.py's without the "JSONL line N: " prefix. The vector form encodes on up to `threads`
    // threads and rethrows the error of the first failing request in order.
    Encoded encode(const nlohmann::ordered_json & request) const;
    std::vector<Encoded> encode(const std::vector<nlohmann::ordered_json> & requests) const;

    // Logits of already-encoded sequences as one window: sorted by length and cut into groups (padding against a
    // graph's fixed cost), one padded graph per group. Returns one logit per marker, in input order.
    std::vector<std::vector<float>> forward(const std::vector<Encoded> & sequences);
    // forward() over consecutive windows of `batch` sequences (the CLI replay input: ids, markers, qtype).
    std::vector<std::vector<float>> replay(const std::vector<Encoded> & rows);

    // Upstream FastEngine.logits(rows): every request validated ("JSONL line <i+1>: " prefix) and encoded, then replay().
    std::vector<std::vector<float>> logits(const std::vector<nlohmann::ordered_json> & requests);
    // Upstream legacy predict(rows): [{index, probabilities}] with float32 softmax and julia/probabilities.py's display rule.
    nlohmann::ordered_json predict_rows(const std::vector<nlohmann::ordered_json> & requests);
    // Upstream julia/typed.py predict_typed(state, questions) -> {"answers": {id: {type, probabilities, choice | score |
    // noul, max_probability (choice, score)}}}: full softmax in double, same validation and messages.
    nlohmann::ordered_json predict_typed(const nlohmann::ordered_json & state, const nlohmann::ordered_json & questions);

    // The encoding half of logits(): validation messages with the "JSONL line <i+1>: " prefix when line_prefix, else as
    // encode(). Model-free, like encode().
    std::vector<Encoded> encode_rows(const std::vector<nlohmann::ordered_json> & requests, bool line_prefix) const;
    // predict_typed / predict_rows split around the model, for callers that run it elsewhere (julia1-cli serve):
    // prepare_* validates and encodes (model-free, like encode()), finish_predict builds the result from the logits of
    // call.sequences. predict_typed(s, q) == finish_predict(c, replay(c.sequences)) with c = prepare_typed(s, q).
    struct PredictCall {
        struct Question { std::string id, kind; std::vector<std::string> keys; };
        bool typed = false;
        std::vector<Question> questions; // typed: one per sequence
        std::vector<Encoded> sequences;
    };
    PredictCall prepare_typed(const nlohmann::ordered_json & state, const nlohmann::ordered_json & questions) const;
    PredictCall prepare_rows(const std::vector<nlohmann::ordered_json> & requests) const;
    nlohmann::ordered_json finish_predict(const PredictCall & call, const std::vector<std::vector<float>> & logits) const;

    // SPEC §4: token ids without CLS/SEP.
    std::vector<std::vector<int32_t>> tokenize(const std::vector<std::string> & texts) const;

private:
    // forward() of rows[0..W) as one window
    std::vector<std::vector<float>> run_window(const Encoded * rows, size_t W);
    void need_tokenizer() const;
    void need_runtime() const;

    EngineOptions opts_;
    int n_threads_ = 0;
    std::string device_used_;
    double load_ms_ = 0.0;
    // declaration order = reverse destruction order: the Runtime and the weights go before the backends, the CPU
    // backend before its threadpool
    std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> pool_{ nullptr, &ggml_threadpool_free };
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu_{ nullptr, &ggml_backend_free };
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> blas_{ nullptr, &ggml_backend_free };
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> gpu_{ nullptr, &ggml_backend_free };
    std::unique_ptr<JuliaModel> model_;
    std::unique_ptr<Runtime> runtime_;
    std::unique_ptr<Runtime> runtime2_; // batch > 1 on the GPU: the second half of the group pipeline (run_window)
};

// "julia1-engine <version> (ggml <version>[, exact kernels])" (info()["version"]).
std::string engine_version();
// Softmax in double of float logits (upstream julia/typed.py: exp(x - max) / sum, summed in order; julia1-cli decide).
std::vector<double> softmax(const std::vector<float> & logits);
// Index of the first maximum.
size_t argmax(const std::vector<double> & values);
