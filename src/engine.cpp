#include "engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <future>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

using nlohmann::ordered_json;

namespace {

// A graph's fixed cost in padded tokens, for the grouping below: on Metal one graph costs about as much as 91 more
// padded tokens (the serial kernel chain against the per-token work).
constexpr double GRAPH_TOKENS = 91.0;

// Splits one window into groups (one graph each): rows sorted by length (stable), then cut into consecutive runs that
// minimise the sum over groups of GRAPH_TOKENS + rows * padded(longest length), i.e. padding waste against the fixed
// cost of one more graph (dynamic programming over the cut points). Returns row indices.
std::vector<std::vector<size_t>> length_groups(const std::vector<size_t> & lengths, const Runtime & runtime) {
    std::vector<size_t> order(lengths.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return lengths[a] < lengths[b]; });
    const size_t W = order.size();
    std::vector<double> best(W + 1, std::numeric_limits<double>::infinity());
    std::vector<size_t> cut(W + 1, 0); // best[j]: cost of order[0..j); its last group is order[cut[j]..j)
    best[0] = 0.0;
    for (size_t j = 1; j <= W; ++j) {
        const double padded = (double) runtime.padded_length((int64_t) lengths[order[j - 1]]);
        for (size_t i = 0; i < j; ++i) {
            const double cost = best[i] + GRAPH_TOKENS + (double) (j - i) * padded;
            if (cost < best[j]) { best[j] = cost; cut[j] = i; }
        }
    }
    std::vector<std::vector<size_t>> groups;
    for (size_t j = W; j > 0; j = cut[j]) groups.emplace_back(order.begin() + cut[j], order.begin() + j);
    std::reverse(groups.begin(), groups.end());
    return groups;
}

// Physical performance cores (Apple silicon: hw.perflevel0), else all logical CPUs.
int performance_cores() {
#ifdef __APPLE__
    int n = 0;
    size_t size = sizeof(n);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &size, nullptr, 0) == 0 && n > 0) return n;
#endif
    return (int) std::max(1u, std::thread::hardware_concurrency());
}

// upstream julia/probabilities.py::display_probabilities
std::vector<double> display_probabilities(const std::vector<double> & values) {
    if (values.empty()) return values;
    const size_t winner = argmax(values);
    bool others_small = true;
    for (size_t i = 0; i < values.size(); ++i) if (i != winner && !(values[i] < 0.045)) others_small = false;
    std::vector<double> out(values.size(), 0.0);
    if (values[winner] > 0.95 && others_small) {
        out[winner] = 1.0;
        return out;
    }
    double total = 0.0;
    for (size_t i = 0; i < values.size(); ++i) { out[i] = values[i] >= 0.01 ? values[i] : 0.0; total += out[i]; }
    for (double & v : out) v /= total;
    return out;
}

} // namespace

std::vector<double> softmax(const std::vector<float> & lg) {
    float mx = lg[0];
    for (float v : lg) mx = std::max(mx, v);
    std::vector<double> probs(lg.size());
    double sum = 0.0;
    for (size_t i = 0; i < lg.size(); ++i) { probs[i] = std::exp((double) lg[i] - mx); sum += probs[i]; }
    for (double & p : probs) p /= sum;
    return probs;
}

std::string engine_version() {
#ifdef GGML_JULIA1_EXACT_MM
    const char * kernels = ", exact kernels";
#else
    const char * kernels = "";
#endif
    return std::string("julia1-engine " JULIA1_ENGINE_VERSION " (ggml ") + ggml_version() + kernels + ")";
}

size_t argmax(const std::vector<double> & values) {
    size_t best = 0;
    for (size_t i = 0; i < values.size(); ++i) if (values[i] > values[best]) best = i;
    return best;
}

Engine::Engine(const EngineOptions & options) : opts_(options), model_(std::make_unique<JuliaModel>()) {
    if (opts_.device != "auto" && opts_.device != "cpu" && opts_.device != "metal") throw std::runtime_error("device must be auto, cpu or metal");
    if (opts_.batch < 1) throw std::runtime_error("batch must be >= 1");

    // The model's metadata and tokenizer are read on another thread while the backends initialise (even for the CPU
    // device, ggml's registry sets up Metal); the weights follow once the backend exists.
    const auto t_load0 = std::chrono::steady_clock::now();
    JuliaModel & model = *model_;
    std::future<void> reading = std::async(std::launch::async, read_model, std::cref(opts_.model), std::ref(model), opts_.tokenizer);

    // backends
    if (opts_.device != "cpu") {
        gpu_.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr));
        if (!gpu_ && opts_.device == "metal") throw std::runtime_error("no Metal/GPU backend available");
    }
    cpu_.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    if (!cpu_) throw std::runtime_error("no CPU backend available");
    // Default 4 threads: ggml's spin-wait barriers collapse when the thread count exceeds the free performance cores.
    n_threads_ = opts_.threads > 0 ? opts_.threads : (int) std::min(4u, std::max(1u, std::thread::hardware_concurrency()));
    ggml_backend_cpu_set_n_threads(cpu_.get(), n_threads_);
    // CPU device: one persistent threadpool for every graph (without it ggml creates and joins a pool per graph
    // compute, i.e. per CPU split with BLAS). Between graphs its workers spin (poll 100) when they leave at least two
    // performance cores to Accelerate's threads, which run the large matmuls; otherwise they would take those cores
    // and sleep instead (poll 0).
    const bool free_cores = n_threads_ + 2 <= performance_cores();
    if (!gpu_) {
        ggml_threadpool_params tp = ggml_threadpool_params_default(n_threads_);
        tp.poll = free_cores ? 100 : 0;
        pool_.reset(ggml_threadpool_new(&tp));
        if (!pool_) throw std::runtime_error("failed to create the CPU threadpool");
        ggml_backend_cpu_set_threadpool(cpu_.get(), pool_.get());
    }

    reading.get(); // a read error surfaces after any backend error, as when the model was read after the backends
    // CPU device: Accelerate takes the large matmuls. With the patched ggml the CPU backend calls it itself for F32
    // weights (one backend: cached graphs, no scheduler), while its other workers spin at the node's barrier; that needs
    // the free cores above. Otherwise (stock ggml, F16 weights, which ggml-blas converts to F32, or no free cores) the
    // BLAS backend (-DJULIA1_BLAS=ON) takes them and the runtime schedules the graph over [BLAS, CPU].
#ifdef GGML_JULIA1_CPU_ACCELERATE
    const bool cpu_accelerate = free_cores && std::all_of(model.layers.begin(), model.layers.end(),
                                                          [](const EncoderLayer & L) { return L.ffn_up->type == GGML_TYPE_F32; });
#else
    const bool cpu_accelerate = false;
#endif
    if (!gpu_ && !cpu_accelerate) blas_.reset(ggml_backend_init_by_name("BLAS", nullptr));
    std::vector<ggml_backend_t> backends;
    if (gpu_) backends.push_back(gpu_.get());
    if (blas_) backends.push_back(blas_.get());
    backends.push_back(cpu_.get());
    device_used_ = gpu_ ? ggml_backend_name(gpu_.get()) : blas_ ? "CPU+BLAS" : ggml_backend_name(cpu_.get());
    if (!opts_.max_length) opts_.max_length = model.n_ctx;
    if (*opts_.max_length < 1 || *opts_.max_length > model.n_ctx) { // julia/inference.py::context_length
        throw std::runtime_error("max_length must be an integer between 1 and " + std::to_string(model.n_ctx));
    }
    load_weights(opts_.model, backends[0], model);
    load_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load0).count();

    if (opts_.runtime) runtime_ = std::make_unique<Runtime>(model, backends, opts_.precise && gpu_ != nullptr);
    // batch > 1 on the GPU: a second Runtime (own inputs and compute buffer), so that a window's groups alternate between
    // the two and the next group is prepared and enqueued while the GPU runs the previous one
    if (opts_.runtime && gpu_ && opts_.batch > 1) runtime2_ = std::make_unique<Runtime>(model, backends, opts_.precise);
}

Engine::~Engine() = default;

ordered_json Engine::info() const {
    const JuliaModel & m = *model_;
    ordered_json j;
    j["name"] = m.name;
    j["file_type"] = m.file_type;
    j["model"] = opts_.model;
    j["device"] = device_used_;
    j["gpu"] = gpu_ != nullptr;
    j["mode"] = !gpu_ ? "cpu" : opts_.precise ? "precise" : "fast";
    j["threads"] = n_threads_;
    j["n_ctx"] = m.n_ctx;
    j["max_length"] = *opts_.max_length;
    j["head_length"] = opts_.head_length;
    j["strict"] = opts_.strict;
    j["batch"] = opts_.batch;
    j["defaults"] = { { "max_length", m.default_max_length }, { "head_length", m.default_head_length } };
    j["load_ms"] = load_ms_;
    j["version"] = engine_version();
    return j;
}

void Engine::need_tokenizer() const {
    if (!opts_.tokenizer) throw std::runtime_error("engine created without a tokenizer (EngineOptions::tokenizer)");
}

void Engine::need_runtime() const {
    if (!runtime_) throw std::runtime_error("engine created without a runtime (EngineOptions::runtime)");
}

Encoded Engine::encode(const ordered_json & request) const {
    need_tokenizer();
    return sequence(*model_, parse_request(request), *opts_.max_length, opts_.head_length, opts_.strict);
}

std::vector<Encoded> Engine::encode(const std::vector<ordered_json> & requests) const {
    return encode_rows(requests, false);
}

std::vector<Encoded> Engine::encode_rows(const std::vector<ordered_json> & requests, bool line_prefix) const {
    need_tokenizer();
    // the requests are encoded on up to n_threads threads (the tokenizer is read-only); the first failing request's
    // error is rethrown
    const size_t n = requests.size();
    std::vector<Encoded> enc(n);
    std::vector<std::exception_ptr> errors(n);
    std::atomic<size_t> next_row{ 0 };
    const auto encode_some = [&]() {
        for (size_t r; (r = next_row++) < n;) {
            try {
                Request request;
                try {
                    request = parse_request(requests[r]);
                } catch (const std::runtime_error & e) {
                    if (!line_prefix) throw;
                    throw std::runtime_error("JSONL line " + std::to_string(r + 1) + ": " + e.what()); // julia/data.py::validate_row
                }
                enc[r] = sequence(*model_, request, *opts_.max_length, opts_.head_length, opts_.strict);
            } catch (...) {
                errors[r] = std::current_exception();
            }
        }
    };
    std::vector<std::thread> workers;
    for (size_t t = 1; t < std::min(n, (size_t) n_threads_); ++t) workers.emplace_back(encode_some);
    encode_some();
    for (std::thread & w : workers) w.join();
    for (const std::exception_ptr & e : errors) if (e) std::rethrow_exception(e);
    return enc;
}

std::vector<std::vector<float>> Engine::forward(const std::vector<Encoded> & sequences) {
    need_runtime();
    return run_window(sequences.data(), sequences.size());
}

std::vector<std::vector<float>> Engine::replay(const std::vector<Encoded> & rows) {
    need_runtime();
    std::vector<std::vector<float>> logits;
    logits.reserve(rows.size());
    for (size_t start = 0; start < rows.size(); start += (size_t) opts_.batch) {
        const size_t W = std::min(rows.size() - start, (size_t) opts_.batch);
        for (std::vector<float> & l : run_window(rows.data() + start, W)) logits.push_back(std::move(l));
    }
    return logits;
}

std::vector<std::vector<float>> Engine::run_window(const Encoded * rows, size_t W) {
    std::vector<std::vector<float>> logits(W);
    if (W == 0) return logits;
    std::vector<size_t> lengths(W);
    for (size_t r = 0; r < W; ++r) lengths[r] = rows[r].ids.size();
    const std::vector<std::vector<size_t>> groups = W == 1 ? std::vector<std::vector<size_t>>{ { 0 } } : length_groups(lengths, *runtime_);
    // group i runs on runtimes[i % 2]; with two runtimes group i + 1 is submitted before group i is collected. The last
    // group is collected with a full backend synchronize (command buffer status checks, released event buffers).
    Runtime * runtimes[2] = { runtime_.get(), runtime2_ ? runtime2_.get() : runtime_.get() };
    const bool pipelined = runtime2_ && groups.size() > 1;
    const auto collect = [&](size_t i) {
        Runtime & rt = *runtimes[i % 2];
        std::vector<std::vector<float>> group_logits = rt.collect();
        for (size_t g = 0; g < groups[i].size(); ++g) logits[groups[i][g]] = std::move(group_logits[g]);
    };
    for (size_t i = 0; i < groups.size(); ++i) {
        std::vector<Runtime::Sequence> seqs;
        for (size_t r : groups[i]) seqs.push_back({ &rows[r].ids, &rows[r].markers, rows[r].qtype });
        runtimes[i % 2]->submit(seqs, pipelined && i + 1 < groups.size());
        if (!pipelined) collect(i);
        else if (i > 0) collect(i - 1);
    }
    if (pipelined) collect(groups.size() - 1);
    return logits;
}

std::vector<std::vector<float>> Engine::logits(const std::vector<ordered_json> & requests) {
    need_runtime();
    return replay(encode_rows(requests, true));
}

ordered_json Engine::predict_rows(const std::vector<ordered_json> & requests) {
    need_runtime();
    const PredictCall call = prepare_rows(requests);
    return finish_predict(call, replay(call.sequences));
}

ordered_json Engine::predict_typed(const ordered_json & state, const ordered_json & questions) {
    const PredictCall call = prepare_typed(state, questions);
    return finish_predict(call, replay(call.sequences));
}

Engine::PredictCall Engine::prepare_rows(const std::vector<ordered_json> & requests) const {
    PredictCall call;
    call.sequences = encode_rows(requests, true);
    return call;
}

Engine::PredictCall Engine::prepare_typed(const ordered_json & state, const ordered_json & questions) const {
    if (!questions.is_object() || questions.empty()) throw std::runtime_error("questions must be a nonempty mapping");
    PredictCall call;
    call.typed = true;
    std::vector<ordered_json> rows;
    for (auto it = questions.begin(); it != questions.end(); ++it) {
        const ordered_json & q = it.value();
        if (it.key().empty() || !q.is_object()) throw std::runtime_error("Questions require nonempty string IDs and question objects");
        const auto type = q.find("type");
        const std::string kind = type != q.end() && type->is_string() ? type->get<std::string>() : "";
        const auto criteria_it = q.find("criteria");
        const ordered_json criteria = criteria_it != q.end() ? *criteria_it : ordered_json();
        std::vector<std::string> keys;
        ordered_json labels = ordered_json::array();
        if (kind == "choice") {
            if (!criteria.is_object()) throw std::runtime_error("Choice criteria must map nonempty IDs to descriptions");
            for (auto c = criteria.begin(); c != criteria.end(); ++c) {
                if (c.key().empty()) throw std::runtime_error("Choice criteria must map nonempty IDs to descriptions");
                keys.push_back(c.key());
                labels.push_back(c.value());
            }
        } else if (kind == "score") {
            if (!criteria.is_array()) throw std::runtime_error("Score requires an ordered rubric");
            labels = criteria;
            for (size_t i = 0; i < criteria.size(); ++i) keys.push_back(std::to_string(i));
        } else if (kind == "noul") {
            keys = { "false", "true" };
            if (criteria.is_null()) {
                labels = ordered_json::array({ "false", "true" });
            } else {
                if (!criteria.is_object() || criteria.size() != 2 || !criteria.contains("false") || !criteria.contains("true")) {
                    throw std::runtime_error("Noul criteria must map false and true to descriptions");
                }
                labels = ordered_json::array({ criteria.at("false"), criteria.at("true") });
            }
        } else {
            throw std::runtime_error("Unsupported question type");
        }
        ordered_json row;
        row["state"] = state;
        const auto instructions = q.find("instructions");
        row["question"] = instructions != q.end() ? *instructions : ordered_json();
        row["type"] = kind;
        row["options"] = std::move(labels);
        try {
            parse_request(row); // julia/data.py::validate_row(row, len(rows) + 1)
        } catch (const std::runtime_error & e) {
            throw std::runtime_error("JSONL line " + std::to_string(rows.size() + 1) + ": " + e.what());
        }
        rows.push_back(std::move(row));
        call.questions.push_back({ it.key(), kind, std::move(keys) });
    }
    call.sequences = encode_rows(rows, true); // upstream engine.logits(rows)
    return call;
}

ordered_json Engine::finish_predict(const PredictCall & call, const std::vector<std::vector<float>> & scores) const {
    if (!call.typed) {
        ordered_json result = ordered_json::array();
        for (const std::vector<float> & lg : scores) {
            // torch.tensor(values).softmax(-1): float32; index = argmax of the logits (first maximum)
            float mx = lg[0];
            size_t index = 0;
            for (size_t i = 0; i < lg.size(); ++i) if (lg[i] > mx) { mx = lg[i]; index = i; }
            std::vector<float> e(lg.size());
            float sum = 0.0f;
            for (size_t i = 0; i < lg.size(); ++i) { e[i] = std::exp(lg[i] - mx); sum += e[i]; }
            std::vector<double> probs(lg.size());
            for (size_t i = 0; i < lg.size(); ++i) probs[i] = (double) (e[i] / sum);
            ordered_json r;
            r["index"] = index;
            r["probabilities"] = display_probabilities(probs);
            result.push_back(std::move(r));
        }
        return result;
    }
    if (scores.size() != call.questions.size()) throw std::runtime_error("Model returned an incorrect answer count");
    ordered_json answers = ordered_json::object();
    for (size_t r = 0; r < call.questions.size(); ++r) {
        const PredictCall::Question & m = call.questions[r];
        const std::vector<float> & z = scores[r];
        if (z.size() != m.keys.size() || !std::all_of(z.begin(), z.end(), [](float x) { return std::isfinite(x); })) {
            throw std::runtime_error("Invalid model scores");
        }
        const std::vector<double> p = softmax(z);
        ordered_json result;
        result["type"] = m.kind;
        ordered_json probabilities = ordered_json::object();
        for (size_t i = 0; i < p.size(); ++i) probabilities[m.keys[i]] = p[i];
        result["probabilities"] = std::move(probabilities);
        const size_t best = argmax(p);
        if (m.kind == "choice") {
            result["choice"] = m.keys[best];
        } else if (m.kind == "score") {
            double score = 0.0;
            for (size_t i = 0; i < p.size(); ++i) score += (double) i * p[i];
            result["score"] = score;
        } else {
            result["noul"] = p[1];
        }
        if (m.kind != "noul") result["max_probability"] = p[best];
        answers[m.id] = std::move(result);
    }
    ordered_json out;
    out["answers"] = std::move(answers);
    return out;
}

std::vector<std::vector<int32_t>> Engine::tokenize(const std::vector<std::string> & texts) const {
    need_tokenizer();
    std::vector<std::vector<int32_t>> out;
    out.reserve(texts.size());
    for (const std::string & t : texts) out.push_back(model_->tokenizer.encode(t));
    return out;
}
