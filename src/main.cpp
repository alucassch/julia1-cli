// julia1-cli: native runtime for Julia-1 GGUF (SPEC §7 / CLI contract): replay | decide | predict | tokenize | serve.
// JSONL I/O and argument parsing over Engine (engine.h), which owns the model, the encoding and the batching; serve:
// server.h.
#include "engine.h"
#include "server.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using nlohmann::ordered_json;

namespace {

struct Args {
    std::string model, device = "auto", command, input, output;
    int threads = 0, max_length = 8192; // upstream julia.load_model defaults (head_length 256, not strict)
    std::optional<int> head_length;     // default 256; serve: 512 (upstream inference-policy.json)
    std::optional<bool> strict;         // default false; serve: true
    int batch = 1;
    bool precise = true, timing = false;
    ServerOptions server;
    bool api_key_flag = false;
    std::string devices;                // serve: "metal,cpu" (a CPU helper engine); empty: the single --device engine
    int cpu_threads = 4;                // serve --devices metal,cpu: the CPU engine's threads
};

void usage() {
    fprintf(stderr,
        "usage: julia1-cli --model <gguf> [--device auto|cpu|metal] [--precise|--fast] [--threads N] [--batch B] [--timing]\n"
        "                  <command> --input <in.jsonl> --output <out.jsonl>\n"
        "       julia1-cli --version\n"
        "  replay   input rows: {\"id\",\"ids\",\"markers\",\"qtype\"} -> {\"id\",\"logits\"}\n"
        "  decide   [--max-length 8192] [--head-length 256] [--strict]   (evaluation protocol: --max-length 1024 --head-length 512 --strict)\n"
        "           input rows: {\"state\",\"question\",\"options\",\"type\"} -> {\"logits\",\"index\",\"probabilities\",\"ids\",\"markers\",\"qtype\"}\n"
        "  predict  [--max-length 8192] [--head-length 256] [--strict]   upstream Julia-1 engine.predict, one call per line:\n"
        "           {\"state\",\"questions\":{id:{\"type\",\"instructions\",\"criteria\"}}} -> {\"answers\":{id:{\"type\",\"probabilities\",\n"
        "           \"choice\"|\"score\"|\"noul\",\"max_probability\"}}} (named questions, full softmax), or the legacy list API\n"
        "           {\"rows\":[request, ...]} -> {\"predictions\":[{\"index\",\"probabilities\"}]} (display probabilities)\n"
        "  tokenize input rows: {\"text\"} -> {\"ids\"}\n"
        "  serve    HTTP API (docs/server.md), no --input/--output: [--host 127.0.0.1] [--port 8080 (0: any)]\n"
        "           [--api-key KEY (or env JULIA1_API_KEY)] [--max-batch 16] [--batch-wait-ms 0 (GPU) | 2 (CPU)] [--max-body-mb 4]\n"
        "           [--http-threads 8] [--quiet] [--max-length 8192] [--head-length 512] [--strict (default) | --no-strict]\n"
        "           [--devices metal,cpu] [--cpu-threads 4]\n"
        "           (encoding defaults: upstream inference-policy.json and README example; decide/predict default to\n"
        "           julia.load_model's 8192/256/not strict). GET / (playground), /health, /v1/model; POST /v1/predict,\n"
        "           /v1/decide, /v1/logits, /v1/tokenize. Requests are encoded on the HTTP threads and batched: the oldest\n"
        "           queued request waits up to --batch-wait-ms for others, up to --max-batch sequences per batch.\n"
        "           --devices metal,cpu (instead of --device): a Metal engine and a CPU engine (--cpu-threads) on the same\n"
        "           file; the CPU engine takes a batch only while Metal runs one and at least --max-batch sequences wait\n"
        "  --device: auto (default) uses Metal when it initialises, else the CPU (with Accelerate BLAS when built with it)\n"
        "  --precise (default): exact F32 arithmetic on Metal (typed max abs < 1e-3 on the F32 file). The default build\n"
        "           (patched ggml, patches/ggml-julia1.patch) runs float-staged GEMM and flash-attention kernels, within ~5%%\n"
        "           of --fast; a build against stock ggml (-DJULIA1_GGML_PATCH=OFF) uses 8-column mat-vec batches\n"
        "           instead (~4x slower). The CPU device is always exact\n"
        "  --fast: Metal GEMM and flash attention with half-precision operands (faster, ~1e-1 logit error); no effect on CPU\n"
        "  --threads N: CPU device threads (default 4; keep it below the free performance cores) and, with --batch, the\n"
        "           threads that encode a window's decide rows\n"
        "  --timing: add \"ms\" (per-row wall time: encoding/graph/compute/readback, excluding JSON I/O) to each output row\n"
        "  --batch B: replay/decide take B consecutive rows at a time, sort them by length and split them into groups\n"
        "           that trade padding against the fixed cost of a graph; each group runs as one padded graph. Outputs keep\n"
        "           the input order; with --timing, \"ms\" is the wall time of the B rows divided by B. predict groups the\n"
        "           questions (rows) of each line B at a time. Default 1\n");
}

Args parse_args(int argc, char ** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&](const char * flag) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + flag);
            return argv[++i];
        };
        if (s == "--model") a.model = next("--model");
        else if (s == "--device") a.device = next("--device");
        else if (s == "--threads") a.threads = std::stoi(next("--threads"));
        else if (s == "--input") a.input = next("--input");
        else if (s == "--output") a.output = next("--output");
        else if (s == "--max-length") a.max_length = std::stoi(next("--max-length"));
        else if (s == "--head-length") a.head_length = std::stoi(next("--head-length"));
        else if (s == "--strict") a.strict = true;
        else if (s == "--no-strict") a.strict = false;
        else if (s == "--host") a.server.host = next("--host");
        else if (s == "--port") a.server.port = std::stoi(next("--port"));
        else if (s == "--api-key") { a.server.api_key = next("--api-key"); a.api_key_flag = true; }
        else if (s == "--max-batch") a.server.max_batch = std::stoi(next("--max-batch"));
        else if (s == "--batch-wait-ms") {
            a.server.batch_wait_ms = std::stod(next("--batch-wait-ms"));
            if (!(a.server.batch_wait_ms >= 0)) throw std::runtime_error("--batch-wait-ms must be >= 0");
        }
        else if (s == "--max-body-mb") a.server.max_body_mb = std::stoi(next("--max-body-mb"));
        else if (s == "--http-threads") a.server.http_threads = std::stoi(next("--http-threads"));
        else if (s == "--quiet") a.server.quiet = true;
        else if (s == "--devices") a.devices = next("--devices");
        else if (s == "--cpu-threads") a.cpu_threads = std::stoi(next("--cpu-threads"));
        else if (s == "--precise") a.precise = true;
        else if (s == "--fast") a.precise = false;
        else if (s == "--timing") a.timing = true;
        else if (s == "--batch") a.batch = std::stoi(next("--batch"));
        else if (s == "-h" || s == "--help") { usage(); exit(0); }
        else if (s == "--version") { // "julia1-cli <version> (ggml <version>[, exact kernels])", no model needed
            const std::string v = engine_version(); // "julia1-engine <version> (...)"
            printf("julia1-cli%s\n", v.substr(v.find(' ')).c_str());
            exit(0);
        }
        else if (s == "replay" || s == "decide" || s == "predict" || s == "tokenize" || s == "serve") a.command = s;
        else throw std::runtime_error("unknown argument " + s);
    }
    if (a.model.empty() || a.command.empty() || (a.command != "serve" && (a.input.empty() || a.output.empty()))) {
        usage();
        throw std::runtime_error("--model, <command>, --input and --output are required (serve: --model)");
    }
    if (a.device != "auto" && a.device != "cpu" && a.device != "metal") throw std::runtime_error("--device must be auto, cpu or metal");
    if (a.batch < 1) throw std::runtime_error("--batch must be >= 1");
    if (!a.devices.empty() && a.devices != "metal,cpu") throw std::runtime_error("--devices must be metal,cpu");
    if (!a.devices.empty() && a.device != "auto") throw std::runtime_error("--devices replaces --device");
    if (a.cpu_threads < 1) throw std::runtime_error("--cpu-threads must be >= 1");
    return a;
}

std::vector<int32_t> to_i32(const ordered_json & v, const char * what) {
    if (!v.is_array()) throw std::runtime_error(std::string(what) + " must be an array");
    std::vector<int32_t> out;
    out.reserve(v.size());
    for (const auto & x : v) {
        if (!x.is_number_integer()) throw std::runtime_error(std::string(what) + " must contain integers");
        out.push_back(x.get<int32_t>());
    }
    return out;
}

// One predict line, dispatched like upstream engine.predict(rows=None, questions=None, *, state=None) (a missing key
// or null is None).
ordered_json predict_line(Engine & engine, const ordered_json & row, size_t index) {
    if (!row.is_object()) throw std::runtime_error("line " + std::to_string(index + 1) + ": expected an object");
    const auto get = [&](const char * key) { const auto it = row.find(key); return it == row.end() ? ordered_json() : *it; };
    const ordered_json rows = get("rows"), state = get("state"), questions = get("questions");
    if (!questions.is_null()) {
        if (!rows.is_null() && !state.is_null()) throw std::runtime_error("Pass state either positionally or by keyword, not both");
        return engine.predict_typed(rows.is_null() ? state : rows, questions);
    }
    if (!state.is_null() || rows.is_null()) throw std::runtime_error("Provide legacy rows or state with questions");
    if (!rows.is_array()) throw std::runtime_error("line " + std::to_string(index + 1) + ": rows must be an array of requests");
    ordered_json result;
    result["predictions"] = engine.predict_rows(std::vector<ordered_json>(rows.begin(), rows.end()));
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        Args args = parse_args(argc, argv);
        if (args.command == "serve") {
            if (!args.api_key_flag) {
                if (const char * key = std::getenv("JULIA1_API_KEY")) args.server.api_key = key;
            }
            EngineOptions options;
            options.model = args.model;
            options.device = args.devices.empty() ? args.device : "metal";
            options.threads = args.threads;
            options.precise = args.precise;
            options.max_length = args.max_length;
            options.head_length = args.head_length.value_or(512);
            options.strict = args.strict.value_or(true);
            options.batch = args.server.max_batch;
            const auto loaded = [&](const Engine & engine) {
                const ordered_json info = engine.info();
                fprintf(stderr, "julia1-cli: loaded %s (%s, %s) on %s in %.0f ms, %d threads (%s; max_length %d, head_length %d%s)\n",
                        args.model.c_str(), info["name"].get<std::string>().c_str(), info["file_type"].get<std::string>().c_str(),
                        info["device"].get<std::string>().c_str(), info["load_ms"].get<double>(), info["threads"].get<int>(),
                        info["mode"].get<std::string>().c_str(), *options.max_length, options.head_length, options.strict ? ", strict" : "");
            };
            Engine engine(options);
            loaded(engine);
            std::unique_ptr<Engine> helper;
            if (!args.devices.empty()) { // the CPU engine only runs the model: requests are encoded with `engine`
                EngineOptions cpu = options;
                cpu.device = "cpu";
                cpu.threads = args.cpu_threads;
                cpu.tokenizer = false;
                helper = std::make_unique<Engine>(cpu);
                loaded(*helper);
            }
            return serve(engine, args.server, helper.get());
        }
        std::ifstream in(args.input);
        if (!in) throw std::runtime_error("cannot open input " + args.input);
        std::ofstream out(args.output);
        if (!out) throw std::runtime_error("cannot open output " + args.output);

        EngineOptions options;
        options.model = args.model;
        options.device = args.device;
        options.threads = args.threads;
        options.precise = args.precise;
        if (args.command == "decide" || args.command == "predict") { // the encoding options (max_length is checked at load)
            options.max_length = args.max_length;
            options.head_length = args.head_length.value_or(256);
            options.strict = args.strict.value_or(false);
        }
        options.batch = args.batch;
        options.tokenizer = args.command != "replay"; // replay feeds token ids
        options.runtime = args.command != "tokenize";
        Engine engine(options);
        const ordered_json info = engine.info();
        const std::string device_used = info["device"].get<std::string>();
        const int n_threads = info["threads"].get<int>();
        fprintf(stderr, "julia1-cli: loaded %s (%s, %s) on %s in %.0f ms, %d threads\n", args.model.c_str(),
                info["name"].get<std::string>().c_str(), info["file_type"].get<std::string>().c_str(), device_used.c_str(),
                info["load_ms"].get<double>(), n_threads);

        // predict: one engine call per line (the engine groups its questions); tokenize: one text per line
        const size_t window_rows = args.command == "tokenize" || args.command == "predict" ? 1 : (size_t) args.batch;
        std::vector<size_t> window_index; // input line of each buffered row
        std::vector<ordered_json> window;
        size_t done = 0;
        double total_ms = 0.0;
        const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        // Processes the buffered rows: every row of a window gets the window's wall time divided by its rows as "ms".
        const auto process_window = [&]() {
            const size_t W = window.size();
            std::vector<ordered_json> res(W, ordered_json::object());
            auto t0 = std::chrono::steady_clock::now();
            if (args.command == "tokenize") {
                const ordered_json & row = window[0];
                if (!row.is_object() || !row.contains("text") || !row["text"].is_string()) {
                    throw std::runtime_error("line " + std::to_string(window_index[0] + 1) + ": expected {\"text\": ...}");
                }
                res[0]["ids"] = engine.tokenize({ row["text"].get<std::string>() })[0];
            } else if (args.command == "predict") {
                res[0] = predict_line(engine, window[0], window_index[0]);
            } else {
                std::vector<Encoded> enc;
                if (args.command == "decide") {
                    enc = engine.encode(window); // on up to --threads threads; the first failing row's error
                } else {
                    enc.resize(W);
                    for (size_t r = 0; r < W; ++r) {
                        const ordered_json & row = window[r];
                        if (!row.is_object()) throw std::runtime_error("line " + std::to_string(window_index[r] + 1) + ": expected an object");
                        enc[r].ids = to_i32(row.value("ids", ordered_json::array()), "ids");
                        enc[r].markers = to_i32(row.value("markers", ordered_json::array()), "markers");
                        enc[r].qtype = row.value("qtype", 0);
                    }
                }
                const std::vector<std::vector<float>> logits = engine.replay(enc); // the window: --batch rows
                for (size_t r = 0; r < W; ++r) {
                    if (args.command == "replay") {
                        const ordered_json & row = window[r];
                        res[r]["id"] = row.contains("id") ? row["id"] : ordered_json((int64_t) window_index[r]);
                        res[r]["logits"] = logits[r];
                        continue;
                    }
                    // decide: softmax + argmax (upstream julia/inference.py::predict)
                    const std::vector<double> probs = softmax(logits[r]);
                    res[r]["logits"] = logits[r];
                    res[r]["index"] = argmax(probs);
                    res[r]["probabilities"] = probs;
                    res[r]["ids"] = enc[r].ids;
                    res[r]["markers"] = enc[r].markers;
                    res[r]["qtype"] = enc[r].qtype;
                }
            }
            auto t1 = std::chrono::steady_clock::now();
            const double window_ms = ms(t0, t1), row_ms = window_ms / (double) W;
            total_ms += window_ms;
            for (size_t r = 0; r < W; ++r) {
                if (args.timing) res[r]["ms"] = row_ms;
                out << res[r].dump() << '\n';
            }
            done += W;
            window.clear();
            window_index.clear();
        };

        std::string line;
        for (size_t index = 0; std::getline(in, line); ++index) {
            if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
            try {
                window.push_back(parse_json_line(line));
            } catch (const std::exception & e) {
                throw std::runtime_error("line " + std::to_string(index + 1) + ": invalid JSON: " + e.what());
            }
            window_index.push_back(index);
            if (window.size() == window_rows) process_window();
        }
        if (!window.empty()) process_window();
        out.close();
        if (!out) throw std::runtime_error("failed to write output " + args.output);
        const std::string mode = info["gpu"].get<bool>() ? ", " + info["mode"].get<std::string>() : ""; // CPU: always exact
        fprintf(stderr, "julia1-cli: %s: %zu rows, %.1f ms total, %.2f ms/row (device %s, %d threads%s)\n", args.command.c_str(), done,
                total_ms, done ? total_ms / (double) done : 0.0, device_used.c_str(), n_threads, mode.c_str());
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "julia1-cli: error: %s\n", e.what());
        return 1;
    }
}
