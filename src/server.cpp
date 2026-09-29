// julia1-cli serve: HTTP threads parse, validate and encode requests (errors return at once); one inference thread owns
// the Engine's model calls and runs the queued jobs as dynamic batches. API: docs/server.md.
#include "server.h"

#include "playground.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <iterator>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;

namespace {

double ms_between(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

// A failure of the model call (HTTP 500), as opposed to a validation or encoding error of the request (400).
struct InferenceError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// An error answered with this status; index: the failing element of a /v1/predict array (-1: none).
struct HttpError : std::runtime_error {
    int status;
    long index;
    HttpError(int status, const std::string & message, long index = -1) : std::runtime_error(message), status(status), index(index) {}
};

// Dynamic batching. A job is the encoded sequences of one HTTP request. The inference thread takes the oldest job, then
// keeps taking queued jobs while they fit in max_batch sequences, waiting for new ones until wait after the oldest
// job's arrival (not at all when it has already waited that long, e.g. behind a running batch), and runs the batch as
// one Engine::replay(): windows of max_batch sequences, each sorted by length and cut into padded graphs.
// A helper engine (--devices metal,cpu) has its own inference thread, which takes a batch the same way but only while
// the first engine runs one, at least max_batch sequences are queued and the oldest job fits in max_batch: the first
// engine takes work first, and the helper takes only load that would otherwise wait behind it.
class Batcher {
public:
    struct Result {
        std::vector<std::vector<float>> logits; // one per sequence of the job, in order
        double queue_ms = 0, forward_ms = 0;    // the job's arrival -> its batch's start; the batch's replay()
        size_t batch_rows = 0;                  // sequences in that batch
        std::string device;                     // the engine that ran it (info()["device"])
    };

    Batcher(Engine & engine, Engine * helper, size_t max_batch, double wait_ms)
        : max_batch_(max_batch),
          wait_(std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(wait_ms))) {
        engines_.push_back({ &engine, engine.info()["device"].get<std::string>() });
        if (helper) engines_.push_back({ helper, helper->info()["device"].get<std::string>() });
        for (size_t e = 0; e < engines_.size(); ++e) threads_.emplace_back([this, e] { loop(e); });
    }

    ~Batcher() { // the first engine answers the queued jobs first
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (std::thread & t : threads_) t.join();
    }

    std::future<Result> submit(std::vector<Encoded> sequences) {
        Job job{ std::move(sequences), {}, Clock::now() };
        std::future<Result> result = job.promise.get_future();
        if (job.sequences.empty()) {
            job.promise.set_value({});
            return result;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queued_rows_ += job.sequences.size();
            queue_.push_back(std::move(job));
        }
        cv_.notify_all(); // both inference threads check their condition
        return result;
    }

private:
    struct Job {
        std::vector<Encoded> sequences;
        std::promise<Result> promise;
        Clock::time_point arrival;
    };
    struct Device {
        Engine * engine;
        std::string name;
    };

    // engine e may take the oldest job: the first engine always, the helper only while the first runs a batch
    bool can_take(size_t e) const {
        return !queue_.empty() && (e == 0 || (first_running_ && queued_rows_ >= max_batch_ && queue_.front().sequences.size() <= max_batch_));
    }

    void loop(size_t e) {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            cv_.wait(lock, [&] { return stop_ || can_take(e); });
            if (e == 0 ? queue_.empty() : stop_) return; // stopping: the first engine answers every queued job
            std::vector<Job> batch;
            size_t rows = 0;
            const Clock::time_point deadline = queue_.front().arrival + wait_;
            for (;;) {
                if (queue_.empty() && (!wait_for_job(lock, deadline) || queue_.empty())) break;
                const size_t n = queue_.front().sequences.size();
                if (!batch.empty() && rows + n > max_batch_) break;
                rows += n;
                queued_rows_ -= n;
                batch.push_back(std::move(queue_.front()));
                queue_.pop_front();
                if (rows >= max_batch_) break;
            }
            if (e == 0) first_running_ = true;
            lock.unlock();
            if (e == 0 && engines_.size() > 1) cv_.notify_all(); // the helper may take the queued jobs now
            run(engines_[e], batch, rows);
            lock.lock();
            if (e == 0) first_running_ = false;
        }
    }

    // Waits for a new job (or stop) until the deadline. Timed waits overshoot on macOS by about half their timeout
    // (timer leeway), so the timeout halves until the deadline is less than 50 us away.
    bool wait_for_job(std::unique_lock<std::mutex> & lock, Clock::time_point deadline) {
        for (;;) {
            if (stop_ || !queue_.empty()) return true;
            const Clock::duration left = deadline - Clock::now();
            if (left < std::chrono::microseconds(50)) return false;
            cv_.wait_for(lock, left / 2);
        }
    }

    void run(const Device & device, std::vector<Job> & batch, size_t rows) {
        std::vector<Encoded> sequences;
        sequences.reserve(rows);
        for (Job & job : batch) for (Encoded & s : job.sequences) sequences.push_back(std::move(s)); // sizes stay
        const Clock::time_point t0 = Clock::now();
        try {
            std::vector<std::vector<float>> logits = device.engine->replay(sequences);
            const double forward_ms = ms_between(t0, Clock::now());
            auto next = logits.begin();
            for (Job & job : batch) {
                Result result;
                const auto end = next + (std::ptrdiff_t) job.sequences.size();
                result.logits.assign(std::make_move_iterator(next), std::make_move_iterator(end));
                next = end;
                result.queue_ms = ms_between(job.arrival, t0);
                result.forward_ms = forward_ms;
                result.batch_rows = rows;
                result.device = device.name;
                job.promise.set_value(std::move(result));
            }
        } catch (const std::exception & e) {
            for (Job & job : batch) job.promise.set_exception(std::make_exception_ptr(InferenceError(e.what())));
        }
    }

    const size_t max_batch_;
    const Clock::duration wait_;
    std::vector<Device> engines_; // the first engine, then the helper
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Job> queue_;
    size_t queued_rows_ = 0;      // sequences in queue_
    bool first_running_ = false;  // the first engine runs a batch
    bool stop_ = false;
    std::vector<std::thread> threads_; // one inference thread per engine
};

// Per-request figures for the log line and the Server-Timing header: set by the handler, read by the logger, which
// runs on the same HTTP thread.
struct RequestLog {
    size_t rows = 0, batch = 0;
    double prepare_ms = 0, queue_ms = 0, forward_ms = 0;
    std::string device;
};
thread_local RequestLog t_log;

std::string dump(const ordered_json & j) { return j.dump(-1, ' ', false, ordered_json::error_handler_t::replace); }

void reply(httplib::Response & res, int status, const ordered_json & body) {
    res.status = status;
    res.set_content(dump(body), "application/json");
}

void reply_error(httplib::Response & res, int status, const std::string & message, long index = -1) {
    ordered_json j;
    j["error"] = message;
    if (index >= 0) j["index"] = index;
    reply(res, status, j);
}

ordered_json parse_body(const httplib::Request & req) {
    try {
        return parse_json_line(req.body);
    } catch (const std::exception & e) {
        throw HttpError(400, std::string("invalid JSON: ") + e.what());
    }
}

// body[key] of an object body, which must be an array
std::vector<ordered_json> body_array(const ordered_json & body, const char * key, const char * item) {
    if (!body.is_object() || !body.contains(key) || !body[key].is_array()) {
        throw HttpError(400, std::string("expected a JSON object {\"") + key + "\": [" + item + ", ...]}");
    }
    return std::vector<ordered_json>(body[key].begin(), body[key].end());
}

// One upstream engine.predict(rows=None, questions=None, *, state=None) call with the object's keys (a missing key or
// null is None), as a julia1-cli predict line.
Engine::PredictCall prepare_call(const Engine & engine, const ordered_json & call) {
    if (!call.is_object()) throw std::runtime_error("expected a JSON object {\"state\", \"questions\"} (or legacy {\"rows\"})");
    const auto get = [&](const char * key) { const auto it = call.find(key); return it == call.end() ? ordered_json() : *it; };
    const ordered_json rows = get("rows"), state = get("state"), questions = get("questions");
    if (!questions.is_null()) {
        if (!rows.is_null() && !state.is_null()) throw std::runtime_error("Pass state either positionally or by keyword, not both");
        return engine.prepare_typed(rows.is_null() ? state : rows, questions);
    }
    if (!state.is_null() || rows.is_null()) throw std::runtime_error("Provide legacy rows or state with questions");
    if (!rows.is_array()) throw std::runtime_error("rows must be an array of requests");
    return engine.prepare_rows(std::vector<ordered_json>(rows.begin(), rows.end()));
}

// Runs a request's sequences in the next batch (InferenceError on a model failure).
Batcher::Result run_model(Batcher & batcher, std::vector<Encoded> sequences) {
    t_log.rows = sequences.size();
    Batcher::Result r = batcher.submit(std::move(sequences)).get();
    t_log.batch = r.batch_rows;
    t_log.queue_ms = r.queue_ms;
    t_log.forward_ms = r.forward_ms;
    t_log.device = r.device;
    return r;
}

// constant time in the contents
bool same_key(const std::string & a, const std::string & b) {
    unsigned char diff = a.size() != b.size();
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) diff |= (unsigned char) (a[i] ^ b[i]);
    return diff == 0;
}

std::atomic<httplib::Server *> g_server{ nullptr };

void on_signal(int sig) {
    std::signal(sig, SIG_DFL); // a second signal terminates at once
    if (httplib::Server * server = g_server.load()) server->stop(); // closes the listening socket (async-signal-safe)
}

} // namespace

int serve(Engine & engine, const ServerOptions & options, Engine * helper) {
    ServerOptions opts = options;
    // Default wait: on Metal a batch's cost is nearly linear in its rows, so waiting only adds latency; on the CPU larger
    // batches are cheaper per row.
    if (opts.batch_wait_ms < 0) opts.batch_wait_ms = engine.info()["gpu"].get<bool>() ? 0.0 : 2.0;
    if (opts.max_batch < 1) throw std::runtime_error("--max-batch must be >= 1");
    if (opts.max_body_mb < 1) throw std::runtime_error("--max-body-mb must be >= 1");
    if (opts.http_threads < 1) throw std::runtime_error("--http-threads must be >= 1");
    Batcher batcher(engine, helper, (size_t) opts.max_batch, opts.batch_wait_ms);
    httplib::Server svr;
    const size_t http_threads = (size_t) opts.http_threads;
    svr.new_task_queue = [http_threads] { return new httplib::ThreadPool(http_threads, 4 * http_threads); };
    svr.set_tcp_nodelay(true);
    svr.set_payload_max_length((size_t) opts.max_body_mb << 20);

    using Handler = std::function<void(const httplib::Request &, httplib::Response &)>;
    const auto route = [](Handler fn) {
        return [fn](const httplib::Request & req, httplib::Response & res) {
            t_log = {};
            try {
                fn(req, res);
            } catch (const HttpError & e) {
                reply_error(res, e.status, e.what(), e.index);
            } catch (const InferenceError & e) {
                reply_error(res, 500, e.what());
            } catch (const std::runtime_error & e) { // request validation and encoding: upstream julia/data.py, julia/typed.py messages
                reply_error(res, 400, e.what());
            }
            if (res.status == 200 && t_log.batch > 0) {
                char timing[224];
                snprintf(timing, sizeof(timing), "prepare;dur=%.3f, queue;dur=%.3f, forward;dur=%.3f;desc=\"batch %zu\", device;desc=\"%s\"",
                         t_log.prepare_ms, t_log.queue_ms, t_log.forward_ms, t_log.batch, t_log.device.c_str());
                res.set_header("Server-Timing", timing);
            }
        };
    };
    const auto prepared = [](Clock::time_point t0) { t_log.prepare_ms = ms_between(t0, Clock::now()); };

    svr.Get("/", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(PLAYGROUND_HTML, sizeof(PLAYGROUND_HTML) - 1, "text/html; charset=utf-8");
    });
    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) { reply(res, 200, { { "status", "ok" } }); });
    svr.Get("/v1/model", [&](const httplib::Request &, httplib::Response & res) {
        ordered_json info = engine.info();
        info["server"] = { { "host", opts.host }, { "max_batch", opts.max_batch }, { "batch_wait_ms", opts.batch_wait_ms },
                           { "max_body_mb", opts.max_body_mb }, { "http_threads", opts.http_threads },
                           { "api_key", !opts.api_key.empty() }, { "httplib", CPPHTTPLIB_VERSION } };
        reply(res, 200, info);
    });
    // One julia1-cli predict line (upstream engine.predict: named questions, or legacy {"rows"}) or an array of them; all
    // of a request's calls run in one job.
    svr.Post("/v1/predict", route([&](const httplib::Request & req, httplib::Response & res) {
        const Clock::time_point t0 = Clock::now();
        const ordered_json body = parse_body(req);
        const bool many = body.is_array();
        std::vector<Engine::PredictCall> calls;
        std::vector<Encoded> sequences;
        for (size_t i = 0; i < (many ? body.size() : 1); ++i) {
            try {
                calls.push_back(prepare_call(engine, many ? body[i] : body));
            } catch (const std::runtime_error & e) {
                throw HttpError(400, e.what(), many ? (long) i : -1);
            }
            for (Encoded & s : calls.back().sequences) sequences.push_back(std::move(s)); // sizes stay
        }
        prepared(t0);
        const Batcher::Result r = run_model(batcher, std::move(sequences));
        ordered_json out = ordered_json::array();
        auto next = r.logits.begin();
        for (const Engine::PredictCall & call : calls) {
            const auto end = next + (std::ptrdiff_t) call.sequences.size();
            ordered_json result;
            try {
                result = engine.finish_predict(call, std::vector<std::vector<float>>(next, end));
            } catch (const std::runtime_error & e) { // "Invalid model scores"
                throw InferenceError(e.what());
            }
            next = end;
            out.push_back(call.typed ? std::move(result) : ordered_json{ { "predictions", std::move(result) } });
        }
        reply(res, 200, many ? out : out[0]);
    }));
    // Legacy list API with raw scores: {"rows": [request]} -> {"results": [{index, probabilities (display), logits}]}
    svr.Post("/v1/decide", route([&](const httplib::Request & req, httplib::Response & res) {
        const Clock::time_point t0 = Clock::now();
        Engine::PredictCall call = engine.prepare_rows(body_array(parse_body(req), "rows", "request"));
        prepared(t0);
        const Batcher::Result r = run_model(batcher, std::move(call.sequences));
        ordered_json results = engine.finish_predict(call, r.logits);
        for (size_t i = 0; i < results.size(); ++i) results[i]["logits"] = r.logits[i];
        reply(res, 200, { { "results", std::move(results) } });
    }));
    // Upstream engine.logits(rows)
    svr.Post("/v1/logits", route([&](const httplib::Request & req, httplib::Response & res) {
        const Clock::time_point t0 = Clock::now();
        std::vector<Encoded> sequences = engine.encode_rows(body_array(parse_body(req), "rows", "request"), true);
        prepared(t0);
        const Batcher::Result r = run_model(batcher, std::move(sequences));
        reply(res, 200, { { "logits", r.logits } });
    }));
    // SPEC §4 token ids (no CLS/SEP)
    svr.Post("/v1/tokenize", route([&](const httplib::Request & req, httplib::Response & res) {
        const std::vector<ordered_json> items = body_array(parse_body(req), "texts", "string");
        std::vector<std::string> texts;
        for (const ordered_json & t : items) {
            if (!t.is_string()) throw HttpError(400, "expected a JSON object {\"texts\": [string, ...]}");
            texts.push_back(t.get<std::string>());
        }
        t_log.rows = texts.size();
        reply(res, 200, { { "ids", engine.tokenize(texts) } });
    }));

    // /v1/* requires the key when one is set; / (the playground asks for it) and /health stay open. Runs after routing,
    // before the body is read.
    if (!opts.api_key.empty()) {
        svr.set_pre_request_handler([&](const httplib::Request & req, httplib::Response & res) {
            if (req.path.rfind("/v1/", 0) != 0 || same_key(req.get_header_value("Authorization"), "Bearer " + opts.api_key) ||
                same_key(req.get_header_value("X-API-Key"), opts.api_key)) {
                return httplib::Server::HandlerResponse::Unhandled;
            }
            res.set_header("WWW-Authenticate", "Bearer");
            reply_error(res, 401, "missing or invalid API key (send Authorization: Bearer <key> or X-API-Key: <key>)");
            return httplib::Server::HandlerResponse::Handled;
        });
    }
    // JSON bodies for httplib's own errors (404, 413, ...); a known path with another method is a 405
    const std::map<std::string, std::string> allowed = { { "/", "GET, HEAD" }, { "/health", "GET, HEAD" }, { "/v1/model", "GET, HEAD" },
                                                         { "/v1/predict", "POST" }, { "/v1/decide", "POST" }, { "/v1/logits", "POST" },
                                                         { "/v1/tokenize", "POST" } };
    svr.set_error_handler([&](const httplib::Request & req, httplib::Response & res) {
        if (!res.body.empty()) return httplib::Server::HandlerResponse::Unhandled; // already a JSON error
        std::string message = httplib::status_message(res.status);
        if (res.status == 404) {
            const auto it = allowed.find(req.path);
            if (it != allowed.end()) {
                res.status = 405;
                res.set_header("Allow", it->second);
                message = req.method + " not allowed on " + req.path + " (allowed: " + it->second + ")";
            } else {
                message = "not found: " + req.path;
            }
        } else if (res.status == 413) {
            message = "request body larger than " + std::to_string(opts.max_body_mb) + " MB (--max-body-mb)";
        }
        reply_error(res, res.status, message);
        return httplib::Server::HandlerResponse::Handled;
    });
    svr.set_exception_handler([](const httplib::Request &, httplib::Response & res, std::exception_ptr ep) {
        std::string message = "internal error";
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception & e) {
            message += std::string(": ") + e.what();
        } catch (...) {
        }
        reply_error(res, 500, message);
    });
    if (!opts.quiet) {
        svr.set_logger([](const httplib::Request & req, const httplib::Response & res) {
            fprintf(stderr, "julia1-cli: %s %s %d rows=%zu batch=%zu device=%s prepare=%.3f queue=%.3f forward=%.3f ms=%.3f\n",
                    req.method.c_str(), req.path.c_str(), res.status, t_log.rows, t_log.batch,
                    t_log.device.empty() ? "-" : t_log.device.c_str(), t_log.prepare_ms, t_log.queue_ms, t_log.forward_ms,
                    ms_between(req.start_time_, Clock::now()));
            t_log = {};
        });
    }

    const int port = opts.port == 0 ? svr.bind_to_any_port(opts.host) : svr.bind_to_port(opts.host, opts.port) ? opts.port : -1;
    if (port <= 0) throw std::runtime_error("cannot listen on " + opts.host + ":" + std::to_string(opts.port));
    g_server = &svr;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    fprintf(stderr, "julia1-cli: serving on http://%s:%d (max batch %d, batch wait %g ms, %d http threads%s)\n", opts.host.c_str(), port,
            opts.max_batch, opts.batch_wait_ms, opts.http_threads, opts.api_key.empty() ? "" : ", API key required");
    // returns after stop() (true), once the HTTP threads have answered the requests in flight, or when accept fails
    const bool stopped = svr.listen_after_bind();
    g_server = nullptr;
    if (!stopped) throw std::runtime_error("accepting connections failed");
    fprintf(stderr, "julia1-cli: serve: stopped\n");
    return 0;
}
