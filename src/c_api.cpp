// C API over Engine (c_api.h): JSON in and out, errors as messages, one mutex per handle.
#include "c_api.h"

#include "encoding.h"
#include "engine.h"

#include <ggml.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

using nlohmann::ordered_json;

namespace {

struct Handle {
    std::unique_ptr<Engine> engine;
    std::mutex mutex;
};

// malloc'd copy, released by julia1_string_free (free)
char * copy_string(const std::string & s) {
    char * out = (char *) std::malloc(s.size() + 1);
    if (out) std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
}

void set_error(char ** error, const char * message) {
    if (error) *error = copy_string(message);
}

// ggml's log without its debug and info messages (Metal device and shader library at load, buffer sizes): warnings and
// errors still go to stderr, as ggml's default logger writes them. GGML_LOG_LEVEL_CONT continues the previous message.
void quiet_log(ggml_log_level level, const char * text, void *) {
    static ggml_log_level last = GGML_LOG_LEVEL_NONE;
    if (level != GGML_LOG_LEVEL_CONT) last = level;
    if (last >= GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
        fflush(stderr);
    }
}

// Once, before the first engine touches ggml: quiet_log unless JULIA1_GGUF_VERBOSE is set (and not 0), which keeps ggml's
// default logger. julia1-cli does not use the C API and keeps ggml's default logger.
void init_log() {
    static std::once_flag once;
    std::call_once(once, [] {
        const char * verbose = std::getenv("JULIA1_GGUF_VERBOSE");
        if (!verbose || !*verbose || std::strcmp(verbose, "0") == 0) ggml_log_set(quiet_log, nullptr);
    });
}

EngineOptions parse_options(const char * options_json) {
    const ordered_json o = parse_json_line(options_json ? options_json : "");
    if (!o.is_object()) throw std::runtime_error("options must be a JSON object");
    EngineOptions opts;
    for (auto it = o.begin(); it != o.end(); ++it) {
        const std::string & key = it.key();
        const ordered_json & v = it.value();
        const auto integer = [&]() {
            if (!v.is_number_integer()) throw std::runtime_error("option " + key + " must be an integer");
            return v.get<int>();
        };
        const auto boolean = [&]() {
            if (!v.is_boolean()) throw std::runtime_error("option " + key + " must be true or false");
            return v.get<bool>();
        };
        const auto string = [&]() {
            if (!v.is_string()) throw std::runtime_error("option " + key + " must be a string");
            return v.get<std::string>();
        };
        if (key == "model") opts.model = string();
        else if (key == "device") opts.device = string();
        else if (key == "threads") opts.threads = integer();
        else if (key == "precise") opts.precise = boolean();
        else if (key == "max_length") {
            // null: the model's context length (upstream max_length=None). Any other value that is not an int is out of
            // range, so that Engine rejects it with upstream's message, which names the context length.
            const bool in_range = v.is_number_integer() && v.get<int64_t>() >= 1 && v.get<int64_t>() <= std::numeric_limits<int>::max();
            opts.max_length = v.is_null() ? std::optional<int>() : in_range ? v.get<int>() : 0;
        }
        else if (key == "head_length") opts.head_length = integer();
        else if (key == "strict") opts.strict = boolean();
        else if (key == "batch") opts.batch = integer();
        else throw std::runtime_error("unknown option " + key);
    }
    if (opts.model.empty()) throw std::runtime_error("option model is required");
    return opts;
}

std::vector<ordered_json> items(const ordered_json & request, const std::string & method) {
    if (!request.is_array()) throw std::runtime_error(method + ": request must be a JSON array");
    return std::vector<ordered_json>(request.begin(), request.end());
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

ordered_json call(Engine & engine, const std::string & method, const char * request_json) {
    if (method == "info") return engine.info();
    const ordered_json request = parse_json_line(request_json ? request_json : "");
    if (method == "logits") return engine.logits(items(request, method));
    if (method == "predict_rows") return engine.predict_rows(items(request, method));
    if (method == "predict_typed") {
        if (!request.is_object()) throw std::runtime_error("predict_typed: request must be a JSON object {\"state\", \"questions\"}");
        const auto get = [&](const char * key) { const auto it = request.find(key); return it == request.end() ? ordered_json() : *it; };
        return engine.predict_typed(get("state"), get("questions"));
    }
    if (method == "replay") {
        std::vector<Encoded> rows;
        for (const ordered_json & row : items(request, method)) {
            if (!row.is_object()) throw std::runtime_error("replay: expected {\"ids\", \"markers\", \"qtype\"} objects");
            Encoded e;
            e.ids = to_i32(row.value("ids", ordered_json::array()), "ids");
            e.markers = to_i32(row.value("markers", ordered_json::array()), "markers");
            e.qtype = row.value("qtype", 0);
            rows.push_back(std::move(e));
        }
        return engine.replay(rows);
    }
    if (method == "encode") {
        ordered_json out = ordered_json::array();
        for (const Encoded & e : engine.encode(items(request, method))) {
            out.push_back({ { "ids", e.ids }, { "markers", e.markers }, { "qtype", e.qtype } });
        }
        return out;
    }
    if (method == "tokenize") {
        std::vector<std::string> texts;
        for (const ordered_json & t : items(request, method)) {
            if (!t.is_string()) throw std::runtime_error("tokenize: texts must be strings");
            texts.push_back(t.get<std::string>());
        }
        return engine.tokenize(texts);
    }
    throw std::runtime_error("unknown method " + method);
}

} // namespace

extern "C" {

void * julia1_engine_new(const char * options_json, char ** error) {
    if (error) *error = nullptr;
    try {
        init_log();
        auto handle = std::make_unique<Handle>();
        handle->engine = std::make_unique<Engine>(parse_options(options_json));
        return handle.release();
    } catch (const std::exception & e) {
        set_error(error, e.what());
    } catch (...) {
        set_error(error, "unknown error");
    }
    return nullptr;
}

char * julia1_engine_call(void * engine, const char * method, const char * request_json, char ** error) {
    if (error) *error = nullptr;
    try {
        if (!engine || !method) throw std::runtime_error("julia1_engine_call: null engine or method");
        Handle & h = *(Handle *) engine;
        std::lock_guard<std::mutex> lock(h.mutex);
        char * out = copy_string(call(*h.engine, method, request_json).dump());
        if (!out) throw std::bad_alloc();
        return out;
    } catch (const std::exception & e) {
        set_error(error, e.what());
    } catch (...) {
        set_error(error, "unknown error");
    }
    return nullptr;
}

void julia1_string_free(char * s) { std::free(s); }

void julia1_engine_free(void * engine) { delete (Handle *) engine; }

const char * julia1_version(void) {
    static const std::string version = engine_version();
    return version.c_str();
}

} // extern "C"
