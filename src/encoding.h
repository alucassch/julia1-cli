// SPEC §5: request validation, JSON-state serialisation (Python json.dumps semantics) and sequence().
#pragma once

#include "gguf_model.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>

struct Request {
    std::string state;   // already serialised (string states verbatim, JSON states via python_json_dumps)
    std::string question;
    std::vector<std::string> options;
    std::string type;    // choice | score | noul
};

struct Encoded {
    std::vector<int32_t> ids;
    std::vector<int32_t> markers;
    int32_t qtype = 0;
};

// nlohmann::ordered_json::parse, except that integer literals outside int64/uint64 (which nlohmann would
// round to double) are kept verbatim as a binary value, so python_json_dumps reproduces Python's exact int.
nlohmann::ordered_json parse_json_line(const std::string & line);

// Python json.dumps(value, ensure_ascii=False) with the default separators (", ", ": ").
std::string python_json_dumps(const nlohmann::ordered_json & value);

// Parses and validates one request object (upstream julia/data.py::validate_row). Throws std::runtime_error.
Request parse_request(const nlohmann::ordered_json & row);

// upstream julia/data.py::sequence. Throws std::runtime_error on rejection.
Encoded sequence(const JuliaModel & model, const Request & row, int max_length, int head_length, bool strict);
