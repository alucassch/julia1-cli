#include "encoding.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <stdexcept>
#include <utility>

using nlohmann::ordered_json;

namespace {

// json_sax_dom_parser with one difference: an integer literal that overflows int64/uint64 (which the
// lexer hands to number_float together with its raw text) is stored as a binary value holding that text.
// JSON text can never produce a binary value otherwise, so dump() can emit it verbatim like Python does.
struct RawIntegerSax : nlohmann::json_sax<ordered_json> {
    ordered_json & root;
    std::vector<ordered_json *> stack;
    ordered_json * slot = nullptr;

    explicit RawIntegerSax(ordered_json & r) : root(r) {}

    ordered_json * put(ordered_json && v) {
        if (stack.empty()) { root = std::move(v); return &root; }
        if (stack.back()->is_array()) { stack.back()->push_back(std::move(v)); return &stack.back()->back(); }
        *slot = std::move(v);
        return slot;
    }
    bool null() override { put(nullptr); return true; }
    bool boolean(bool v) override { put(v); return true; }
    bool number_integer(number_integer_t v) override { put(v); return true; }
    bool number_unsigned(number_unsigned_t v) override { put(v); return true; }
    bool number_float(number_float_t v, const string_t & text) override {
        if (text.find_first_of(".eE") == std::string::npos) put(ordered_json::binary(std::vector<uint8_t>(text.begin(), text.end())));
        else put(v);
        return true;
    }
    bool string(string_t & v) override { put(v); return true; }
    bool binary(binary_t & v) override { put(std::move(v)); return true; }
    bool start_object(std::size_t) override { stack.push_back(put(ordered_json(ordered_json::value_t::object))); return true; }
    bool key(string_t & k) override { slot = &(*stack.back())[k]; return true; } // duplicate key: last value wins, first position kept
    bool end_object() override { stack.pop_back(); return true; }
    bool start_array(std::size_t) override { stack.push_back(put(ordered_json(ordered_json::value_t::array))); return true; }
    bool end_array() override { stack.pop_back(); return true; }
    bool parse_error(std::size_t, const std::string &, const nlohmann::detail::exception & ex) override {
        throw std::runtime_error(ex.what());
    }
};

void dump_string(const std::string & s, std::string & out) {
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char) c; // non-ASCII passes through (ensure_ascii=False)
                }
        }
    }
    out += '"';
}

// Python repr(float): shortest round-trip digits; exponent form iff decpt <= -4 or decpt > 16.
void dump_double(double v, std::string & out) {
    if (v != v) { out += "NaN"; return; }
    if (v == 1.0 / 0.0) { out += "Infinity"; return; }
    if (v == -1.0 / 0.0) { out += "-Infinity"; return; }
    char buf[64];
    auto r = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::scientific);
    std::string sci(buf, r.ptr); // e.g. "-3.837288e+05", "1e+16"
    size_t p = 0;
    if (sci[p] == '-') { out += '-'; ++p; }
    size_t e = sci.find('e', p);
    std::string digits;
    for (size_t i = p; i < e; ++i) if (sci[i] != '.') digits += sci[i];
    int exp10 = std::atoi(sci.c_str() + e + 1);
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    const int decpt = exp10 + 1;
    const int n = (int) digits.size();
    if (decpt <= -4 || decpt > 16) {
        out += digits[0];
        if (n > 1) { out += '.'; out += digits.substr(1); }
        snprintf(buf, sizeof(buf), "e%c%02d", exp10 < 0 ? '-' : '+', std::abs(exp10));
        out += buf;
    } else if (decpt <= 0) {
        out += "0.";
        out.append(-decpt, '0');
        out += digits;
    } else if (decpt >= n) {
        out += digits;
        out.append(decpt - n, '0');
        out += ".0";
    } else {
        out += digits.substr(0, decpt);
        out += '.';
        out += digits.substr(decpt);
    }
}

void dump(const ordered_json & v, std::string & out) {
    switch (v.type()) {
        case ordered_json::value_t::null: out += "null"; break;
        case ordered_json::value_t::boolean: out += v.get<bool>() ? "true" : "false"; break;
        case ordered_json::value_t::number_integer: out += std::to_string(v.get<int64_t>()); break;
        case ordered_json::value_t::number_unsigned: out += std::to_string(v.get<uint64_t>()); break;
        case ordered_json::value_t::number_float: dump_double(v.get<double>(), out); break;
        case ordered_json::value_t::string: dump_string(v.get_ref<const std::string &>(), out); break;
        case ordered_json::value_t::binary: { const auto & b = v.get_binary(); out.append(b.begin(), b.end()); break; } // verbatim big integer
        case ordered_json::value_t::array: {
            out += '[';
            bool first = true;
            for (const auto & x : v) {
                if (!first) out += ", ";
                first = false;
                dump(x, out);
            }
            out += ']';
            break;
        }
        case ordered_json::value_t::object: {
            out += '{';
            bool first = true;
            for (auto it = v.begin(); it != v.end(); ++it) {
                if (!first) out += ", ";
                first = false;
                dump_string(it.key(), out);
                out += ": ";
                dump(it.value(), out);
            }
            out += '}';
            break;
        }
        default: throw std::runtime_error("unsupported JSON value in state");
    }
}

std::string replace_all(std::string s, const std::string & from, const std::string & to) {
    if (from.empty()) return s;
    for (size_t p = 0; (p = s.find(from, p)) != std::string::npos; p += to.size()) s.replace(p, from.size(), to);
    return s;
}

} // namespace

std::string python_json_dumps(const ordered_json & value) {
    std::string out;
    dump(value, out);
    return out;
}

ordered_json parse_json_line(const std::string & line) {
    ordered_json value;
    RawIntegerSax sax(value);
    ordered_json::sax_parse(line, &sax);
    return value;
}

Request parse_request(const ordered_json & row) {
    if (!row.is_object()) throw std::runtime_error("request must be a JSON object");
    Request r;
    const auto state = row.find("state");
    const auto question = row.find("question");
    if (state == row.end() || !(state->is_string() || state->is_object() || state->is_array()) ||
        question == row.end() || !question->is_string()) {
        throw std::runtime_error("state must be text/JSON and question must be text");
    }
    r.state = state->is_string() ? state->get<std::string>() : python_json_dumps(*state);
    r.question = question->get<std::string>();
    const auto options = row.find("options");
    if (options == row.end() || !options->is_array() || options->size() < 2 || options->size() > 20) {
        throw std::runtime_error("options must contain 2\xE2\x80\x93" "20 nonempty rendered descriptions");
    }
    for (const auto & o : *options) {
        if (!o.is_string() || o.get_ref<const std::string &>().empty()) {
            throw std::runtime_error("options must contain 2\xE2\x80\x93" "20 nonempty rendered descriptions");
        }
        r.options.push_back(o.get<std::string>());
    }
    const auto type = row.find("type");
    r.type = type == row.end() ? "choice" : (type->is_string() ? type->get<std::string>() : "");
    if (r.type != "choice" && r.type != "score" && r.type != "noul") throw std::runtime_error("type must be choice, score, or noul");
    if (r.type == "noul" && r.options.size() != 2) throw std::runtime_error("noul options must be ordered [false, true]");
    // training-only fields that upstream validate_row still checks at inference
    const auto target = row.find("target");
    if (target != row.end()) {
        bool ok = target->is_number_integer(); // int64 or uint64 (a bool or float is not an int in Python either)
        if (ok) ok = target->is_number_unsigned() ? target->get<uint64_t>() < r.options.size()
                                                  : target->get<int64_t>() >= 0 && (uint64_t) target->get<int64_t>() < r.options.size();
        if (!ok) throw std::runtime_error("target must index the supplied option list");
    }
    const auto teacher = row.find("teacher_logits");
    if (teacher != row.end() && !teacher->is_null()) {
        bool ok = teacher->is_array() && teacher->size() == r.options.size();
        if (ok) for (const auto & x : *teacher) ok = ok && (x.is_number() || x.is_binary()); // parsed numbers are always finite
        if (!ok) throw std::runtime_error("teacher logits must be finite and match option count/order");
    }
    return r;
}

Encoded sequence(const JuliaModel & m, const Request & row, int max_length, int head_length, bool strict) {
    if (head_length + 4 >= max_length) throw std::runtime_error("max_length must leave room beyond the question head");
    const std::string & mask = m.marker_text;
    auto contains = [&](const std::string & s) { return s.find(mask) != std::string::npos; };
    if (strict && (contains(row.state) || contains(row.question) || std::any_of(row.options.begin(), row.options.end(), contains))) {
        throw std::runtime_error("Reserved model marker in request");
    }
    auto clean = [&](const std::string & s) { return replace_all(s, mask, " "); };
    auto encode = [&](const std::string & s) { return m.tokenizer.encode(s); };

    std::string head_text = replace_all(replace_all(m.head_template, "{type}", row.type), "{question}", clean(row.question));
    std::vector<int32_t> head = encode(head_text);
    std::vector<std::vector<int32_t>> option_ids;
    for (const std::string & o : row.options) option_ids.push_back(encode(m.option_prefix + clean(o)));
    const size_t limit = (size_t) m.option_token_limit;
    if (strict && std::any_of(option_ids.begin(), option_ids.end(), [&](const auto & x) { return x.size() > limit; })) {
        throw std::runtime_error("Option exceeds " + std::to_string(limit) + "-token model contract");
    }
    std::vector<std::vector<int32_t>> options;
    for (const auto & x : option_ids) {
        std::vector<int32_t> o{ m.marker_id };
        o.insert(o.end(), x.begin(), x.begin() + std::min(x.size(), limit));
        options.push_back(std::move(o));
    }
    auto total = [&]() { int64_t s = 0; for (const auto & o : options) s += (int64_t) o.size(); return s; };
    int64_t budget = head_length - total();
    if (budget < 16) {
        int64_t per_option = std::max<int64_t>(4, (int64_t) ((head_length - 16) / (int) options.size()));
        for (auto & o : options) if ((int64_t) o.size() > per_option) o.resize((size_t) per_option);
        budget = head_length - total();
    }
    if (strict) {
        bool truncated = false;
        for (size_t i = 0; i < options.size(); ++i) truncated |= options[i].size() != option_ids[i].size() + 1;
        if ((int64_t) head.size() > budget || truncated) throw std::runtime_error("Question/options exceed lossless head budget");
    }
    Encoded e;
    e.ids.push_back(m.cls_id);
    const size_t keep = (size_t) std::max<int64_t>(8, budget);
    e.ids.insert(e.ids.end(), head.begin(), head.begin() + std::min(head.size(), keep));
    e.ids.push_back(m.sep_id);
    for (const auto & o : options) {
        e.markers.push_back((int32_t) e.ids.size());
        e.ids.insert(e.ids.end(), o.begin(), o.end());
    }
    e.ids.push_back(m.sep_id);
    std::vector<int32_t> state_ids = encode(clean(row.state));
    const int64_t room = (int64_t) max_length - (int64_t) e.ids.size() - 1;
    if (room < 1) throw std::runtime_error("Question/options exceed sequence budget; shorten descriptions");
    if (strict && (int64_t) state_ids.size() > room) throw std::runtime_error("Game state exceeds lossless context budget");
    e.ids.insert(e.ids.end(), state_ids.begin(), state_ids.begin() + std::min<int64_t>((int64_t) state_ids.size(), room));
    e.ids.push_back(m.sep_id);
    e.qtype = row.type == "choice" ? 0 : row.type == "score" ? 1 : 2;
    return e;
}
