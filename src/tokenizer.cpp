#include "tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <queue>
#include <stdexcept>
#include <thread>

namespace {

const char META[] = "\xE2\x96\x81"; // U+2581 '▁'

// length of the UTF-8 sequence starting with lead byte c (invalid leads count as 1)
size_t utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

uint32_t utf8_decode(const std::string & s, size_t pos, size_t len) {
    unsigned char c = (unsigned char) s[pos];
    if (len == 1) return c;
    uint32_t cp = len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
    for (size_t i = 1; i < len && pos + i < s.size(); ++i) {
        cp = (cp << 6) | ((unsigned char) s[pos + i] & 0x3F);
    }
    return cp;
}

// Rust char::is_whitespace (Unicode White_Space)
bool is_unicode_space(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000;
}

} // namespace

void Tokenizer::init(std::vector<std::string> tokens, const std::vector<std::string> & merges,
                     const std::vector<std::string> & added_tokens, const std::vector<int32_t> & added_ids,
                     int32_t unk_id, int32_t mask_id) {
    tokens_ = std::move(tokens);
    unk_id_ = unk_id;
    mask_id_ = mask_id;
    vocab_.reserve(tokens_.size() * 2);
    for (size_t i = 0; i < tokens_.size(); ++i) {
        vocab_.emplace(tokens_[i], (int32_t) i);
    }
    // merges: resolving the ~580k "left right" entries (three vocab lookups each) dominated the model load, so it runs
    // on several threads over contiguous rank ranges; the map is then filled in rank order. A bad entry fails with the
    // lowest such rank, like a serial pass.
    struct Resolved { uint64_t key; int32_t merged; };
    std::vector<Resolved> resolved(merges.size());
    const size_t n_workers = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    std::vector<size_t> first_bad(n_workers, merges.size());
    const auto resolve = [&](size_t w) {
        std::string joined;
        for (size_t rank = merges.size() * w / n_workers, end = merges.size() * (w + 1) / n_workers; rank < end; ++rank) {
            const std::string & m = merges[rank];
            const size_t sp = m.find(' ');
            if (sp == std::string::npos) { first_bad[w] = rank; return; }
            joined.assign(m, 0, sp).append(m, sp + 1);
            auto l = vocab_.find(m.substr(0, sp)), r = vocab_.find(m.substr(sp + 1)), n = vocab_.find(joined);
            if (l == vocab_.end() || r == vocab_.end() || n == vocab_.end()) { first_bad[w] = rank; return; }
            resolved[rank] = { ((uint64_t) (uint32_t) l->second << 32) | (uint32_t) r->second, n->second };
        }
    };
    std::vector<std::thread> workers;
    for (size_t w = 1; w < n_workers; ++w) workers.emplace_back(resolve, w);
    resolve(0);
    for (std::thread & t : workers) t.join();
    const size_t bad = *std::min_element(first_bad.begin(), first_bad.end());
    if (bad < merges.size()) {
        const std::string & m = merges[bad];
        throw std::runtime_error((m.find(' ') == std::string::npos ? "bad merge entry: " : "merge references unknown token: ") + m);
    }
    merges_.reserve(merges.size() * 2);
    for (size_t rank = 0; rank < merges.size(); ++rank) {
        merges_[resolved[rank].key] = { (int32_t) rank, resolved[rank].merged }; // last duplicate wins, like HF's HashMap collect
    }
    if (added_tokens.size() != added_ids.size()) throw std::runtime_error("added token arrays differ in length");
    for (size_t i = 0; i < added_tokens.size(); ++i) {
        if (added_tokens[i].empty()) continue;
        added_.push_back({ added_tokens[i], added_ids[i] });
    }
    std::stable_sort(added_.begin(), added_.end(),
                     [](const Added & a, const Added & b) { return a.text.size() > b.text.size(); });
    added_by_first_.assign(256, {});
    for (size_t i = 0; i < added_.size(); ++i) {
        added_by_first_[(unsigned char) added_[i].text[0]].push_back((int) i);
    }
    for (int b = 0; b < 256; ++b) {
        char name[8];
        snprintf(name, sizeof(name), "<0x%02X>", b);
        auto it = vocab_.find(name);
        byte_ids_[b] = it == vocab_.end() ? -1 : it->second;
    }
}

std::vector<int32_t> Tokenizer::encode(const std::string & text) const {
    std::vector<int32_t> out;
    const size_t n = text.size();
    size_t gap_start = 0;
    size_t i = 0;
    while (i < n) {
        // leftmost-longest added-token match starting at byte i
        const Added * best = nullptr;
        for (int idx : added_by_first_[(unsigned char) text[i]]) {
            const Added & a = added_[idx];
            if (a.text.size() <= n - i && text.compare(i, a.text.size(), a.text) == 0) {
                best = &a; // added_ is sorted longest first
                break;
            }
        }
        if (!best) {
            ++i;
            continue;
        }
        size_t start = i;
        if (best->id == mask_id_) {
            // <mask> is lstrip=true: the match absorbs preceding whitespace (within the current gap)
            while (start > gap_start) {
                size_t p = start; // back up to the lead byte of the previous code point
                do { --p; } while (p > gap_start && ((unsigned char) text[p] & 0xC0) == 0x80);
                if (!is_unicode_space(utf8_decode(text, p, start - p))) break;
                start = p;
            }
        }
        encode_gap(text.substr(gap_start, start - gap_start), out);
        out.push_back(best->id);
        i += best->text.size();
        gap_start = i;
    }
    encode_gap(text.substr(gap_start), out);
    return out;
}

void Tokenizer::encode_gap(const std::string & gap, std::vector<int32_t> & out) const {
    if (gap.empty()) return; // HF drops empty pieces before the pre-tokenizer runs
    std::string norm;
    norm.reserve(gap.size() + 3);
    if (gap.compare(0, 3, META) != 0 && gap[0] != ' ') norm += META; // prepend_scheme=always
    for (char c : gap) {
        if (c == ' ') norm += META; else norm += c;
    }
    // split so that every '▁' starts a new chunk (MergedWithNext)
    size_t chunk_start = 0;
    for (size_t p = 3; p + 2 < norm.size(); ++p) {
        if (norm.compare(p, 3, META) == 0) {
            bpe(norm.substr(chunk_start, p - chunk_start), out);
            chunk_start = p;
            p += 2;
        }
    }
    bpe(norm.substr(chunk_start), out);
}

void Tokenizer::bpe(const std::string & chunk, std::vector<int32_t> & out) const {
    struct Sym { int32_t id; int prev; int next; int len; };
    std::vector<Sym> syms;
    syms.reserve(chunk.size());
    auto add = [&](int32_t id, int len) {
        int idx = (int) syms.size();
        syms.push_back({ id, idx - 1, idx + 1, len });
    };

    // symbols: code points, with byte fallback for code points absent from the vocab (HF merge_word)
    bool have_unk = false;
    int unk_len = 0;
    for (size_t p = 0; p < chunk.size();) {
        size_t len = std::min(utf8_len((unsigned char) chunk[p]), chunk.size() - p);
        auto it = vocab_.find(chunk.substr(p, len));
        if (it != vocab_.end()) {
            if (have_unk) { add(unk_id_, unk_len); have_unk = false; }
            add(it->second, (int) len);
        } else {
            bool all_bytes = true;
            for (size_t b = 0; b < len; ++b) if (byte_ids_[(unsigned char) chunk[p + b]] < 0) all_bytes = false;
            if (all_bytes) {
                for (size_t b = 0; b < len; ++b) add(byte_ids_[(unsigned char) chunk[p + b]], 1);
            } else if (have_unk) {
                unk_len += (int) len; // fuse_unk
            } else {
                have_unk = true;
                unk_len = (int) len;
            }
        }
        p += len;
    }
    if (have_unk) add(unk_id_, unk_len);
    if (syms.empty()) return;
    syms.back().next = -1;

    // HF Word::merge_all: lowest rank first, ties by leftmost position
    struct Merge { int pos; int32_t rank; int32_t new_id; };
    auto worse = [](const Merge & a, const Merge & b) {
        return a.rank != b.rank ? a.rank > b.rank : a.pos > b.pos;
    };
    std::priority_queue<Merge, std::vector<Merge>, decltype(worse)> queue(worse);
    auto lookup = [&](int32_t l, int32_t r) {
        auto it = merges_.find(((uint64_t) (uint32_t) l << 32) | (uint32_t) r);
        return it == merges_.end() ? nullptr : &it->second;
    };
    for (size_t i = 0; i + 1 < syms.size(); ++i) {
        if (auto m = lookup(syms[i].id, syms[i + 1].id)) queue.push({ (int) i, m->first, m->second });
    }
    while (!queue.empty()) {
        Merge top = queue.top();
        queue.pop();
        Sym & cur = syms[top.pos];
        if (cur.len == 0 || cur.next == -1) continue;
        Sym & right = syms[cur.next];
        auto m = lookup(cur.id, right.id);
        if (!m || m->second != top.new_id) continue; // expired entry
        cur.id = top.new_id;
        cur.len += right.len;
        right.len = 0;
        cur.next = right.next;
        if (right.next != -1) syms[right.next].prev = top.pos;
        if (cur.prev >= 0) {
            if (auto pm = lookup(syms[cur.prev].id, cur.id)) queue.push({ cur.prev, pm->first, pm->second });
        }
        if (cur.next != -1) {
            if (auto nm = lookup(cur.id, syms[cur.next].id)) queue.push({ top.pos, nm->first, nm->second });
        }
    }
    for (const Sym & s : syms) {
        if (s.len != 0) out.push_back(s.id);
    }
}
