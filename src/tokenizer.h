// SPEC §4: HF-tokenizers-exact Gemma-style BPE (Replace ' '->'▁', Metaspace always/split, byte fallback).
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class Tokenizer {
public:
    // tokens: id -> text. merges: "left right" strings in rank order.
    // added_tokens / added_ids: the added-token strings and their ids (matched on raw text).
    void init(std::vector<std::string> tokens, const std::vector<std::string> & merges,
              const std::vector<std::string> & added_tokens, const std::vector<int32_t> & added_ids,
              int32_t unk_id, int32_t mask_id);

    // encode(text) -> ids, no CLS/SEP (add_special_tokens=False).
    std::vector<int32_t> encode(const std::string & text) const;

    const std::string & token_text(int32_t id) const { return tokens_[id]; }
    size_t n_vocab() const { return tokens_.size(); }

private:
    struct Added { std::string text; int32_t id; };

    void encode_gap(const std::string & gap, std::vector<int32_t> & out) const;
    void bpe(const std::string & chunk, std::vector<int32_t> & out) const;

    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int32_t> vocab_;
    // (left_id << 32 | right_id) -> (rank, merged_id)
    std::unordered_map<uint64_t, std::pair<int32_t, int32_t>> merges_;
    std::vector<Added> added_;                 // sorted by length, longest first
    std::vector<std::vector<int>> added_by_first_; // indices into added_, keyed by first byte
    int32_t unk_id_ = 3;
    int32_t mask_id_ = 4;
    int32_t byte_ids_[256];                    // id of <0xNN>, or -1
};
