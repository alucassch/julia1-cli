// Loads Julia-1-<TYPE>.gguf (SPEC §2): metadata, tokenizer and the 165 tensors into a backend buffer.
#pragma once

#include "tokenizer.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <string>
#include <utility>
#include <vector>

struct EncoderLayer {
    ggml_tensor * attn_norm = nullptr; // null for layer 0
    ggml_tensor * wqkv = nullptr;
    ggml_tensor * wo = nullptr;
    ggml_tensor * ffn_norm = nullptr;
    ggml_tensor * ffn_up = nullptr;
    ggml_tensor * ffn_down = nullptr;
};

struct HeadLayer {
    ggml_tensor * attn_norm_w, * attn_norm_b;
    ggml_tensor * wqkv, * bqkv;
    ggml_tensor * wo, * bo;
    ggml_tensor * ffn_norm_w, * ffn_norm_b;
    ggml_tensor * ffn_up, * ffn_up_b;
    ggml_tensor * ffn_down, * ffn_down_b;
};

struct JuliaModel {
    // hparams (julia1.*)
    int32_t n_ctx = 0, n_embd = 0, n_layer = 0, n_ff = 0, n_head = 0, n_rot = 0, n_vocab = 0;
    int32_t swa = 0, swa_pattern = 0;
    float eps = 1e-5f, rope_base = 0.f;
    int32_t head_layers = 0, head_n_head = 0, head_ff = 0;
    float head_eps = 1e-5f;
    // encoding (julia1.encoding.*)
    int32_t cls_id = 2, sep_id = 1, marker_id = 4;
    std::string marker_text, head_template, option_prefix;
    int32_t option_token_limit = 48, min_options = 2, max_options = 20;
    int32_t default_max_length = 8192, default_head_length = 512;
    std::string name, file_type;

    ggml_tensor * tok_embd = nullptr;
    ggml_tensor * tok_norm = nullptr;
    ggml_tensor * out_norm = nullptr;
    std::vector<EncoderLayer> layers;
    ggml_tensor * type_embd = nullptr;
    std::vector<HeadLayer> head;
    ggml_tensor * sc_norm_w, * sc_norm_b, * sc_up_w, * sc_up_b, * sc_out_w, * sc_out_b;

    Tokenizer tokenizer;

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    void * map_addr = nullptr; // read-only mapping of the file when buf wraps the tensor data in place
    size_t map_size = 0;
    size_t data_offset = 0;    // file offset of the tensor data, and each tensor's offset within it (read_model)
    std::vector<std::pair<ggml_tensor *, size_t>> tensor_offsets;

    ~JuliaModel();
};

// Both throw std::runtime_error with a message on any failure.
// read_model: the metadata, the tokenizer and the tensor descriptions, without the weights (needs no backend, so it
// runs while the backends initialise). with_tokenizer = false leaves model.tokenizer empty (replay feeds token ids;
// building it is ~40% of the load).
void read_model(const std::string & path, JuliaModel & model, bool with_tokenizer = true);
// load_weights: the weights, in a buffer of `backend`: the mapped file itself when the backend's device can wrap host
// memory (Metal, CPU, BLAS), else a copy streamed from the file.
void load_weights(const std::string & path, ggml_backend_t backend, JuliaModel & model);
