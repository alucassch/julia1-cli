# Changelog

The version is set once in `CMakeLists.txt` (`project(julia1-gguf VERSION …)`), which feeds `julia1-cli --version`,
the C API's `julia1_version()` and `info()["version"]`, and is repeated in `pyproject.toml` and
`julia1_gguf/__init__.py`; `tools/package_release.sh` stops if they disagree. It is the version of the runtime and
tools; the GGUF files carry their own provenance (`manifest.json` of the model repository).

## [0.2.0] — 2026-10-02

julia1-cli loads the GGUF layout of llama.cpp ([PR #29818](https://github.com/ggml-org/llama.cpp/pull/29818), merged
2026-10-02), and the model files are now in that layout: the same files run in julia1-cli and in llama.cpp master.

### Changed

- **Model files**: [andrelucas/Julia-1-GGUF](https://huggingface.co/andrelucas/Julia-1-GGUF) holds `Julia-1-F32.gguf`
  and `Julia-1-F16.gguf` (same names) in llama.cpp's layout: architecture `modern-bert` with `modern-bert.decision.*`
  keys and Julia-1's `tokenizer.chat_template.systemone`, written by llama.cpp's `convert_hf_to_gguf.py`. They also run
  in `llama-server` (`POST /v1/systemone`). The encoder-only, `laya` (llama.cpp PR #29363) and `F16-embdQ8_0` files of
  0.1.0 are no longer published.
- **Loader** (`src/gguf_model.cpp`, used by julia1-cli and the C API, and `julia1_gguf/reader.py` for the numpy
  backend): the layout follows `general.architecture`. A `modern-bert` file must carry the decision keys
  (`modern-bert.decision.type` = `laya`) and Julia-1's template byte for byte; the decision blocks `blk.22`/`blk.23`,
  `cls.*` and `token_types.weight` are the decision head and scorer, and the encoding ids and added tokens come from the
  tokenizer keys ([SPEC.md](SPEC.md) §2–§3). The `julia1` layout of the 0.1.0 files still loads. Encoder-only files,
  `laya`-architecture files and models with another template (Laya) are rejected with a message.
- In llama.cpp's layout the head and scorer matrices may be stored in the file's type (F16, BF16, Q8_0); in the `julia1`
  layout they stay F32. The numpy backend accepts `cls.output.weight` stored 1-D, as `llama-quantize` writes it.
- [SPEC.md](SPEC.md) documents llama.cpp's layout as the primary file format and `julia1` as legacy;
  `tools/convert_julia1_to_gguf.py` still writes the `julia1` layout. [docs/BENCHMARKS.md](docs/BENCHMARKS.md) §F
  compares llama.cpp master's `/v1/systemone` with the reference and with `julia1-cli serve`.

The runtime still builds against the ggml of llama.cpp v0.5.0 with `patches/ggml-julia1.patch`.

### Results

The same F32 weights give byte-identical output in both layouts (`decide` on Metal, exact and `--fast --batch 16`, the C
API on the CPU, and the numpy backend): typed set 2000/2000, accuracy 1451, max abs logit 8.24e-4 (Metal exact), 0
encoding mismatches; parity set 100/100 on the CPU, max abs 1.57e-4. The tokenizer matches HF `tokenizers` on the 4595
strings in both layouts. The 0.1.0 results below and in docs/BENCHMARKS.md hold for the F32 file.

## [0.1.0] — 2026-09-29

First release: a GGUF runtime for [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1), a decision
model (state + question + type `choice`/`score`/`noul` + 2–20 options → one logit per option), not a generative model.

### Contents

- **GGUF format** `julia1` for the full model ([SPEC.md](SPEC.md)) and the converter
  `tools/convert_julia1_to_gguf.py` (F32, F16, BF16, Q8_0, Q5_0, Q4_0, per-tensor overrides, encoder-only files for
  stock llama.cpp). Model files: [andrelucas/Julia-1-GGUF](https://huggingface.co/andrelucas/Julia-1-GGUF).
- **`julia1-cli`** (C++17/ggml, Metal and CPU): `replay`, `decide`, `predict` (upstream's named-question API and the
  legacy list API), `tokenize`, and `serve` (HTTP API with dynamic batching, a playground page, and
  `--devices metal,cpu` for a CPU helper engine; [docs/server.md](docs/server.md)). `--precise` (default) runs exact
  F32 Metal kernels; `--fast` uses half-precision operands; `--batch B` groups rows into padded graphs.
- **ggml patch** `patches/ggml-julia1.patch` (MIT), applied at build time: exact float-staged Metal GEMM and flash
  attention, Metal GEMM epilogue fusion (residual add, GEGLU, bias and ReLU, NEOX RoPE), smaller GEMM tiles for small
  grids, cheaper flash-attention padding, and on the CPU a vectorised GEGLU, flash-attention tiling and F32 matmuls
  through Accelerate.
- **C API** (`include/julia1_gguf.h`) and the shared library `libjulia1_gguf_native`: JSON in, JSON out, the same
  results as julia1-cli byte for byte.
- **Python package `julia1-gguf`** (`import julia1_gguf`): `load_model(...)` with a native backend (the C++ runtime
  through ctypes, the default when its library is installed) and a pure numpy backend, mirroring upstream's `predict`
  and `logits`; `pip install .` builds the library.
- **Build and packaging**: CMake ≥ 3.18 downloads llama.cpp v0.5.0 (SHA-256 pinned) and patches a copy of its ggml in
  the build directory; `cmake --install` lays out `bin/`, `lib/`, `include/`, `share/julia1-cli/`;
  `tools/package_release.sh` can build, check and package a self-contained macOS archive and a wheel locally (no
  prebuilt binaries are published); GitHub Actions CI (macOS arm64; Linux CPU build, experimental).
- **Validation**: reference logits of upstream PyTorch (`data/reference/`), `tools/eval_gguf.py`,
  `tools/compare_replay.py` and the tokenizer parity tools.

### Results (Apple M1 Pro; the 2000 typed questions, against upstream PyTorch CPU FP32, accuracy 1451)

| runtime | decisions equal to the reference | accuracy | max abs logit | ms per request | × PyTorch CPU |
| --- | --- | --- | --- | --- | --- |
| Metal, exact F32 (default), F32 file | 2000/2000 | 1451 | 8.24e-4 | 12.3 | 4.9× |
| Metal `--fast`, F16 file | 2000/2000 | 1451 | 0.561 | 11.4 | 5.3× |
| CPU, F32 file, 4 threads | 2000/2000 | 1451 | 8.43e-4 | 41.9 | 1.43× |

The 100 parity cases give 100/100 on all three; the tokenizer matches HF `tokenizers` on 4595 strings. Details:
[docs/BENCHMARKS.md](docs/BENCHMARKS.md).

### Known limitations

- Tested on macOS on Apple silicon only. Linux builds from source and is experimental; no CUDA.
- `--fast` trades exactness for speed; encoder matrices below F16 (Q8_0, Q5_0, Q4_0) change decisions.
- Batching pads grouped rows: logits stay within the guards, but `predict` probabilities can differ from batch 1 in the
  last digits.
- ggml compiles its Metal shaders at process start; when macOS's shader cache misses, the start takes about 20 s.
