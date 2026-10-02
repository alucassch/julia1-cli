# julia1-cli

*Português: [README.pt-BR.md](README.pt-BR.md)*

A native C++/ggml runtime (Metal and CPU) for [Julia-1](https://huggingface.co/SupersonicLabs/Julia-1), the decision
model by Supersonic Labs ([official page](https://supersoniclabs.ia.br/pt/julia-1/)), in GGUF. It reproduces the
upstream PyTorch results and ships as a CLI with an HTTP server, a C API and a Python package (`julia1-gguf`).

The GGUF files are on Hugging Face: [andrelucas/Julia-1-GGUF](https://huggingface.co/andrelucas/Julia-1-GGUF).

The names `julia1-cli`, `julia1-gguf` and `julia1` refer to the Julia-1 model; the project is unrelated to the Julia
programming language. It is an independent port, not affiliated with or endorsed by Supersonic Labs.

## Julia-1

Julia-1 is a **decision model, not a generative model**. A request is a `state` (text or JSON), a `question`, a
`type` (`choice`, `score` or `noul`, a Boolean) and 2–20 `options`; the model returns one logit per option, and the
caller takes the softmax and argmax. It is an mmBERT-small encoder (ModernBERT: 22 layers, hidden 384, vocabulary
256 000, context 8192) followed by a typed decision head (a question-type embedding, two transformer layers and a
scorer read at the `<mask>` marker before each option); 144.19 M parameters.

The GGUF files are in llama.cpp's layout for this model (architecture `modern-bert` with two decision blocks, written by
llama.cpp's `convert_hf_to_gguf.py`), so the same files run in julia1-cli and in upstream llama.cpp master
([below](#julia-1-in-llamacpp)). julia1-cli also loads the `julia1` layout of its 0.1.0 files. [SPEC.md](SPEC.md)
defines both layouts, the tokenizer, the request encoding, the forward pass and the validation protocol.

## Contents

| path | what |
| --- | --- |
| `src/` | the runtime: GGUF loader, tokenizer, request encoding, ggml graph, CLI, HTTP server, C API |
| `patches/ggml-julia1.patch` | changes to ggml (MIT): exact F32 Metal kernels, fused Metal GEMM epilogues, CPU speed-ups; applied at build time |
| `julia1_gguf/` | the Python package `julia1-gguf`: the same runtime through the C API, or a pure numpy runtime |
| `tools/convert_julia1_to_gguf.py` | converter from the upstream safetensors checkpoint to the legacy `julia1` layout |
| `tools/eval_gguf.py`, `tools/compare_replay.py`, `tools/tokenizer_parity*.py` | validation against the reference data |
| `tools/systemone_parity.py` | the same 2000 typed questions through an HTTP server (llama-server `/v1/systemone` or `julia1-cli serve`) |
| `data/reference/` | reference logits of upstream PyTorch: 2000 typed-decisions questions and 100 parity cases |
| `data/tokenizer-corpus.jsonl` | 4595 strings for the tokenizer parity check |
| `docs/server.md`, `docs/BENCHMARKS.md` | the HTTP API; benchmark setup and full tables |
| `packaging/`, `tools/package_release.sh` | optional: a self-contained macOS archive and a wheel, built locally |

## Install

No prebuilt binaries are published; build from source.

**From source** (CMake ≥ 3.18, a C++17 compiler, `patch`; configure downloads the llama.cpp v0.5.0 release, SHA-256
pinned, and applies `patches/ggml-julia1.patch` to a copy of its ggml):

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8          # build/julia1-cli and build/libjulia1_gguf_native.dylib (.so on Linux)
```

* Offline, from an unpacked llama.cpp v0.5.0: add `-DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=<absolute path>`.
* `-DJULIA1_GGML_PATCH=OFF` builds against unmodified ggml: results stay exact, `--precise` on Metal is about 4x
  slower.
* macOS: Metal, and Accelerate for the CPU device. Linux: CPU only (llamafile's sgemm; `-DJULIA1_BLAS=ON
  -DGGML_BLAS_VENDOR=OpenBLAS` for OpenBLAS). The Linux build is experimental: CI compiles it (allowed to fail), and
  it has not been validated against the reference data. No CUDA.

**Python package from source** (builds the native library with CMake, as above; numpy and gguf are the only
dependencies):

```sh
pip install .              # or: pip install ".[fast]" for the HF tokenizers used by the numpy backend
```

## Quick start

Download `Julia-1-F32.gguf` from [andrelucas/Julia-1-GGUF](https://huggingface.co/andrelucas/Julia-1-GGUF), then run the
example of upstream's README:

```sh
cat > questions.jsonl <<'JSONL'
{"state": "I was charged twice for the same order.", "questions": {"team": {"type": "choice", "instructions": "Which team should handle this request?", "criteria": {"billing": "Billing and payment disputes", "shipping": "Shipping and delivery", "access": "Account access and login"}}}}
JSONL
build/julia1-cli --model Julia-1-F32.gguf predict --input questions.jsonl --output answers.jsonl --max-length 8192 --head-length 512 --strict
cat answers.jsonl
```

```json
{"answers":{"team":{"type":"choice","probabilities":{"billing":0.8544651362474033,"shipping":0.14277066945266953,"access":0.002764194299927172},"choice":"billing","max_probability":0.8544651362474033}}}
```

(Apple M1 Pro, Metal. On the CPU the same request gives billing 0.8544669661607737: exact F32, different summation
order.)

## CLI

```
julia1-cli --model <gguf> [--device auto|cpu|metal] [--threads N] [--precise|--fast] [--batch B] [--timing] <command> --input <in.jsonl> --output <out.jsonl>
  decide   [--max-length 8192] [--head-length 256] [--strict]   {"state","question","options","type"} -> {"logits","index","probabilities","ids","markers","qtype"}
  predict  [--max-length 8192] [--head-length 256] [--strict]   {"state","questions":{...}} -> {"answers":{...}} (upstream engine.predict)
  replay   {"id","ids","markers","qtype"} -> {"id","logits"}     (an already-encoded sequence)
  tokenize {"text"} -> {"ids"}
julia1-cli --model <gguf> [...] serve [--host 127.0.0.1] [--port 8080] [--api-key KEY] [--devices metal,cpu]
```

* `--device auto` (default) uses Metal when it initialises, else the CPU. `--threads` (default 4) should stay below
  the number of free performance cores.
* `--precise` (default) runs exact F32 kernels on Metal; `--fast` uses half-precision operands (faster; on the F16
  file, logits within 0.6 of the reference and the same 2000/2000 typed decisions). The CPU is always exact.
* `--batch B` groups B rows by length into padded graphs, for throughput.
* `decide` and `predict` default to upstream `julia.load_model`'s encoding (`max_length` 8192, `head_length` 256, not
  strict). Upstream's recommended policy is 8192 / 512 / strict; the evaluation protocol is 1024 / 512 (typed) or
  256 (parity) / strict.

## HTTP server

`julia1-cli serve` keeps the model loaded and batches concurrent requests dynamically
([docs/server.md](docs/server.md)):

| method, path | body | response |
| --- | --- | --- |
| `POST /v1/predict` | `{"state", "questions"}` (upstream `engine.predict`), legacy `{"rows": [...]}`, or an array of them | `{"answers": {...}}` or `{"predictions": [...]}` |
| `POST /v1/decide` | `{"rows": [{state, question, options, type}]}` | `{"results": [{"index", "probabilities", "logits"}]}` |
| `POST /v1/logits` | `{"rows": [...]}` | `{"logits": [[...]]}` |
| `POST /v1/tokenize` | `{"texts": [string]}` | `{"ids": [[...]]}` |
| `GET /health`, `GET /v1/model`, `GET /` | - | status, model info, a playground page |

```sh
build/julia1-cli --model Julia-1-F32.gguf serve            # http://127.0.0.1:8080
curl -s http://127.0.0.1:8080/v1/predict -H 'Content-Type: application/json' -d @questions.jsonl
```

Security: the server binds `127.0.0.1` by default and has no TLS and no rate limit. Set `JULIA1_API_KEY` (or
`--api-key`) to require `Authorization: Bearer <key>` on `/v1/*`, and put a reverse proxy in front before binding
another address.

## Python package

```python
from julia1_gguf import load_model

engine = load_model("Julia-1-F32.gguf", max_length=8192, head_length=512, strict_encoding=True)
result = engine.predict(
    state="I was charged twice for the same order.",
    questions={"team": {"type": "choice", "instructions": "Which team should handle this request?",
                        "criteria": {"billing": "Billing and payment disputes", "shipping": "Shipping and delivery",
                                     "access": "Account access and login"}}},
)
print(result["answers"]["team"]["choice"])  # billing
```

`load_model(path, max_length=None, head_length=256, strict_encoding=False, *, backend="auto", device="auto",
precise=True, threads=4, batch=1)` has upstream's defaults. `backend="native"` runs julia1-cli's runtime through its
C API (the default when the library is installed), `backend="numpy"` a pure numpy runtime. Both offer
`predict(state=..., questions=...)`, the legacy `predict(rows)`, `logits(rows)`, `replay(ids, markers, qtype)`,
`encode(row)` and `tokenize(text)`. `JULIA1_GGUF_LIB` selects another native library, and `JULIA1_GGUF_VERBOSE=1`
shows ggml's full log. `python -m julia1_gguf` (console script `julia1-gguf`) is a CLI for the numpy runtime with
julia1-cli's contract.

## C API

`julia1_gguf.h` (library `libjulia1_gguf_native`): JSON in, JSON out, the same results as julia1-cli.

```c
void * engine = julia1_engine_new("{\"model\": \"Julia-1-F32.gguf\", \"head_length\": 512, \"strict\": true}", &error);
char * result = julia1_engine_call(engine, "predict_typed", "{\"state\": ..., \"questions\": {...}}", &error);
julia1_string_free(result);
julia1_engine_free(engine);
```

Methods: `info`, `logits`, `predict_rows`, `predict_typed`, `replay`, `encode`, `tokenize`. A complete example is in
the [quickstart](packaging/README-quickstart.md).

## Model files

From [andrelucas/Julia-1-GGUF](https://huggingface.co/andrelucas/Julia-1-GGUF), converted with llama.cpp's
`convert_hf_to_gguf.py` ([PR #29818](https://github.com/ggml-org/llama.cpp/pull/29818)); the same files run in
llama.cpp:

| file | size | use |
| --- | --- | --- |
| `Julia-1-F32.gguf` | 592 MB | exact reference |
| `Julia-1-F16.gguf` | 303 MB | half the size; exact kernels or `--fast` |

The `julia1`-layout files of julia1-cli 0.1.0 (the repository's previous revision) still load; its encoder-only,
`laya` and `F16-embdQ8_0` files are no longer published. julia1-cli stops with an error on encoder-only files and on
models of the same family with another prompt template (Laya). BF16 and Q8_0 matrices change decisions (ggml-org's
conversions, [ggml-org/Julia-1-GGUF](https://huggingface.co/ggml-org/Julia-1-GGUF): 1986 and 1946 of 2000 in
julia1-cli) and are not published. To convert the upstream checkpoint yourself, with llama.cpp master:

```sh
python convert_hf_to_gguf.py <SupersonicLabs/Julia-1 download> --outtype f32 --outfile Julia-1-F32.gguf   # or f16
```

`tools/convert_julia1_to_gguf.py` (Python with `numpy`, `safetensors` and `gguf`) still writes the legacy `julia1`
layout, with per-tensor overrides ([SPEC.md](SPEC.md) §2.3).

## Results

Apple M1 Pro (8 performance + 2 efficiency cores, 16 GB), macOS 27. Reference: upstream PyTorch on the CPU in FP32,
which reproduces upstream's published result on the typed-decisions test set (accuracy 1451/2000). Typed set = its
2000 questions (`max_length` 1024, `head_length` 512, strict); latency = median of one request per call on
pre-tokenised input (end to end adds 0.2–0.3 ms); batch 16 = requests/s through each runtime's own batching.

| runtime | ms per request | × PyTorch CPU | requests/s, batch 16 | decisions equal to the reference | accuracy | max abs logit |
| --- | --- | --- | --- | --- | --- | --- |
| julia1-cli, Metal `--fast`, F16 file | 11.4 | 5.3× | 111 | 2000/2000 | 1451 | 0.56 |
| julia1-cli, Metal exact (default), F32 file | 12.3 | 4.9× | 102 | 2000/2000 | 1451 | 8.2e-4 |
| julia1-cli, CPU, F32 file, 4 threads | 41.9 | 1.43× | 32 (6 threads) | 2000/2000 | 1451 | 8.4e-4 |
| MLX fp16 (julia-mlx) | 15.6 | 3.85× | 87 | 1997/2000 | 1452 | 0.48 |
| MLX fp32 (julia-mlx) | 17.2 | 3.50× | 77 | 2000/2000 | 1451 | 8.6e-4 |
| PyTorch MPS (upstream code) | 30.0 | 2.01× | 62 | 2000/2000 | 1451 | 5.0e-4 |
| **PyTorch CPU, 4 threads (upstream)** | 60.1 | 1.00× | 26 | 2000/2000 | 1451 | 2.3e-4 |
| ONNX Runtime CPU, 8 threads (Julia-1-ONNX) | 85.3 | 0.70× | 11 | 2000/2000 | 1451 | 7.6e-4 |

* Measured with julia1-cli 0.1.0 on its `julia1`-layout files; on the same F32 weights, 0.2.0 gives byte-identical
  output from the llama.cpp-layout `Julia-1-F32.gguf`.
* The 100 parity cases give 100/100 decisions on every julia1-cli configuration above; the Python package's native
  backend gives the same logits as julia1-cli and adds 0.0–0.3 ms per call.
* End to end, julia1-cli's tokenizer and encoding produce exactly the reference token ids (0 mismatches on the 2000
  typed and 100 parity requests); the tokenizer matches HF `tokenizers` on all 4595 strings of
  `data/tokenizer-corpus.jsonl`, in both layouts.
* `julia1-cli serve` answers 103 requests/s (exact) and 112 (fast) with 32 concurrent clients, 128 and 136 with
  `--devices metal,cpu`. Process start to the first answer takes about 0.2 s.

Setup, protocol and the full tables (per sequence length, end to end, server, cold start):
[docs/BENCHMARKS.md](docs/BENCHMARKS.md). To check a build against the reference data:

```sh
python tools/eval_gguf.py --runtime cli --cli build/julia1-cli --model Julia-1-F32.gguf --suite typed --mode e2e \
    --min-argmax 2000 --max-abs 1e-3 --accuracy 1451
python tools/eval_gguf.py --runtime cli --cli build/julia1-cli --model Julia-1-F32.gguf --suite parity --mode replay \
    --device cpu --min-argmax 100 --max-abs 2e-3
```

## Julia-1 in llama.cpp

llama.cpp master runs Julia-1 from the same files since 2026-10-02
([PR #29818](https://github.com/ggml-org/llama.cpp/pull/29818); no tagged release contains it yet). `llama-server`
takes the `{"state", "questions"}` body of `julia1-cli serve`'s `/v1/predict` at `POST /v1/systemone`:

```sh
llama-server -m Julia-1-F16.gguf -b 2048 -ub 2048      # -ub >= the longest prompt in tokens (up to 8192)
curl -s http://127.0.0.1:8080/v1/systemone -H 'Content-Type: application/json' -d @questions.jsonl
```

* `-ub` (default 512) must hold the whole prompt: with the defaults, 35 of the 2000 typed questions fail with HTTP 500
  (`input (N tokens) is too large to process`).
* Send a JSON state as a string you serialise yourself (Python `json.dumps`). llama.cpp's `tojson` writes floats with
  6 significant digits and drops `.0` (`532.0` → `532`, `1234567.5` → `1.23457e+06`): with the state sent as an
  object, 480 of the 2000 typed prompts tokenise differently from upstream's, and 51 of the 52 changed decisions come
  from those.
* llama.cpp runs ModernBERT's `gelu` as ggml's tanh-approximation GeGLU, where upstream uses the exact erf; with the
  state sent as a string, this is the main remaining source of error. On the CPU, add `-fa off` (flash attention adds
  a little error there).

The 2000 typed questions, one question per request, sequential HTTP from the same client for both servers (Apple M1
Pro, llama.cpp master `1fb7ef3e`). Logit differences are recovered from the returned probabilities; ms include the
HTTP round trip, so they are not comparable with the table above:

| server, file, device | decisions equal to the reference | median / max abs logit difference | ms per request |
| --- | --- | --- | --- |
| llama-server, F32, Metal | 1999/2000 | 0.037 / 0.40 | 22.8 |
| llama-server, F32, CPU, 6 threads | 1999/2000 | 0.030 / 0.29 | 62.2 |
| llama-server, F16, Metal | 2000/2000 | 0.039 / 0.58 | 21.8 |
| julia1-cli serve, F32, Metal exact | 2000/2000 | 5.0e-5 / 8.5e-4 | 13.4 |
| julia1-cli serve, F16, Metal `--fast` | 2000/2000 | 0.023 / 0.41 | 12.5 |

llama-server ran with `-b 1024 -ub 1024` and the state sent as a string; julia1-cli with its 0.1.0 `julia1`-layout
files. julia1-cli is about 1.7× faster here and exact on Metal. ggml-org's BF16 and Q8_0 files give 1987 and 1947 of
2000 on llama-server. Details and the other settings: [docs/BENCHMARKS.md](docs/BENCHMARKS.md) §F.

## Limitations

* Julia-1 compares the options it is given; it is not a knowledge or reasoning model. Evaluate it on your own
  questions and options before acting on its output.
* Tested on macOS on Apple silicon only. Linux builds from source and is experimental; no CUDA.
* ggml compiles its Metal shaders when a process starts; when macOS's shader cache misses (the first run on a
  machine, or after another ggml build ran), the start takes about 20 s instead of 0.1–0.3 s. `--device cpu` is
  affected too, because every backend is registered at start.
* A build against unmodified ggml (`-DJULIA1_GGML_PATCH=OFF`) is exact but about 4x slower in `--precise` mode on
  Metal.
* Batching pads grouped requests: logits stay within the guards, but probabilities can differ from batch 1 in the
  last digits.

## References

* [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1): the model, weights and reference code by
  Supersonic Labs (Apache-2.0); official page: https://supersoniclabs.ia.br/pt/julia-1/.
* [SupersonicLabs/Julia-1-ONNX](https://huggingface.co/SupersonicLabs/Julia-1-ONNX): the ONNX export by Supersonic
  Labs; its 100 parity cases are part of `data/reference/`.
* [zainmerchan/Julia-1-MLX](https://huggingface.co/zainmerchan/Julia-1-MLX) and the
  [julia-mlx](https://github.com/zm2231/julia-mlx) runtime (MIT): the MLX port, used as a speed and accuracy
  reference.
* [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) (MIT; arXiv:2509.06888): the encoder and
  tokenizer Julia-1 builds on. ModernBERT by Answer.AI and LightOn (arXiv:2412.13663): the encoder architecture.
* [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp) v0.5.0 (MIT): the tensor library, the GGUF format and
  gguf-py; llamafile's sgemm (MIT), [cpp-httplib](https://github.com/yhirose/cpp-httplib) (MIT) and
  [nlohmann/json](https://github.com/nlohmann/json) (MIT) come with it.
* [llama.cpp PR #29818](https://github.com/ggml-org/llama.cpp/pull/29818) (MIT): Julia-1 in llama.cpp (`llama-server`
  `/v1/systemone`) and the `convert_hf_to_gguf.py` that wrote the published files;
  [ggml-org/Julia-1-GGUF](https://huggingface.co/ggml-org/Julia-1-GGUF): ggml-org's BF16 and Q8_0 conversions, used
  for comparison.
* [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (Apache-2.0): the test set
  of the reference data.
* [PyTorch](https://pytorch.org/), Hugging Face [transformers](https://github.com/huggingface/transformers) and
  [tokenizers](https://github.com/huggingface/tokenizers), [ONNX Runtime](https://onnxruntime.ai/) and
  [MLX](https://github.com/ml-explore/mlx): used for validation and benchmarks only.

## License

Apache-2.0 ([LICENSE](LICENSE)), except `patches/ggml-julia1.patch`, which is MIT like ggml. Attributions are in
[NOTICE](NOTICE) and the licences of the third-party code compiled into the binaries in
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES). The model weights are Apache-2.0, by Supersonic Labs.
