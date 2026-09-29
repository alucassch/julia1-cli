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

Stock llama.cpp cannot run the decision head: its loader rejects the head tensors and its graph has no input for the
question type or the marker positions. The full model therefore uses its own GGUF architecture, `julia1`, which this
runtime loads. [SPEC.md](SPEC.md) defines the file format, the tokenizer, the request encoding, the forward pass and
the validation protocol.

**llama.cpp support.** [llama.cpp PR #29363](https://github.com/ggml-org/llama.cpp/pull/29363) adds support for this
family of decision models (the `laya` architecture). `Julia-1-laya-F32.gguf` is exported in that layout and already
reproduces the original PyTorch model on the PR's runtime (2000/2000 decisions on the typed-decisions test set, PR
commit `ffc55c93bc`, CPU); that commit fixes the issues found while validating Julia-1 on the PR
([report](https://github.com/ggml-org/llama.cpp/pull/29363#issuecomment-5861969778)). If the PR is merged, that file should run in upstream llama.cpp without julia1-cli.
The PR's CLI (`llama-laya-cli`) does not yet encode `noul` questions that carry descriptions the way Julia-1 does.
Until then, use julia1-cli or the `julia1-gguf` Python package.

## Contents

| path | what |
| --- | --- |
| `src/` | the runtime: GGUF loader, tokenizer, request encoding, ggml graph, CLI, HTTP server, C API |
| `patches/ggml-julia1.patch` | changes to ggml (MIT): exact F32 Metal kernels, fused Metal GEMM epilogues, CPU speed-ups; applied at build time |
| `julia1_gguf/` | the Python package `julia1-gguf`: the same runtime through the C API, or a pure numpy runtime |
| `tools/convert_julia1_to_gguf.py` | converter from the upstream safetensors checkpoint to GGUF |
| `tools/eval_gguf.py`, `tools/compare_replay.py`, `tools/tokenizer_parity*.py` | validation against the reference data |
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

From [andrelucas/Julia-1-GGUF](https://huggingface.co/andrelucas/Julia-1-GGUF):

| file | size | use |
| --- | --- | --- |
| `Julia-1-F32.gguf` | 593 MB | exact reference |
| `Julia-1-F16.gguf` | 312 MB | half the size; exact kernels or `--fast` |
| `Julia-1-F16-embdQ8_0.gguf` | 220 MB | smallest file (embedding table in Q8_0); 1990–1992 of 2000 decisions |
| `Julia-1-encoder-{F32,F16}.gguf` | 578 / 297 MB | encoder only (`modern-bert`), for stock llama.cpp: token hidden states, no decisions |
| `Julia-1-laya-F32.gguf` | 592 MB | the same model in the `laya` layout of [llama.cpp PR #29363](https://github.com/ggml-org/llama.cpp/pull/29363) |

BF16 and Q8_0/Q5_0/Q4_0 encoder matrices change decisions and are not published. To convert the upstream checkpoint
yourself (Python with `numpy`, `safetensors` and `gguf`):

```sh
python tools/convert_julia1_to_gguf.py --upstream <SupersonicLabs/Julia-1 download> --outdir models --types F32,F16 --encoder-types F32,F16
python tools/convert_julia1_to_gguf.py --upstream <SupersonicLabs/Julia-1 download> --outdir models --types F16 \
    --override token_embd.weight=Q8_0 --name-suffix=-embdQ8_0
```

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

* The 100 parity cases give 100/100 decisions on every julia1-cli configuration above; the Python package's native
  backend gives the same logits as julia1-cli and adds 0.0–0.3 ms per call.
* End to end, julia1-cli's tokenizer and encoding produce exactly the reference token ids (0 mismatches on the 2000
  typed and 100 parity requests); the tokenizer matches HF `tokenizers` on all 4595 strings of
  `data/tokenizer-corpus.jsonl`.
* `Julia-1-F16-embdQ8_0.gguf` keeps 1990–1992/2000 decisions (accuracy 1452) and 99–100/100 parity cases.
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

## Limitations

* Julia-1 compares the options it is given; it is not a knowledge or reasoning model. Evaluate it on your own
  questions and options before acting on its output.
* Tested on macOS on Apple silicon only. Linux builds from source and is experimental; no CUDA.
* ggml compiles its Metal shaders when a process starts; when macOS's shader cache misses (the first run on a
  machine, or after another ggml build ran), the start takes about 20 s instead of 0.1–0.3 s. `--device cpu` is
  affected too, because every backend is registered at start.
* `Julia-1-F16-embdQ8_0.gguf` is at the acceptance floor of [SPEC.md](SPEC.md) §8 (at least 1990/2000): the Q8_0
  embedding table flips a few borderline decisions. Use F32 or F16 when every decision must match the reference.
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
* [llama.cpp PR #29363](https://github.com/ggml-org/llama.cpp/pull/29363): the `laya` architecture of the laya export.
* [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (Apache-2.0): the test set
  of the reference data.
* [PyTorch](https://pytorch.org/), Hugging Face [transformers](https://github.com/huggingface/transformers) and
  [tokenizers](https://github.com/huggingface/tokenizers), [ONNX Runtime](https://onnxruntime.ai/) and
  [MLX](https://github.com/ml-explore/mlx): used for validation and benchmarks only.

## License

Apache-2.0 ([LICENSE](LICENSE)), except `patches/ggml-julia1.patch`, which is MIT like ggml. Attributions are in
[NOTICE](NOTICE) and the licences of the third-party code compiled into the binaries in
[THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES). The model weights are Apache-2.0, by Supersonic Labs.
