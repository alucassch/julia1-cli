# julia1-cli 0.2.0 — quickstart

Native runtime for the [Julia-1](https://huggingface.co/SupersonicLabs/Julia-1) decision model by Supersonic Labs, in
GGUF: a state, a question, a type (`choice`, `score` or `noul`) and 2–20 options in, one logit per option out. It is
not a generative model. Source and documentation: https://github.com/alucassch/julia1-cli. Model files:
https://huggingface.co/andrelucas/Julia-1-GGUF.

This archive (macOS, Apple silicon) contains:

| path | what |
| --- | --- |
| `bin/julia1-cli` | CLI and HTTP server: `replay`, `decide`, `predict`, `tokenize`, `serve` (`julia1-cli --help`) |
| `lib/libjulia1_gguf_native.dylib` | the same runtime as a shared library with a C API (JSON in, JSON out) |
| `include/julia1_gguf.h` | that C API |
| `share/julia1-cli/` | this file, `LICENSE` (Apache-2.0), `NOTICE`, `THIRD_PARTY_LICENSES` |

ggml (with the repository's exact Metal kernels) and its Metal shader library are compiled in; the binaries depend
only on macOS system libraries and frameworks. They are built for macOS 14 or later and were tested on macOS 27
(M1 Pro). They are not signed by a developer ID: if macOS refuses to open a file downloaded with a browser, run
`xattr -dr com.apple.quarantine julia1-cli-0.2.0-macos-arm64` once. The Python package (`julia1-gguf`) is not in this
archive: install its wheel from the release page, or build it from the source repository (`pip install .`).

The first Metal run on a machine compiles the embedded shader source and takes about 20 s; macOS caches the result,
and later starts take 0.1–0.3 s. `--device cpu` does not need the shaders.

## Model

Download a model file from https://huggingface.co/andrelucas/Julia-1-GGUF: `Julia-1-F32.gguf` (592 MB, exact
reference: 2000/2000 decisions of upstream PyTorch on the typed test set) or `Julia-1-F16.gguf` (303 MB, 2000/2000
with `--fast`). They are in llama.cpp's layout and also run in llama.cpp master (`llama-server`, `POST /v1/systemone`);
julia1-cli still loads the `julia1`-layout files of version 0.1.0.

## CLI

```sh
bin/julia1-cli --version
cat > request.jsonl <<'EOF'
{"state": "I was charged twice for the same order.", "question": "Which team should handle this request?", "options": ["Billing and payment disputes", "Shipping and delivery", "Account access and login"], "type": "choice"}
EOF
bin/julia1-cli --model Julia-1-F32.gguf decide --input request.jsonl --output answer.jsonl --max-length 8192 --head-length 512 --strict
```

`answer.jsonl` gets one line per request with `logits`, `index` (0 = "Billing and payment disputes" here),
`probabilities` and the encoded sequence. `--device auto` (default) runs on Metal and falls back to the CPU;
`--precise` (default) uses exact F32 kernels, `--fast` half-precision ones; `--batch 16` groups rows for throughput.
`predict` takes upstream's named-question form, one call per line:

```sh
cat > questions.jsonl <<'EOF'
{"state": "I was charged twice for the same order.", "questions": {"team": {"type": "choice", "instructions": "Which team should handle this request?", "criteria": {"billing": "Billing and payment disputes", "shipping": "Shipping and delivery", "access": "Account access and login"}}}}
EOF
bin/julia1-cli --model Julia-1-F32.gguf predict --input questions.jsonl --output answers.jsonl --max-length 8192 --head-length 512 --strict
```

## HTTP server

```sh
bin/julia1-cli --model Julia-1-F32.gguf serve            # http://127.0.0.1:8080; --host, --port, --api-key KEY
curl -s http://127.0.0.1:8080/v1/predict -H 'Content-Type: application/json' -d @- <<'EOF'
{"state": "I was charged twice for the same order.", "questions": {"team": {"type": "choice", "instructions": "Which team should handle this request?", "criteria": {"billing": "Billing and payment disputes", "shipping": "Shipping and delivery", "access": "Account access and login"}}}}
EOF
```

The answer is `{"answers":{"team":{"type":"choice","probabilities":{...},"choice":"billing","max_probability":...}}}`.
Other endpoints: `GET /` (playground), `/health`, `/v1/model`; `POST /v1/decide`, `/v1/logits`, `/v1/tokenize`.
The server has no TLS; it binds 127.0.0.1 unless `--host` says otherwise.

## C API

```c
#include <stdio.h>
#include "julia1_gguf.h"

int main(void) {
    char * error = NULL;
    void * engine = julia1_engine_new("{\"model\": \"Julia-1-F32.gguf\", \"head_length\": 512, \"strict\": true}", &error);
    if (!engine) { fprintf(stderr, "%s\n", error); julia1_string_free(error); return 1; }
    char * result = julia1_engine_call(engine, "predict_typed",
        "{\"state\": \"I was charged twice for the same order.\", \"questions\": {\"team\": {\"type\": \"choice\","
        " \"instructions\": \"Which team should handle this request?\", \"criteria\": {\"billing\": \"Billing and"
        " payment disputes\", \"shipping\": \"Shipping and delivery\", \"access\": \"Account access and login\"}}}}", &error);
    printf("%s\n%s\n", julia1_version(), result ? result : error);
    julia1_string_free(result ? result : error);
    julia1_engine_free(engine);
    return 0;
}
```

```sh
cc -I include example.c -L lib -ljulia1_gguf_native -Wl,-rpath,"$PWD/lib" -o example && ./example
```

Methods: `info`, `logits`, `predict_rows`, `predict_typed`, `replay`, `encode`, `tokenize` (see the header).
Returned strings are freed with `julia1_string_free`; calls on one engine are serialised by the library. The library
writes only ggml's warnings and errors to stderr; `JULIA1_GGUF_VERBOSE=1` shows ggml's full log (julia1-cli always does).
