# julia1-cli serve: HTTP API

`julia1-cli serve` keeps one model loaded and answers JSON requests over HTTP. It runs the same engine as the other
commands and adds dynamic batching of concurrent requests and a playground page. The HTTP layer is cpp-httplib (MIT),
from llama.cpp's `vendor/`. TLS is not supported.

```sh
build/julia1-cli --model Julia-1-F32.gguf serve            # http://127.0.0.1:8080, Metal when available
curl -s http://127.0.0.1:8080/v1/predict -H 'Content-Type: application/json' -d '{
  "state": "I was charged twice for the same order.",
  "questions": {"team": {"type": "choice", "instructions": "Which team should handle this request?",
    "criteria": {"billing": "Billing and payment disputes", "shipping": "Shipping and delivery",
                 "access": "Account access and login"}}}}'
```

```json
{"answers":{"team":{"type":"choice","probabilities":{"billing":0.8544651362474033,"shipping":0.14277066945266953,"access":0.002764194299927172},"choice":"billing","max_probability":0.8544651362474033}}}
```

Stderr prints one line at load, one when the server is ready, and one per request (`--quiet` omits the latter):

```
julia1-cli: loaded Julia-1-F32.gguf (Julia-1, F32) on MTL0 in 448 ms, 4 threads (precise; max_length 8192, head_length 512, strict)
julia1-cli: serving on http://127.0.0.1:8080 (max batch 16, batch wait 0 ms, 8 http threads)
julia1-cli: POST /v1/predict 200 rows=1 batch=1 device=MTL0 prepare=0.315 queue=0.010 forward=35.482 ms=36.614
```

`rows`: the request's sequences; `batch`: the sequences of the forward that carried them; `device`: the engine that ran
it; `prepare`: parsing, validation and tokenization (ms); `queue`: from enqueue to the start of the batch; `forward`:
the batch's forward; `ms`: the whole request.

## Flags

`julia1-cli --model <gguf> [--device auto|cpu|metal] [--threads N] [--precise|--fast] serve [flags]`

| flag | default | meaning |
|---|---|---|
| `--host` | `127.0.0.1` | bind address; `0.0.0.0` exposes the server to the network (see Security) |
| `--port` | `8080` | `0`: any free port; the `serving on` line names it |
| `--api-key KEY` | env `JULIA1_API_KEY` | require the key on `/v1/*`; the flag wins over the environment |
| `--max-batch` | 16 | most sequences in one batch |
| `--batch-wait-ms` | 0 on a GPU, 2 on the CPU | how long the oldest queued request waits for others |
| `--max-body-mb` | 4 | larger bodies get 413 |
| `--http-threads` | 8 | HTTP worker threads; the pool grows up to 4x under load |
| `--quiet` | off | no per-request log line |
| `--devices metal,cpu` | - | a Metal engine and a CPU helper engine on the same file, instead of `--device` |
| `--cpu-threads` | 4 | threads of the CPU engine of `--devices metal,cpu` |
| `--max-length` | 8192 | encoding: combined token budget (1..8192) |
| `--head-length` | 512 | encoding: question + options budget |
| `--strict` / `--no-strict` | strict | encoding: reject overflow and reserved markers |

The encoding defaults (8192 / 512 / strict) follow upstream's `inference-policy.json`; `julia1-cli decide` and
`predict` keep the defaults of upstream `julia.load_model` (8192 / 256 / not strict). `--device`, `--threads` and
`--precise`/`--fast` mean the same as for the other commands.

## Endpoints

All bodies are JSON. An error is `{"error": message}`, with an `"index"` field for the failing element of a
`/v1/predict` array. Validation and encoding messages are upstream's (`julia/data.py`, `julia/typed.py`,
`engine.predict`).

| method, path | body | response |
|---|---|---|
| `GET /health` | - | `{"status":"ok"}` (no key needed) |
| `GET /v1/model` | - | the engine's info plus `"server": {host, max_batch, batch_wait_ms, max_body_mb, http_threads, api_key (bool), httplib}` |
| `POST /v1/predict` | one `julia1-cli predict` line (upstream `engine.predict`): `{"state", "questions"}`, or legacy `{"rows": [...]}`; or a JSON array of them | `{"answers": {...}}` (named questions: full softmax) or `{"predictions": [{index, probabilities}]}` (legacy); an array gives an array |
| `POST /v1/decide` | `{"rows": [{state, question, options, type}]}` | `{"results": [{"index", "probabilities" (upstream's display rounding), "logits"}]}` |
| `POST /v1/logits` | `{"rows": [...]}` | `{"logits": [[...]]}` (upstream `engine.logits`) |
| `POST /v1/tokenize` | `{"texts": [string]}` | `{"ids": [[...]]}` (SPEC §4, no CLS/SEP) |
| `GET /` | - | the playground (no key needed) |

| status | when |
|---|---|
| 400 | invalid JSON, wrong body shape, validation or encoding (strict) error |
| 401 | missing or wrong key when a key is set; the response carries `WWW-Authenticate: Bearer` |
| 404 | unknown path |
| 405 | known path, other method; the response carries `Allow` |
| 413 | body over `--max-body-mb` |
| 500 | model failure, e.g. non-finite scores |

Every 200 response that ran the model carries `Server-Timing: prepare;dur=…, queue;dur=…, forward;dur=…;desc="batch N",
device;desc="MTL0"`.

```sh
curl -s localhost:8080/v1/decide -H 'Content-Type: application/json' -d '{"rows": [{"state": "I was charged twice.",
  "question": "Which team should handle this request?", "options": ["Billing", "Shipping", "Account access"], "type": "choice"}]}'
curl -s localhost:8080/v1/tokenize -H 'Content-Type: application/json' -d '{"texts": ["hello world", "olá"]}'
{"ids":[[25612,2134],[6989,235354]]}
```

Send `Content-Type: application/json`; a body sent without it (`curl -d`) is parsed as JSON too.

The playground (`GET /`) is one self-contained page: a state as text or JSON, questions of the three types, the
answers with their probabilities, and two examples. When a key is set, the page asks for it and keeps it in
`sessionStorage`.

## Batching

* HTTP threads parse, validate and encode each request, so errors return at once without touching the model. A
  request's sequences are then queued as one job.
* One inference thread owns the model. It takes the oldest job, then keeps taking queued jobs while they fit in
  `--max-batch` sequences; with an empty queue it waits for more until `--batch-wait-ms` after the oldest job's
  arrival. The batch runs as one forward: windows of `--max-batch` sequences, each sorted by length and cut into
  padded graphs. A single job larger than `--max-batch` runs alone in windows of `--max-batch`.
* If the forward fails, every job of that batch gets a 500.
* On Metal a batch's cost is nearly linear in its rows, so the default wait is 0; on the CPU larger batches are
  cheaper per row, so the default wait is 2 ms.
* SIGINT or SIGTERM closes the listening socket; the requests in flight and the queued jobs are answered, and the
  process exits 0. A second signal kills it at once.

## Two devices (`--devices metal,cpu`)

`--devices metal,cpu` loads the model twice from the same file: a Metal engine (with `--precise`/`--fast`,
`--threads`) and a CPU engine (`--cpu-threads`, the same encoding options and the same `--max-batch`). Requests are
encoded with the Metal engine's tokenizer. The CPU engine takes a batch only while the Metal engine runs one, at least
`--max-batch` sequences are queued and the oldest queued job fits in `--max-batch`; an idle server always uses Metal.
Responses carry the numbers of the engine that ran them (CPU rows are within the CPU guard of SPEC §8); the engine is
named in the `Server-Timing` `device` entry and in the log line. `/v1/model` describes the Metal engine. Without a GPU
backend the server stops at load with `no Metal/GPU backend available`.

## Security

* The server binds `127.0.0.1` by default. Expose it (`--host 0.0.0.0`) only behind a reverse proxy, which also
  provides TLS and rate limits.
* `--api-key` or `JULIA1_API_KEY` (preferred: a flag is visible in `ps`) protects `/v1/*`. The server accepts
  `Authorization: Bearer <key>` or `X-API-Key: <key>` and compares in constant time. `/` and `/health` stay open.
* Without a key, any local process can call the API, and a web page in the user's browser can send simple
  cross-origin POSTs to it (it cannot read the answers: no CORS headers are sent).
* Bodies are capped at `--max-body-mb`; cpp-httplib applies 5 s read and write timeouts. There is no rate limit and no
  queue cap. A request can cost up to 8192 tokens × 20 options of compute; lower `--max-length` or `--max-body-mb` to
  bound it.
