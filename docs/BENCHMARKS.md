# Benchmarks

julia1-cli 0.1.0 against the original PyTorch runtime of Julia-1 and the other published runtimes (MLX, ONNX Runtime),
on one machine. Every run also records its parity with the reference, so each timing comes with the accuracy it was
measured at. julia1-cli 0.2.0 loads the llama.cpp-layout files of the model repository; on the same F32 weights its
output is byte-identical to 0.1.0's on the `julia1`-layout file, so the F32 numbers below hold. §F compares upstream
llama.cpp's `/v1/systemone` with `julia1-cli serve`.

## Setup

* **Machine:** Apple M1 Pro (8 performance + 2 efficiency CPU cores, 16 GPU cores), 16 GB, macOS 27.0.
* **Runtimes:**
  * julia1-cli 0.1.0: ggml of llama.cpp v0.5.0 with `patches/ggml-julia1.patch`, the `julia1`-layout files
    `Julia-1-F32.gguf`, `Julia-1-F16.gguf`, `Julia-1-F16-embdQ8_0.gguf` (the last no longer published); the Python
    package's native backend (`julia1_gguf`, the same library through ctypes) and its numpy backend.
  * PyTorch 2.14.0 (transformers 5.0.0) with upstream's reference code and checkpoint
    ([SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1)), on the CPU and on MPS. MPS is not a
    device upstream supports (`JuliaDecisionModel.to("mps")`).
  * ONNX Runtime 1.30.0 with [SupersonicLabs/Julia-1-ONNX](https://huggingface.co/SupersonicLabs/Julia-1-ONNX), CPU
    and CoreML execution providers.
  * MLX 0.32.2 with [julia-mlx](https://github.com/zm2231/julia-mlx) 0.2.0 on the upstream checkpoint (the weights of
    [zainmerchan/Julia-1-MLX](https://huggingface.co/zainmerchan/Julia-1-MLX)), fp32 and fp16.
* **Threads:** PyTorch CPU 4 (upstream's default); ONNX Runtime CPU and CoreML 8; julia1-cli CPU 4 (its default) and
  6; the native Python backend on the CPU 4; numpy 8. `OMP_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`,
  `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` were set to the same number.
* **Encoding:** typed `max_length` 1024, `head_length` 512, strict; parity `head_length` 256 (as in
  `data/reference/`).

## Protocol

* One configuration per process, one process at a time. Between runs: a 15 s pause, then a wait until the 1-minute
  load average is below 5 (at most 60 s); the load at the start of the runs was 1.2–6.8, median 3.0.
* Each run: load the model, time one call on row 0 (first call), run rows 0–19 once and discard them (warm-up), then
  time every row once, in reference order. Tables give the median (and p90) over the rows.
* **Replay (A):** the reference token ids, so only the model runs. **End to end (B):** the raw request, tokenised
  and encoded by the runtime. **batch16 (C):** consecutive chunks of 16 raw requests through each runtime's own
  batching (julia1-cli: `--batch 16`); rows/s.
* The four GPU headline configurations (julia1-cli `--fast` F16 and exact F32, MLX fp16 and fp32) ran twice in A, B
  and C, interleaved; both values are shown (run 1 / run 2).
* "× PyTorch CPU" is the median of PyTorch CPU divided by the runtime's median (or the rows/s ratio in C).
* Parity columns: decisions equal to the reference (argmax), typed accuracy (the reference has 1451), and the largest
  absolute logit difference to the reference.

What one timed call includes:

| runtime | replay (A) | end to end (B) | batch16 (C) | timed |
| --- | --- | --- | --- | --- |
| julia1-cli | graph, input upload, compute, readback of the logits | + tokenisation and encoding (`decide`) | `decide --batch 16`: wall time of 16 rows / 16 | inside the process (`--timing`), JSON I/O excluded |
| Python native (`julia1_gguf`) | `engine.replay(...)`: julia1-cli's work plus JSON and ctypes | `engine.logits([request])` | `engine.logits(16 requests)`, `batch=16` | Python wall time |
| PyTorch CPU | upstream `FastEngine` packing + forward + `.tolist()` | `engine.logits([request])` | `engine.logits(16 requests)` | Python wall time |
| PyTorch MPS | upstream `Collator` + model on `mps` + copy back | + upstream's encoding | 16 rows in one batch | Python wall time |
| MLX | julia-mlx collation + model + `mx.eval` + `.tolist()` | `engine.logits([request])` | `engine.logits(16 requests)` | Python wall time |
| ONNX Runtime | input arrays + `session.run` | + upstream's encoding and HF tokenizer | 16 rows padded into one run | Python wall time |
| numpy (`julia1_gguf`) | `engine.replay(...)` | – | – | Python wall time |

julia1-cli is timed without a Python boundary, the other runtimes with it. The native backend rows measure
that boundary for julia1-cli's runtime: 0.0–0.2 ms per call on Metal, 0.25–0.3 ms on the CPU.

## Reference and parity

The reference is upstream's code in PyTorch on the CPU in FP32 (torch 2.14.0, transformers 5.0.0, one example per
forward):

* typed set, the 2000 questions of [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions):
  accuracy **1451/2000** (choice 426/600, noul 483/600, score 542/800), upstream's published CPU FP32 result;
* parity set, the 100 cases of Julia-1-ONNX: 1.03e-4 from the PyTorch logits stored upstream.

Both are in `data/reference/`, and `tools/eval_gguf.py` / `tools/compare_replay.py` check a runtime against them
([SPEC.md](../SPEC.md) §8).

| file | julia1-cli configuration | typed argmax | accuracy | typed max abs | parity argmax | parity max abs |
| --- | --- | --- | --- | --- | --- | --- |
| F32 | Metal exact (default) | 2000/2000 | 1451 | 8.24e-4 | 100/100 | 2.23e-4 |
| F32 | CPU, 4 or 6 threads | 2000/2000 | 1451 | 8.43e-4 | 100/100 | 1.57e-4 |
| F32 | Metal `--fast` | 1998/2000 | 1452 | 0.224 | 100/100 | 0.0758 |
| F16 | Metal `--fast` | 2000/2000 | 1451 | 0.561 | 100/100 | 0.0603 |
| F16-embdQ8_0 | Metal `--fast` | 1990/2000 | 1452 | 1.24 | 99/100 | 0.442 |

* End to end (`decide`, raw requests), the encodings equal the reference ids, markers and question types on all 2000
  typed and 100 parity requests, and the logits equal those of replay; `--batch 16` gives the same parity.
* The Python package's native backend gives the same values as julia1-cli; the numpy backend gives 2000/2000,
  accuracy 1451, max abs 5.05e-4 (typed) and 100/100, 1.63e-4 (parity) on the F32 file.
* The tokenizer matches HF `tokenizers` on the 4595 strings of `data/tokenizer-corpus.jsonl` (0 mismatches).
* Rounding the weights moves the logits even in an exact runtime: PyTorch with its encoder matrices rounded to F16
  is 0.042 from the reference on the parity set, and julia1-cli's exact kernels on the F16 file are within 1.3e-4 of
  that PyTorch run; `--fast` adds half-precision arithmetic on top. Files other than F32 are therefore accepted on
  their decisions. Encoder matrices in BF16 (1994/2000), Q8_0 (about 1940/2000), Q5_0 or Q4_0 lose decisions and are
  not published.

## A. Latency, batch 1, replay

### A1. Typed set

| runtime | threads | rows | median ms | p90 ms | × PyTorch CPU | argmax, accuracy, max abs |
|---|---|---|---|---|---|---|
| julia1-cli, Metal `--fast`, F16 | – | 2000 | 11.30 / 11.42 | 14.65 / 14.68 | 5.29× | 2000/2000, 1451, 0.561 |
| julia1-cli, Metal `--fast`, F16-embdQ8_0 | – | 2000 | 11.31 | 14.68 | 5.31× | 1990/2000, 1452, 1.24 |
| Python native, Metal fast, F16 | – | 2000 | 11.52 | 14.87 | 5.22× | 2000/2000, 1451, 0.561 |
| julia1-cli, Metal `--fast`, F32 | – | 2000 | 11.85 | 15.26 | 5.07× | 1998/2000, 1452, 0.224 |
| julia1-cli, Metal exact (default), F32 | – | 2000 | 12.25 / 12.29 | 15.74 / 15.70 | 4.90× | 2000/2000, 1451, 8.24e-4 |
| Python native, Metal exact, F32 | – | 2000 | 12.36 | 15.94 | 4.86× | 2000/2000, 1451, 8.24e-4 |
| MLX fp16 (julia-mlx) | – | 2000 | 15.71 / 15.48 | 19.83 / 19.55 | 3.85× | 1997/2000, 1452, 0.478 |
| MLX fp32 (julia-mlx) | – | 2000 | 17.20 / 17.19 | 21.69 / 21.72 | 3.50× | 2000/2000, 1451, 8.64e-4 |
| PyTorch MPS (upstream code) | – | 2000 | 29.95 | 34.47 | 2.01× | 2000/2000, 1451, 4.99e-4 |
| julia1-cli, CPU, F32 | 6 | 2000 | 39.46 | 55.92 | 1.52× | 2000/2000, 1451, 8.43e-4 |
| julia1-cli, CPU, F32 | 4 | 2000 | 41.91 | 55.61 | 1.43× | 2000/2000, 1451, 8.43e-4 |
| Python native, CPU, F32 | 4 | 2000 | 42.22 | 55.33 | 1.42× | 2000/2000, 1451, 8.43e-4 |
| **PyTorch CPU (upstream)** | 4 | 2000 | 60.10 | 73.68 | 1.00× | 2000/2000, 1451, 2.29e-4 |
| ONNX Runtime CPU (Julia-1-ONNX) | 8 | 2000 | 85.31 | 129.1 | 0.70× | 2000/2000, 1451, 7.59e-4 |
| `julia1_gguf` numpy, F32 | 8 | 400 | 77.02 | 83.51 | – | 400/400, 279 (reference 279), 5.05e-4 |
| ONNX Runtime CoreML EP | 8 | 400 | 260.7 | 371.7 | – | 400/400, 279 (reference 279), 0.169 |

numpy and ONNX Runtime CoreML (0.08 s and 0.26 s per row) ran the first 400 typed rows only. Those rows are shorter
than the whole set, so their medians are not comparable with the others and get no ratio. The CoreML provider placed
660 of the 1395 ONNX nodes on CoreML, in 197 partitions, and cannot run batches.

### A2. Parity set (100 rows)

| runtime | threads | rows | median ms | p90 ms | × PyTorch CPU | argmax, max abs |
|---|---|---|---|---|---|---|
| julia1-cli, Metal `--fast`, F16 | – | 100 | 4.94 / 5.01 | 8.78 / 9.02 | 5.18× | 100/100, 0.0603 |
| julia1-cli, Metal `--fast`, F16-embdQ8_0 | – | 100 | 4.92 | 8.75 | 5.24× | 99/100, 0.442 |
| Python native, Metal fast, F16 | – | 100 | 5.01 | 8.92 | 5.14× | 100/100, 0.0603 |
| julia1-cli, Metal `--fast`, F32 | – | 100 | 5.42 | 9.24 | 4.76× | 100/100, 0.0758 |
| julia1-cli, Metal exact (default), F32 | – | 100 | 5.35 / 5.39 | 9.46 / 9.55 | 4.80× | 100/100, 2.23e-4 |
| Python native, Metal exact, F32 | – | 100 | 5.44 | 9.72 | 4.74× | 100/100, 2.23e-4 |
| MLX fp16 (julia-mlx) | – | 100 | 8.27 / 8.08 | 13.89 / 12.20 | 3.16× | 100/100, 0.079 |
| MLX fp32 (julia-mlx) | – | 100 | 8.37 / 8.35 | 13.24 / 13.20 | 3.08× | 100/100, 1.85e-4 |
| PyTorch MPS (upstream code) | – | 100 | 19.34 | 27.19 | 1.33× | 100/100, 1.31e-4 |
| julia1-cli, CPU, F32 | 6 | 100 | 17.43 | 27.87 | 1.48× | 100/100, 1.57e-4 |
| julia1-cli, CPU, F32 | 4 | 100 | 18.10 | 30.96 | 1.43× | 100/100, 1.57e-4 |
| Python native, CPU, F32 | 4 | 100 | 18.35 | 30.73 | 1.41× | 100/100, 1.57e-4 |
| **PyTorch CPU (upstream)** | 4 | 100 | 25.80 | 46.51 | 1.00× | 100/100, 7.84e-5 |
| ONNX Runtime CPU (Julia-1-ONNX) | 8 | 100 | 21.65 | 57.72 | 1.19× | 100/100, 1.35e-4 |
| `julia1_gguf` numpy, F32 | 8 | 100 | 35.78 | 101.0 | 0.72× | 100/100, 1.63e-4 |
| ONNX Runtime CoreML EP | 8 | 100 | 178.3 | 298.6 | 0.14× | 100/100, 0.0557 |

### A3. Typed set by sequence length (median ms; × PyTorch CPU in parentheses)

| runtime | threads | ≤ 128 tokens (18 rows) | 129–256 (650 rows) | 257–512 (1297 rows) | 513–1024 (35 rows) |
|---|---|---|---|---|---|
| julia1-cli, Metal `--fast`, F16 | – | 5.64 (6.15×) | 7.44 (5.69×) | 13.01 (4.95×) | 19.30 (5.32×) |
| julia1-cli, Metal `--fast`, F16-embdQ8_0 | – | 5.61 (6.18×) | 7.45 (5.69×) | 13.05 (4.94×) | 19.61 (5.24×) |
| Python native, Metal fast, F16 | – | 5.69 (6.09×) | 7.58 (5.59×) | 13.23 (4.87×) | 19.86 (5.17×) |
| julia1-cli, Metal `--fast`, F32 | – | 6.08 (5.71×) | 7.90 (5.36×) | 13.69 (4.71×) | 20.12 (5.10×) |
| julia1-cli, Metal exact (default), F32 | – | 6.05 (5.73×) | 8.02 (5.28×) | 14.04 (4.59×) | 21.29 (4.82×) |
| Python native, Metal exact, F32 | – | 6.16 (5.63×) | 8.15 (5.20×) | 14.15 (4.55×) | 21.45 (4.79×) |
| MLX fp16 (julia-mlx) | – | 8.70 (3.99×) | 10.84 (3.91×) | 17.72 (3.64×) | 27.73 (3.70×) |
| MLX fp32 (julia-mlx) | – | 8.92 (3.89×) | 11.40 (3.72×) | 19.32 (3.33×) | 31.14 (3.30×) |
| PyTorch MPS (upstream code) | – | 20.16 (1.72×) | 22.70 (1.87×) | 31.90 (2.02×) | 44.73 (2.30×) |
| julia1-cli, CPU, F32 | 6 | 20.19 (1.72×) | 26.78 (1.58×) | 43.39 (1.48×) | 69.60 (1.47×) |
| julia1-cli, CPU, F32 | 4 | 21.29 (1.63×) | 27.69 (1.53×) | 47.24 (1.36×) | 76.98 (1.33×) |
| Python native, CPU, F32 | 4 | 20.64 (1.68×) | 27.89 (1.52×) | 47.37 (1.36×) | 78.91 (1.30×) |
| **PyTorch CPU (upstream)** | 4 | 34.68 (1.00×) | 42.35 (1.00×) | 64.42 (1.00×) | 102.7 (1.00×) |
| ONNX Runtime CPU (Julia-1-ONNX) | 8 | 36.19 (0.96×) | 51.39 (0.82×) | 98.87 (0.65×) | 173.7 (0.59×) |

## B. Latency, batch 1, end to end (typed set)

| runtime | threads | rows | median ms | p90 ms | × PyTorch CPU | argmax, accuracy, max abs |
|---|---|---|---|---|---|---|
| julia1-cli, Metal `--fast`, F16 | – | 2000 | 11.62 / 11.66 | 15.00 / 14.97 | 5.11× | 2000/2000, 1451, 0.561 |
| Python native, Metal fast, F16 | – | 2000 | 11.73 | 15.15 | 5.07× | 2000/2000, 1451, 0.561 |
| julia1-cli, Metal exact (default), F32 | – | 2000 | 12.43 / 12.52 | 15.96 / 16.03 | 4.77× | 2000/2000, 1451, 8.24e-4 |
| Python native, Metal exact, F32 | – | 2000 | 12.63 | 16.22 | 4.71× | 2000/2000, 1451, 8.24e-4 |
| MLX fp16 (julia-mlx) | – | 2000 | 15.84 / 15.67 | 20.14 / 19.83 | 3.78× | 1997/2000, 1452, 0.478 |
| MLX fp32 (julia-mlx) | – | 2000 | 17.29 / 17.32 | 22.05 / 21.99 | 3.44× | 2000/2000, 1451, 8.64e-4 |
| julia1-cli, CPU, F32 | 6 | 2000 | 40.20 | 57.39 | 1.48× | 2000/2000, 1451, 8.43e-4 |
| **PyTorch CPU (upstream)** | 4 | 2000 | 59.51 | 73.44 | 1.00× | 2000/2000, 1451, 2.29e-4 |
| ONNX Runtime CPU (Julia-1-ONNX) | 8 | 2000 | 84.14 | 133.1 | 0.71× | 2000/2000, 1451, 7.59e-4 |

## C. Throughput, batch16 (typed set)

| runtime | threads | rows/s | median ms per chunk of 16 | × PyTorch CPU | argmax, accuracy, max abs |
|---|---|---|---|---|---|
| julia1-cli, Metal `--fast`, F16 | – | 111.0 / 111.0 | 154.4 / 153.7 | 4.33× | 2000/2000, 1451, 0.561 |
| Python native, Metal fast, F16 | – | 110.5 | 153.8 | 4.32× | 2000/2000, 1451, 0.561 |
| julia1-cli, Metal exact (default), F32 | – | 102.0 / 101.9 | 167.5 / 168.2 | 3.98× | 2000/2000, 1451, 8.24e-4 |
| Python native, Metal exact, F32 | – | 101.6 | 168.0 | 3.97× | 2000/2000, 1451, 8.24e-4 |
| MLX fp16 (julia-mlx) | – | 86.5 / 86.6 | 204.4 / 204.6 | 3.38× | 1997/2000, 1451, 0.759 |
| MLX fp32 (julia-mlx) | – | 77.3 / 77.3 | 228.4 / 227.7 | 3.02× | 2000/2000, 1451, 8.77e-4 |
| PyTorch MPS (upstream code) | – | 62.4 | 269.9 | 2.43× | 2000/2000, 1451, 4.99e-4 |
| julia1-cli, CPU, F32 | 6 | 32.1 / 32.3 | 540.8 / 529.6 | 1.26× | 2000/2000, 1451, 8.43e-4 |
| **PyTorch CPU (upstream)** | 4 | 25.6 | 682.2 | 1.00× | 2000/2000, 1451, 2.29e-4 |
| ONNX Runtime CPU (Julia-1-ONNX) | 8 | 10.9 | 1534.0 | 0.42× | 2000/2000, 1451, 7.59e-4 |

The two julia1-cli CPU values are two runs; the first started at a load average of 6.8.

## D. Server

`julia1-cli serve` with the default settings (`--max-batch 16`, `--batch-wait-ms 0`, 8 HTTP threads); 2000 typed
requests of one row each, sent by 1, 16 or 32 concurrent clients. "CPU share": requests answered by the CPU engine of
`--devices metal,cpu`.

| server | clients | requests/s | latency p50 / p90 / p99 ms | mean batch | CPU share | argmax | accuracy | max abs |
|---|---|---|---|---|---|---|---|---|
| exact, F32 | 1 | 77.9 | 12.9 / 16.5 / 21.9 | 1.0 | – | 2000/2000 | 1451 | 8.24e-4 |
| exact, F32 | 16 | 100.8 | 168.8 / 201.5 / 261.3 | 14.1 | – | 2000/2000 | 1451 | 8.24e-4 |
| exact, F32 | 32 | 102.7 | 329.1 / 394.4 / 440.8 | 16.0 | – | 2000/2000 | 1451 | 8.24e-4 |
| exact, F32, `--devices metal,cpu` | 1 | 77.8 | 12.9 / 16.5 / 21.8 | 1.0 | 0% | 2000/2000 | 1451 | 8.24e-4 |
| exact, F32, `--devices metal,cpu` | 16 | 100.3 | 164.4 / 189.8 / 232.8 | 13.0 | 0% | 2000/2000 | 1451 | 8.24e-4 |
| exact, F32, `--devices metal,cpu` | 32 | 128.2 | 185.9 / 609.5 / 930.0 | 12.2 | 22% | 2000/2000 | 1451 | Metal 8.24e-4, CPU 4.35e-4 |
| `--fast`, F16 | 1 | 83.8 | 12.0 / 15.4 / 20.1 | 1.0 | – | 2000/2000 | 1451 | 0.561 |
| `--fast`, F16 | 16 | 109.1 | 150.8 / 174.4 / 212.3 | 11.4 | – | 2000/2000 | 1451 | 0.561 |
| `--fast`, F16 | 32 | 112.0 | 294.5 / 350.0 / 404.9 | 15.5 | – | 2000/2000 | 1451 | 0.561 |
| `--fast`, F16, `--devices metal,cpu` | 1 | 83.6 | 12.0 / 15.4 / 20.2 | 1.0 | 0% | 2000/2000 | 1451 | 0.561 |
| `--fast`, F16, `--devices metal,cpu` | 16 | 109.6 | 155.3 / 186.1 / 241.2 | 14.1 | 0% | 2000/2000 | 1451 | 0.561 |
| `--fast`, F16, `--devices metal,cpu` | 32 | 135.9 | 153.9 / 656.5 / 784.9 | 13.7 | 20% | 2000/2000 | 1451 | Metal 0.561, CPU 0.123 |

At 32 clients the CPU helper engine adds 25% (exact) and 21% (fast) requests/s and roughly doubles the p99 latency.

## E. Start-up

Process start to the first result written, one request, model file in the page cache, 7 runs per row:

| julia1-cli | command | median ms | min–max ms | model load ms (median) |
|---|---|---|---|---|
| Metal exact, F32 | `decide` | 193.0–204.1 | 187.8–224.6 | 147–163 |
| Metal exact, F32 | `replay` | 121.6–122.3 | 112.4–124.9 | 86 |
| CPU, F32, 4 threads | `decide` | 193.1–196.6 | 183.3–207.7 | 139–143 |
| CPU, F32, 4 threads | `replay` | 113.3–116.4 | 111.1–120.8 | 77–78 |

Ranges over two rounds. `decide` loads the tokenizer too. Time to the first result of the other runtimes, from the
replay runs (model load plus the first call, after the interpreter started): native Python backend 0.27–0.37 s,
ONNX Runtime CPU 0.76 s, MLX 1.3–1.4 s, PyTorch CPU 8.4 s, PyTorch MPS 9.6 s, numpy 11.7 s.

ggml compiles its embedded Metal shader library at every process start (under 0.3 s when macOS's shader cache
hits).
When the cache misses, the compile takes about 21 s. This happened at the first start of a build on the machine and
when two builds of the same ggml alternated. Timings per row are not affected. MLX and PyTorch ship precompiled Metal
libraries and do not have this cost.

## F. Upstream llama.cpp (`/v1/systemone`)

llama.cpp master runs Julia-1 since [PR #29818](https://github.com/ggml-org/llama.cpp/pull/29818) (merged
2026-10-02; no tagged release contains it yet): `llama-server` answers `POST /v1/systemone` with the
`{"state", "questions"}` body of `julia1-cli serve`'s `/v1/predict`. (The `laya` file and the
llama.cpp PR #29363 rows of julia1-cli 0.1.0's version of this page are retired; #29818 supersedes that PR.)

* **llama.cpp:** master `1fb7ef3e` (2026-10-02), `llama-server` on Metal or on the CPU (6 threads), with the files of
  the model repository (`Julia-1-F32.gguf`, `Julia-1-F16.gguf`, converted by that tree's `convert_hf_to_gguf.py`) and
  ggml-org's BF16 and Q8_0 conversions ([ggml-org/Julia-1-GGUF](https://huggingface.co/ggml-org/Julia-1-GGUF)).
* **julia1-cli:** `serve` with its defaults, on the 0.1.0 `julia1`-layout files: F32 exact, F16 `--fast`.
* **Method:** each of the 2000 typed questions is one request (its state and one named question), sent one after the
  other over HTTP by the same Python client to either server. Per row: the decision against the reference, the prompt
  length (`usage.input_tokens`; llama-server only) against the reference ids, and the logits recovered from the
  returned probabilities (centred log-probabilities; Julia-1 has no temperature; for `noul`, the difference of the two
  logits) against the centred reference logits. ms = median wall time of a request in the client, HTTP included; not a
  throughput test. Harness: `tools/systemone_parity.py` (`--endpoint /v1/predict` for julia1-cli, `--state-as-string`
  for the string rows).
* **State:** sent as a JSON object, which llama.cpp renders with its template's `tojson`, or as a string
  pre-serialised with Python's `json.dumps`, as upstream encodes it. julia1-cli always gets the object.

llama-server ran with `-b 1024 -ub 1024`, except in the first row (its defaults, `-ub 512`).

| server, state, options | file, device | decisions | accuracy | median / max abs Δlogit | ms |
|---|---|---|---|---|---|
| llama-server defaults, object | F32, Metal | 1913/1965, 35 HTTP 500 | 1410 | 0.055 / 17.1 | 22.4 |
| llama-server, object | F32, Metal | 1948/2000 | 1436 | 0.055 / 17.1 | 22.5 |
| llama-server, string | F32, Metal | 1999/2000 | 1452 | 0.037 / 0.40 | 22.8 |
| llama-server, string | F32, CPU | 1999/2000 | 1452 | 0.030 / 0.29 | 62.2 |
| llama-server, string, `-fa off` | F32, CPU | 1999/2000 | 1452 | 0.029 / 0.28 | 73.5 |
| llama-server, string | F16, Metal | 2000/2000 | 1451 | 0.039 / 0.58 | 21.8 |
| llama-server, string, `-fa off` | F16, CPU | 1999/2000 | 1452 | 0.034 / 0.36 | 91.5 |
| llama-server, string | F16 with `token_embd` in Q8_0 (not published), Metal | 1990/2000 | 1452 | 0.11 / 1.05 | 21.9 |
| llama-server, object | ggml-org BF16, Metal | 1938/2000 | 1434 | 0.23 / 16.7 | 24.0 |
| llama-server, string | ggml-org BF16, Metal | 1987/2000 | 1451 | 0.17 / 4.1 | 24.0 |
| llama-server, string | ggml-org Q8_0, Metal | 1947/2000 | 1458 | 0.52 / 7.0 | 22.7 |
| erf GeGLU (diagnostic), string | F32, Metal | 2000/2000 | 1451 | 0.024 / 0.29 | 22.0 |
| erf GeGLU (diagnostic), string | F32, CPU | 2000/2000 | 1451 | 0.010 / 0.12 | 75.8 |
| erf GeGLU (diagnostic), string, `-fa off` | F32, CPU | 2000/2000 | 1451 | 4.6e-5 / 1.05e-3 | 78.1 |
| erf GeGLU (diagnostic), string, `-fa off` | F16, CPU | 2000/2000 | 1451 | 0.014 / 0.15 | 97.2 |
| erf GeGLU (diagnostic), string, `-fa off` | ggml-org BF16, CPU | 1994/2000 | 1453 | 0.11 / 2.4 | 91.3 |
| erf GeGLU (diagnostic), string, `-fa off` | ggml-org Q8_0, CPU | 1917/2000 | 1448 | 0.83 / 8.6 | 71.0 |
| **julia1-cli serve**, object | F32, Metal exact | 2000/2000 | 1451 | 5.0e-5 / 8.5e-4 | 13.4 |
| **julia1-cli serve** `--fast`, object | F16, Metal | 2000/2000 | 1451 | 0.023 / 0.41 | 12.5 |

"erf GeGLU (diagnostic)": the same llama.cpp tree with one local change, ggml's exact-erf GeGLU in place of the tanh
one in its FFN; it is not in llama.cpp.

* **Prompt length.** `-ub` (the physical batch) must hold the whole prompt. With the default 512, the 35 typed prompts
  of 524–607 tokens fail with HTTP 500 (`input (607 tokens) is too large to process. increase the physical batch
  size`); `-b 1024 -ub 1024` removes every error. Prompts can reach 8192 tokens, so set `-ub` to the longest one
  expected.
* **JSON states.** llama.cpp's Jinja `tojson` writes floats with 6 significant digits and drops `.0` (`532.0` → `532`,
  `1234567.5` → `1.23457e+06`), where upstream serialises with Python's `json.dumps`. With the state sent as an
  object, 480 of the 2000 prompts tokenise differently and 51 of the 52 changed decisions are among them; sent as a
  string, every prompt has the reference length.
* **GELU.** llama.cpp runs ModernBERT's `gelu` as ggml's tanh-approximation GeGLU; upstream (HF transformers) uses the
  exact erf. With the erf GeGLU and `-fa off`, F32 on the CPU is as close to the reference as julia1-cli (median
  4.6e-5, max 1.05e-3); CPU flash attention adds a little error. On Metal the median stays at 0.02–0.04 with or
  without it (llama.cpp's stock F32 matrix multiplication on Metal).
* **Quantisation.** Below F16 decisions change: Julia-1 has massive activations in residual dimensions 265 and 290. In
  julia1-cli (Metal exact, `tools/eval_gguf.py`), ggml-org's BF16 and Q8_0 files give 1986/2000 (accuracy 1454) and
  1946/2000 (accuracy 1460).
* **Speed.** One request at a time, `julia1-cli serve` answers in 13.4 ms (F32, exact) and 12.5 ms (F16, `--fast`),
  `llama-server` in 22.8 ms (F32) and 21.8 ms (F16) on Metal: julia1-cli is about 1.7× faster, and exact on Metal.

## Limits

* One machine (M1 Pro); the thread counts are the upstream defaults or the fastest measured settings on it.
* The MLX and PyTorch MPS rows use the GPU, like julia1-cli on Metal; the julia-mlx authors report their own numbers
  on other hardware.
* Benchmarks use `max_length` 1024; requests of up to 8192 tokens were checked for parity, not for speed.
