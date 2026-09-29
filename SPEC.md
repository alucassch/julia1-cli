# Julia-1 in GGUF — specification (julia1, v1)

This document is the contract between the converter (`tools/convert_julia1_to_gguf.py`), the runtimes (C++/ggml in
`src/`, Python/numpy in `julia1_gguf/`), the tokenizer implementations and the validation tools. Every number here was
read from the upstream checkpoint and reference code, [SupersonicLabs/Julia-1](https://huggingface.co/SupersonicLabs/Julia-1)
(weights SHA-256 `df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72`). References to `julia/...` are to
upstream's Python package of that name.

Julia-1 is **not** a generative model. It is a *decision model*: given a `state`, a `question`, a `type` and 2–20
`options`, it returns one logit per option. Stock llama.cpp cannot execute the decision graph (its loader rejects
unknown tensors, and its graph has no input for the question type or the option marker positions), so the full model
has its own architecture:

| Artifact | Architecture key | Runs in |
| --- | --- | --- |
| `Julia-1-<TYPE>.gguf` (full model) | `general.architecture = "julia1"` | julia1-cli (C++/ggml) and the `julia1_gguf` Python package (the same runtime, or numpy) |
| `Julia-1-encoder-<TYPE>.gguf` (encoder only) | `general.architecture = "modern-bert"` | stock llama.cpp (`llama-embedding --pooling none`); token-level hidden states only |

`<TYPE>` ∈ `F32`, `F16`, `BF16`, `Q8_0`, `Q5_0`, `Q4_0`, optionally with per-tensor overrides (§2.3). K-quants are
not used: almost every matrix has `ne[0]` = 384 or 1152, which is not a multiple of the 256-wide k-quant block.

## 1. Model summary

* Encoder: mmBERT-small = ModernBERT, 22 layers, hidden 384, 6 heads × 64, GeGLU FFN 1152 (fused up projection
  2304), vocab 256 000, RoPE θ = 160 000 (both layer kinds), NEOX/rotate-half rotation over all 64 dims, LayerNorm
  without bias (eps 1e-5), no positional embeddings, layer 0 has **no** attention pre-norm, final norm after the last
  layer. Layer `il` uses global attention when `il % 3 == 0`, otherwise sliding attention with window
  `|pos_q − pos_k| ≤ 64` (`local_attention = 128`).
* Decision head (`julia_config.json`: `head_layers=2, n_act=2`): a query-type embedding added to every encoder output,
  two standard pre-norm `nn.TransformerEncoderLayer`s (6 heads, FFN 1536, **ReLU**, LayerNorm with bias, biases
  everywhere, no RoPE, no window), a gather of the marker positions, and a scorer
  `LayerNorm → Linear(384,384) → GELU(erf) → Linear(384,1)`.
* Not exported (never used at inference): `act_head.*` (only with `return_actions=True`) and the `temperature` buffer
  (registered, never read).

Parameters exported: 144 292 870 − act_head (99 584 + 256 + 514 = 100 354) − temperature (3) = **144 192 513**.

## 2. Full file `Julia-1-<TYPE>.gguf`

### 2.1 Metadata

| Key | Type | Value |
| --- | --- | --- |
| `general.architecture` | str | `julia1` |
| `general.type` | str | `model` |
| `general.name` | str | `Julia-1` |
| `general.author` / `general.organization` | str | `Supersonic Labs` |
| `general.license` | str | `apache-2.0` |
| `general.description` | str | short description incl. "decision model, not generative" |
| `general.source.url` | str | `https://huggingface.co/SupersonicLabs/Julia-1` |
| `general.base_model.count` | u32 | 1 |
| `general.base_model.0.name` / `.organization` / `.repo_url` | str | `mmBERT-small` / `jhu-clsp` / `https://huggingface.co/jhu-clsp/mmBERT-small` |
| `general.file_type` | u32 | `LlamaFileType` of the base TYPE (ALL_F32=0, MOSTLY_F16=1, MOSTLY_Q4_0=2, MOSTLY_Q8_0=7, MOSTLY_Q5_0=8, MOSTLY_BF16=32) |
| `general.quantization_version` | u32 | 2 |
| `julia1.context_length` | u32 | 8192 |
| `julia1.embedding_length` | u32 | 384 |
| `julia1.block_count` | u32 | 22 |
| `julia1.feed_forward_length` | u32 | 1152 |
| `julia1.attention.head_count` / `.head_count_kv` | u32 | 6 / 6 |
| `julia1.attention.layer_norm_epsilon` | f32 | 1e-5 |
| `julia1.attention.sliding_window` | u32 | 128 |
| `julia1.attention.sliding_window_pattern` | u32 | 3 (layer `il` is global iff `il % 3 == 0`) |
| `julia1.attention.causal` | bool | false |
| `julia1.rope.freq_base` / `.freq_base_swa` | f32 | 160000 / 160000 (runtimes reject files where they differ) |
| `julia1.rope.dimension_count` | u32 | 64 |
| `julia1.rope.scaling.type` | str | `none` |
| `julia1.vocab_size` | u32 | 256000 |
| `julia1.hidden_activation` | str | `gelu` (erf) |
| `julia1.pooling_type` | u32 | 0 (none) |
| `julia1.head.block_count` | u32 | 2 |
| `julia1.head.attention.head_count` | u32 | 6 |
| `julia1.head.feed_forward_length` | u32 | 1536 |
| `julia1.head.layer_norm_epsilon` | f32 | 1e-5 |
| `julia1.head.activation` | str | `relu` |
| `julia1.scorer.activation` | str | `gelu` (erf) |
| `julia1.qtype_names` | [str] | `["choice", "score", "noul"]` (index = qtype id) |
| `julia1.encoding.cls_token_id` | u32 | 2 (`<bos>`) |
| `julia1.encoding.sep_token_id` | u32 | 1 (`<eos>`) |
| `julia1.encoding.marker_token_id` | u32 | 4 (`<mask>`) |
| `julia1.encoding.pad_token_id` | u32 | 0 |
| `julia1.encoding.marker_token_text` | str | `<mask>` |
| `julia1.encoding.head_template` | str | `{type} question: {question}` |
| `julia1.encoding.option_prefix` | str | ` ` (one space) |
| `julia1.encoding.option_token_limit` | u32 | 48 |
| `julia1.encoding.min_options` / `.max_options` | u32 | 2 / 20 |
| `julia1.encoding.default_max_length` | u32 | 8192 (upstream `inference-policy.json`) |
| `julia1.encoding.default_head_length` | u32 | 512 (upstream `inference-policy.json`) |
| `julia1.upstream.weights_sha256` | str | `df853bf7…95b72` |
| `julia1.upstream.repo` | str | `SupersonicLabs/Julia-1` |
| `julia1.upstream.checkpoint_step` | u32 | 500 |
| `julia1.quantization.recipe` | str | only in files written with overrides (§2.3), e.g. `base=F16;token_embd.weight=Q8_0` |
| `tokenizer.ggml.model` | str | `llama` |
| `tokenizer.ggml.pre` | str | `default` |
| `tokenizer.ggml.tokens` | [str] | 256000 token strings, index = id |
| `tokenizer.ggml.scores` | [f32] | `-rank` of the earliest merge producing the token; `0.0` for tokens no merge produces |
| `tokenizer.ggml.token_type` | [i32] | 1 NORMAL, 3 CONTROL (special added: `<pad> <eos> <bos> <unk> <mask> <start_of_turn> <end_of_turn>`), 4 USER_DEFINED (all other added tokens), 6 BYTE (`<0xNN>`) |
| `tokenizer.ggml.merges` | [str] | 580604 entries `"left right"` in HF rank order (no token contains a space) |
| `tokenizer.ggml.bos_token_id` / `eos_token_id` / `unknown_token_id` / `padding_token_id` / `mask_token_id` / `seperator_token_id` | u32 | 2 / 1 / 3 / 0 / 4 / 1 |
| `tokenizer.ggml.add_bos_token` / `add_eos_token` | bool | false / false (the encoding adds CLS/SEP itself) |
| `tokenizer.ggml.add_space_prefix` | bool | true |
| `julia1.tokenizer.added_tokens` | [str] | the 249 added-token strings (matched on raw text before normalisation) |
| `julia1.tokenizer.added_token_ids` | [i32] | their ids, same order |

### 2.2 Tensors

GGUF stores a PyTorch `[out, in]` matrix with `ne = [in, out]`; `ggml_mul_mat(W, x)` then yields `out` rows. Shapes
below are PyTorch shapes.

Encoder (identical names in the encoder-only file):

| GGUF name | PyTorch source | shape | dtype |
| --- | --- | --- | --- |
| `token_embd.weight` | `encoder.embeddings.tok_embeddings.weight` | [256000, 384] | TYPE |
| `token_embd_norm.weight` | `encoder.embeddings.norm.weight` | [384] | F32 |
| `blk.{i}.attn_norm.weight` (i = 1..21 only) | `encoder.layers.{i}.attn_norm.weight` | [384] | F32 |
| `blk.{i}.attn_qkv.weight` | `encoder.layers.{i}.attn.Wqkv.weight` (rows 0:384 = Q, 384:768 = K, 768:1152 = V) | [1152, 384] | TYPE |
| `blk.{i}.attn_output.weight` | `encoder.layers.{i}.attn.Wo.weight` | [384, 384] | TYPE |
| `blk.{i}.ffn_norm.weight` | `encoder.layers.{i}.mlp_norm.weight` | [384] | F32 |
| `blk.{i}.ffn_up.weight` | `encoder.layers.{i}.mlp.Wi.weight` (rows 0:1152 = GELU branch, 1152:2304 = linear gate) | [2304, 384] | TYPE |
| `blk.{i}.ffn_down.weight` | `encoder.layers.{i}.mlp.Wo.weight` | [384, 1152] | TYPE |
| `output_norm.weight` | `encoder.final_norm.weight` | [384] | F32 |

Decision head (full file only; always F32):

| GGUF name | PyTorch source | shape |
| --- | --- | --- |
| `julia1.type_embd.weight` | `type_emb.weight` | [3, 384] |
| `julia1.head.{j}.attn_norm.weight` / `.bias` | `head.layers.{j}.norm1.*` | [384] |
| `julia1.head.{j}.attn_qkv.weight` / `.bias` | `head.layers.{j}.self_attn.in_proj_weight` / `in_proj_bias` (Q, K, V stacked) | [1152, 384] / [1152] |
| `julia1.head.{j}.attn_output.weight` / `.bias` | `head.layers.{j}.self_attn.out_proj.*` | [384, 384] / [384] |
| `julia1.head.{j}.ffn_norm.weight` / `.bias` | `head.layers.{j}.norm2.*` | [384] |
| `julia1.head.{j}.ffn_up.weight` / `.bias` | `head.layers.{j}.linear1.*` | [1536, 384] / [1536] |
| `julia1.head.{j}.ffn_down.weight` / `.bias` | `head.layers.{j}.linear2.*` | [384, 1536] / [384] |
| `julia1.scorer.norm.weight` / `.bias` | `scorer.0.*` | [384] |
| `julia1.scorer.up.weight` / `.bias` | `scorer.1.*` | [384, 384] / [384] |
| `julia1.scorer.out.weight` / `.bias` | `scorer.3.*` | [1, 384] / [1] |

Dtype policy: `TYPE` applies to the five encoder matrix families only. All 1-D tensors and every `julia1.*` tensor stay
F32 in every variant (≈15 MB). `F32` variant: everything F32. `BF16`/`F16`: encoder matrices in that type. `Q8_0`,
`Q5_0`, `Q4_0`: quantised with `gguf.quants.quantize` (block 32; all row lengths 384/1152/2304 are multiples of 32).
`token_embd.weight` follows TYPE like the other matrices (it holds 98.3 M of the 144 M parameters).

Tensor count: full file = 1 + 1 + 21 + 22·5 + 1 = 134 encoder tensors + 1 + 2·12 + 6 = 31 head tensors = **165**.
Encoder-only file = **134**.

### 2.3 Per-tensor overrides

`--override 'pattern=TYPE[,pattern=TYPE...]'` (repeatable) sets per-tensor types after the base TYPE: `fnmatch` on the
GGUF tensor names, 2-D tensors only, last matching rule wins. 1-D tensors always stay F32; `julia1.*` 2-D tensors change
only when a pattern matches them explicitly, and julia1-cli requires every `julia1.*` and 1-D tensor to be F32. A file
written with overrides carries `julia1.quantization.recipe`; `general.file_type` keeps the base TYPE's value. The
published recipe is `Julia-1-F16-embdQ8_0.gguf`: `token_embd.weight` in Q8_0, everything else as in F16.

## 3. Encoder-only file `Julia-1-encoder-<TYPE>.gguf`

Same encoder tensors, `general.architecture = "modern-bert"`, and the hparams stock llama.cpp's `ModernBertModel`
converter would write, under the `modern-bert.` prefix: `context_length 8192`, `embedding_length 384`, `block_count 22`,
`feed_forward_length 1152`, `attention.head_count 6`, `attention.head_count_kv 6`, `attention.layer_norm_epsilon 1e-5`,
`rope.freq_base 160000`, `rope.freq_base_swa 160000`, `rope.dimension_count 64`, `attention.sliding_window 128`,
`attention.sliding_window_pattern 3`, `rope.scaling.type none`, `vocab_size 256000`, `hidden_activation gelu`,
`attention.causal false`, `pooling_type 0` (none). Same tokenizer block as §2.1 (`add_bos_token=false`,
`add_eos_token=false`, `add_space_prefix=true`, `julia1.tokenizer.added_tokens` / `added_token_ids`).

Known deviations of stock llama.cpp on this file: (a) its SPM tokenizer merges over the whole string and prefixes
exactly one space, while HF splits at every `▁` (runs of spaces, tabs, newlines and added tokens differ); (b) its GeGLU
uses the tanh GELU approximation. Users who need exact parity feed token ids and compare hidden states; the
encoder-only file is an embeddings and validation vehicle, not the decision model.

## 4. Tokenizer (must equal HF `tokenizers` on `tokenizer.json`)

Configuration: normalizer `Replace(" " → "▁")`; pre-tokenizer `Metaspace(replacement="▁", prepend_scheme="always",
split=true)`; model BPE (`byte_fallback=true`, `fuse_unk=true`, `ignore_merges=false`, no dropout, no prefix/suffix);
249 added tokens, all `normalized=false, single_word=false, rstrip=false`, `lstrip=false` except `<mask>`
(`lstrip=true`, irrelevant because the encoding strips `<mask>` from every input). The encoding always calls
`add_special_tokens=False`, so the post-processor never runs.

Algorithm `encode(text) -> ids`:

1. **Added-token matching on the raw text**, leftmost-longest (Aho-Corasick semantics), over all 249 added tokens
   (special and not): `\n`×1..31 (ids 108–138), `▁`×2..31 (139–168), HTML tags `<table>`…`</code>` (169–216),
   `\t`×2..31 (255969–255998), `[toxicity=0]`, `<2mass>`, `[@BOS@]`, `<unused0..>`, `<start_of_turn>`,
   `<end_of_turn>`, `<pad> <eos> <bos> <unk> <mask>`. Each match emits its id; the raw-text gaps go to step 2.
2. For each gap: replace every `' '` (U+0020) with `▁` (U+2581); if the result does not start with `▁`, prepend one.
   Split into chunks so that every `▁` starts a new chunk (`"▁a▁▁b"` → `["▁a", "▁", "▁b"]`).
3. For each chunk: split into Unicode code points; if a code point is not in the vocabulary, replace it by its UTF-8
   bytes as `<0xNN>` tokens (byte fallback; `<unk>` is never emitted for valid UTF-8). Then apply BPE: repeatedly merge
   the adjacent pair with the lowest merge rank (`tokenizer.ggml.merges` order); ties resolved by leftmost position.
   Emit ids.

Reference outcomes: `"hello"` and `" hello"` → `[25612]`; `"  hello"` → `[235248, 25612]`; `"a  b   c"` →
`[476, 235248, 518, 235248, 235248, 498]`; `"Hello world\n\nNew para"` → `[25957, 2134, 109, 1622, 1301]`;
`"\thello"` → `[235248, 226, 17534]`; `"1234567"` → `[235248, 235274, 235284, 235304, 235310, 235308, 235318, 235324]`;
`"<unused0> y"` → `[7, 597]`.

Acceptance: 0 mismatching sequences against HF `tokenizers` on `data/tokenizer-corpus.jsonl` (4595 strings: every
state, question, option and head string of both reference sets, `" "` + option, plus fuzz strings with whitespace runs,
tabs, newlines, tags, digits, emoji, CJK, Arabic, Devanagari and out-of-vocabulary code points).

## 5. Request encoding (`julia/data.py::sequence`)

Inputs: `state` (str, or a JSON object or list), `question` (str), `options` (2–20 non-empty str), `type` ∈
{choice, score, noul} (noul requires exactly 2 options, ordered false/true), `max_length` (≤ 8192), `head_length`
(must satisfy `head_length + 4 < max_length`), `strict` (bool).

```
MASK=4  CLS=2  SEP=1
clean(t)   = t.replace("<mask>", " ")
enc(t)     = tokenizer.encode(clean(t))          # §4, no specials
head       = enc(f"{type} question: {question}")
option_ids = [enc(" " + o) for o in options]
strict: reject if any option_ids[i] longer than 48 tokens
options    = [[MASK] + o[:48] for o in option_ids]
budget     = head_length - sum(len(o) for o in options)
if budget < 16:
    per_option = max(4, (head_length - 16) // len(options))
    options    = [o[:per_option] for o in options]
    budget     = head_length - sum(len(o) for o in options)
strict: reject if len(head) > budget or any option was truncated
ids        = [CLS] + head[:max(8, budget)] + [SEP]
markers    = []
for o in options: markers.append(len(ids)); ids += o
ids       += [SEP]
state_ids  = enc(clean(state))
room       = max_length - len(ids) - 1
reject if room < 1
strict: reject if len(state_ids) > room
ids       += state_ids[:room] + [SEP]
qtype      = {"choice": 0, "score": 1, "noul": 2}[type]
```

Strict mode also rejects any input containing the literal `<mask>`.

Validation and serialisation rules (upstream `validate_row` and `FastEngine`):

* **JSON states** are serialised like Python `json.dumps(state, ensure_ascii=False)` with the default separators
  `", "` and `": "`. Integers of any size are written verbatim (`123456789012345678901234567890`); floats follow Python
  `repr` (exponent form iff decpt ≤ −4 or > 16), `-0.0` stays `-0.0`, `-0` becomes `0`, `1E2` becomes `100.0`; a
  duplicate key keeps the last value at the first key's position. NaN and infinite floats are rejected
  (`Out of range float values are not JSON compliant`).
* `target`, if present, must be an int (not a bool) with `0 <= target < len(options)`
  (`target must index the supplied option list`); `teacher_logits`, if not null, must be a list of exactly
  `len(options)` finite numbers (`teacher logits must be finite and match option count/order`).
* `max_length` is validated when the engine is created: `1 ≤ max_length ≤ julia1.context_length`, else
  `max_length must be an integer between 1 and 8192`.
* Defaults are those of upstream `julia.load_model`: `max_length=None` (→ `julia1.context_length` = 8192),
  `head_length=256`, strict off. Upstream's recommended policy (`inference-policy.json`, recorded in
  `julia1.encoding.default_*`) is 8192 / 512 / strict, which `julia1-cli serve` uses by default. The evaluation
  protocol (§8) is 1024 / 512 (typed) or 256 (parity) / strict.

## 6. Forward pass (single unpadded sequence of n tokens, positions 0..n−1)

All arithmetic in F32 (weights dequantised on the fly). `LN(x; w, b)` = `(x − mean) / sqrt(var + 1e-5) * w (+ b)`,
variance biased (divide by 384).

Encoder:
```
x = LN(tok_embd[ids]; token_embd_norm)
for il in 0..21:
    h   = x if il == 0 else LN(x; blk.il.attn_norm)
    qkv = h @ Wqkv.T                       # [n, 1152] → q,k,v [n, 6, 64]
    q,k = rope_neox(q,k, pos, theta=160000, dims=64)   # pairs (d, d+32), angle pos*theta^(-2d/64)
    S   = q·k / 8                          # per head [n, n]
    if il % 3 != 0: S[i,j] = -inf where |i − j| > 64
    x   = x + (softmax(S) @ v).merge_heads @ Wo.T
    h   = LN(x; blk.il.ffn_norm)
    u   = h @ Wi.T                         # [n, 2304]
    x   = x + (gelu_erf(u[:, :1152]) * u[:, 1152:]) @ Wo_mlp.T
hidden = LN(x; output_norm)
```

Head (upstream's `marker_only_head` evaluates the last layer at the markers only, with identical math):
```
h = hidden + type_embd[qtype]              # added to every position
for j in 0..1:
    a  = LN(h; head.j.attn_norm.w, .b)
    qkv = a @ in_proj.T + in_proj_b        # q,k,v [n, 6, 64]; no RoPE, no mask
    h  = h + (softmax(q·k / 8) @ v).merge_heads @ out_proj.T + out_proj_b
    f  = LN(h; head.j.ffn_norm.w, .b)
    h  = h + relu(f @ linear1.T + b1) @ linear2.T + b2
m = h[markers]                             # [k, 384]
s = LN(m; scorer.norm) @ scorer.up.T + b → gelu_erf → @ scorer.out.T + b   # [k]
logits = s (option order preserved)
```

## 7. Outputs and API semantics (mirror `julia/typed.py`, `julia/inference.py`)

* `logits(rows)` → list of `len(options)` floats each; must be finite (`Inference returned nonfinite logits`).
* `probabilities = softmax(logits)`; `choice = argmax`; `score = Σ i·p_i` (expected zero-based index; accuracy still
  uses argmax); `noul = p[1]`; `max_probability = max(p)` for choice/score.
* Named-question API `predict(state=..., questions={id: {type, instructions, criteria}})`: choice criteria = mapping
  id → description (option order = mapping order); score criteria = ordered list; noul criteria = optional mapping of
  `false`/`true` → descriptions, otherwise literal `"false"`, `"true"`; `question = instructions`. Validation errors
  carry upstream's messages, with the `JSONL line N: ` prefix that `engine.logits` adds.
* Legacy list API `predict(rows)` returns `index` and *display* probabilities: if the winner > 0.95 and every other
  < 0.045 → one-hot; otherwise zero out values < 0.01 and renormalise.

## 8. Validation protocol and acceptance thresholds

Golden reference: upstream's reference code (torch 2.14.0, transformers 5.0.0, CPU FP32, one example per forward,
Collator padding to a multiple of 8, `marker_only_head=False`) produced `data/reference/typed.jsonl` (the 2000
questions of the [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) test set,
`max_length` 1024, `head_length` 512, strict) and `data/reference/parity.jsonl` (the 100 parity cases of
[SupersonicLabs/Julia-1-ONNX](https://huggingface.co/SupersonicLabs/Julia-1-ONNX), `max_length` 1024, `head_length`
256, strict). Each row carries `id`, `type`, `keys`, `gold`, `ids`, `markers`, `qtype`, `logits` and the raw
`request`; parity rows also carry `upstream_logits` (the PyTorch logits stored upstream, 1.03e-4 from the reference).
The reference reproduces upstream's published CPU FP32 result on the typed set: 1451/2000 (choice 426/600, noul
483/600, score 542/800).

Each runtime × file is evaluated two ways: (a) **replay** — feed the reference `ids/markers/qtype` and compare logits;
(b) **end-to-end** — feed the raw request and let the runtime tokenise and encode (checks §4–§5 too; the encoding must
equal the reference `ids/markers/qtype` exactly). `tools/eval_gguf.py` and `tools/compare_replay.py` apply these
thresholds:

| Runtime and file | max abs Δlogit | typed argmax agreement | typed accuracy | parity argmax |
| --- | --- | --- | --- | --- |
| F32, exact (Metal `--precise`, numpy) | ≤ 1e-3 | 2000/2000 | 1451 | 100/100, ≤ 1e-3 |
| F32, CPU | ≤ 2e-3 | 2000/2000 | 1451 | 100/100, ≤ 2e-3 |
| F16 (exact or `--fast`) | reported | 2000/2000 | 1451 ± 3 | 100/100 |
| F16-embdQ8_0 (mixed recipe) | reported | ≥ 1990/2000 | 1451 ± 5 | reported |
| BF16, Q8_0, Q5_0, Q4_0 | reported | reported | reported | reported |

Max abs is an acceptance criterion for F32 only: rounding the weights to F16, BF16 or Q8_0 moves the logits by 0.04 to
2 even in an exact runtime (upstream PyTorch with identically rounded weights shows the same deviation), so the other
files are accepted on decisions. BF16 (1994/2000) and plain Q8_0 (about 1940/2000) do not meet the F16 and mixed-recipe
bars, and Q5_0/Q4_0 lose accuracy; they are not published.

The encoder-only file in stock llama.cpp is compared on hidden states for identical token ids (looser: tanh GELU).

## 9. Non-goals

Router / hierarchical grouping (`julia/router/router.py`), CUDA INT8 and Bend kernels, `act_head`.
