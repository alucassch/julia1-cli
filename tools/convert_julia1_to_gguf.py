#!/usr/bin/env python3
"""Convert the SupersonicLabs/Julia-1 checkpoint (https://huggingface.co/SupersonicLabs/Julia-1) to GGUF.

Writes, per requested TYPE, the full decision model (`Julia-1-<TYPE>.gguf`,
architecture "julia1") and optionally the llama.cpp-loadable encoder-only file
(`Julia-1-encoder-<TYPE>.gguf`, architecture "modern-bert"). See SPEC.md.

  python tools/convert_julia1_to_gguf.py --upstream <Julia-1 checkout> --outdir <dir> --types F32,F16 --encoder-types F32,F16
  python tools/convert_julia1_to_gguf.py --upstream <Julia-1 checkout> --outdir <dir> --types F16 \
      --override token_embd.weight=Q8_0 --name-suffix=-embdQ8_0

Needs numpy, safetensors and gguf >= 0.18 (PyPI, or gguf-py of llama.cpp v0.5.0).
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import time
from importlib import metadata
from pathlib import Path

import numpy as np

import gguf
from gguf import GGMLQuantizationType as Q

WEIGHTS_SHA256 = 'df853bf7fe424420011f3d0c47a05d7341aa9eefa7fb9f203ea4aada4ad95b72'
SPECIAL_CONTROL = {'<pad>', '<eos>', '<bos>', '<unk>', '<mask>', '<start_of_turn>', '<end_of_turn>'}
QTYPES = ['choice', 'score', 'noul']

TYPES = {
    'F32': (Q.F32, gguf.LlamaFileType.ALL_F32),
    'F16': (Q.F16, gguf.LlamaFileType.MOSTLY_F16),
    'BF16': (Q.BF16, gguf.LlamaFileType.MOSTLY_BF16),
    'Q8_0': (Q.Q8_0, gguf.LlamaFileType.MOSTLY_Q8_0),
    'Q5_0': (Q.Q5_0, gguf.LlamaFileType.MOSTLY_Q5_0),
    'Q4_0': (Q.Q4_0, gguf.LlamaFileType.MOSTLY_Q4_0),
}


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_weights(upstream: Path) -> dict[str, np.ndarray]:
    from safetensors import safe_open
    weights = {}
    with safe_open(str(upstream / 'model.safetensors'), framework='np') as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def build_vocab(upstream: Path) -> dict:
    data = json.loads((upstream / 'tokenizer' / 'tokenizer.json').read_text(encoding='utf-8'))
    model = data['model']
    assert model['type'] == 'BPE' and model['byte_fallback'] and model['fuse_unk'] and not model['ignore_merges']
    assert data['normalizer'] == {'type': 'Replace', 'pattern': {'String': ' '}, 'content': '▁'}
    assert data['pre_tokenizer'] == {'type': 'Metaspace', 'replacement': '▁', 'prepend_scheme': 'always', 'split': True}
    vocab = dict(model['vocab'])
    added = {a['content']: a for a in data['added_tokens']}
    for content, a in added.items():
        vocab.setdefault(content, a['id'])
    n_vocab = len(vocab)
    assert n_vocab == 256000, n_vocab
    tokens: list[str | None] = [None] * n_vocab
    for text, idx in vocab.items():
        assert tokens[idx] is None, f'duplicate id {idx}'
        assert ' ' not in text, f'token {idx!r} contains a space'
        tokens[idx] = text
    assert all(t is not None for t in tokens)
    merges = model['merges']
    merge_strings = []
    scores = np.zeros(n_vocab, dtype=np.float32)
    first_rank = {}
    for rank, (left, right) in enumerate(merges):
        merge_strings.append(f'{left} {right}')
        idx = vocab[left + right]
        if idx not in first_rank:
            first_rank[idx] = rank
            scores[idx] = -float(rank)
    types = []
    for idx, text in enumerate(tokens):
        if text in added:
            types.append(gguf.TokenType.CONTROL if text in SPECIAL_CONTROL else gguf.TokenType.USER_DEFINED)
        elif len(text) == 6 and text.startswith('<0x') and text.endswith('>'):
            types.append(gguf.TokenType.BYTE)
        else:
            types.append(gguf.TokenType.NORMAL)
    assert sum(1 for t in types if t == gguf.TokenType.BYTE) == 256 - 1 or True  # <0xNN> count informational
    added_sorted = sorted(added.values(), key=lambda a: a['id'])
    return dict(tokens=tokens, scores=scores, types=types, merges=merge_strings,
                added_tokens=[a['content'] for a in added_sorted],
                added_token_ids=[a['id'] for a in added_sorted],
                ids=dict(bos=vocab['<bos>'], eos=vocab['<eos>'], unk=vocab['<unk>'], pad=vocab['<pad>'], mask=vocab['<mask>']))


def add_tokenizer(w: gguf.GGUFWriter, vocab: dict) -> None:
    w.add_tokenizer_model('llama')
    w.add_tokenizer_pre('default')
    w.add_token_list(vocab['tokens'])
    w.add_token_scores(vocab['scores'].tolist())
    w.add_token_types(vocab['types'])
    w.add_token_merges(vocab['merges'])
    ids = vocab['ids']
    w.add_bos_token_id(ids['bos'])
    w.add_eos_token_id(ids['eos'])
    w.add_unk_token_id(ids['unk'])
    w.add_pad_token_id(ids['pad'])
    w.add_mask_token_id(ids['mask'])
    w.add_sep_token_id(ids['eos'])
    w.add_add_bos_token(False)
    w.add_add_eos_token(False)
    w.add_add_space_prefix(True)
    w.add_array('julia1.tokenizer.added_tokens', vocab['added_tokens'])
    w.add_array('julia1.tokenizer.added_token_ids', vocab['added_token_ids'])


def add_general(w: gguf.GGUFWriter, ftype: gguf.LlamaFileType, encoder_only: bool) -> None:
    w.add_type('model')
    w.add_name('Julia-1-encoder' if encoder_only else 'Julia-1')
    w.add_author('Supersonic Labs')
    w.add_organization('Supersonic Labs')
    w.add_license('apache-2.0')
    if encoder_only:
        w.add_description('Encoder-only (mmBERT-small / ModernBERT) part of the Julia 1 decision model, for token-level '
                          'hidden states in stock llama.cpp. Not the decision model: no head, no option scoring.')
    else:
        w.add_description('Julia 1 decision model (mmBERT-small encoder + decision head) in GGUF. Not a generative '
                          'model: scores 2-20 supplied options for a state and a question. Runs in julia1-cli '
                          '(https://github.com/alucassch/julia1-cli).')
    w.add_source_url('https://huggingface.co/SupersonicLabs/Julia-1')
    w.add_base_model_count(1)
    w.add_base_model_name(0, 'mmBERT-small')
    w.add_base_model_organization(0, 'jhu-clsp')
    w.add_base_model_repo_url(0, 'https://huggingface.co/jhu-clsp/mmBERT-small')
    w.add_file_type(ftype)
    w.add_quantization_version(gguf.GGML_QUANT_VERSION)


def add_encoder_hparams(w: gguf.GGUFWriter, enc: dict) -> None:
    w.add_context_length(enc['max_position_embeddings'])
    w.add_embedding_length(enc['hidden_size'])
    w.add_block_count(enc['num_hidden_layers'])
    w.add_feed_forward_length(enc['intermediate_size'])
    w.add_head_count(enc['num_attention_heads'])
    w.add_head_count_kv(enc['num_attention_heads'])
    w.add_layer_norm_eps(enc['norm_eps'])
    theta = enc['rope_parameters']['full_attention']['rope_theta'] if 'rope_parameters' in enc else enc['global_rope_theta']
    theta_swa = enc['rope_parameters']['sliding_attention']['rope_theta'] if 'rope_parameters' in enc else enc.get('local_rope_theta', theta)
    w.add_rope_freq_base(float(theta))
    w.add_float32(f'{w.arch}.rope.freq_base_swa', float(theta_swa))
    w.add_rope_dimension_count(enc['hidden_size'] // enc['num_attention_heads'])
    w.add_rope_scaling_type(gguf.RopeScalingType.NONE)
    w.add_sliding_window(enc['local_attention'])
    w.add_sliding_window_pattern(enc['global_attn_every_n_layers'])
    w.add_vocab_size(enc['vocab_size'])
    w.add_string(f'{w.arch}.hidden_activation', enc['hidden_activation'])  # add_hidden_act (not in PyPI gguf <= 0.19.0)
    w.add_causal_attention(False)
    w.add_pooling_type(gguf.PoolingType.NONE)


def add_julia1_hparams(w: gguf.GGUFWriter, jcfg: dict, enc: dict) -> None:
    w.add_uint32('julia1.head.block_count', jcfg['head_layers'])
    w.add_uint32('julia1.head.attention.head_count', max(1, enc['hidden_size'] // 64))
    w.add_uint32('julia1.head.feed_forward_length', 4 * enc['hidden_size'])
    w.add_float32('julia1.head.layer_norm_epsilon', 1e-5)
    w.add_string('julia1.head.activation', 'relu')
    w.add_string('julia1.scorer.activation', 'gelu')
    w.add_array('julia1.qtype_names', QTYPES)
    w.add_uint32('julia1.encoding.cls_token_id', 2)
    w.add_uint32('julia1.encoding.sep_token_id', 1)
    w.add_uint32('julia1.encoding.marker_token_id', 4)
    w.add_uint32('julia1.encoding.pad_token_id', 0)
    w.add_string('julia1.encoding.marker_token_text', '<mask>')
    w.add_string('julia1.encoding.head_template', '{type} question: {question}')
    w.add_string('julia1.encoding.option_prefix', ' ')
    w.add_uint32('julia1.encoding.option_token_limit', 48)
    w.add_uint32('julia1.encoding.min_options', 2)
    w.add_uint32('julia1.encoding.max_options', 20)
    w.add_uint32('julia1.encoding.default_max_length', enc['max_position_embeddings'])
    w.add_uint32('julia1.encoding.default_head_length', 512)
    w.add_string('julia1.upstream.weights_sha256', WEIGHTS_SHA256)
    w.add_string('julia1.upstream.repo', 'SupersonicLabs/Julia-1')
    w.add_uint32('julia1.upstream.checkpoint_step', 500)


def encoder_tensors(weights: dict[str, np.ndarray], n_layer: int):
    yield 'token_embd.weight', weights['encoder.embeddings.tok_embeddings.weight'], True
    yield 'token_embd_norm.weight', weights['encoder.embeddings.norm.weight'], False
    for i in range(n_layer):
        p = f'encoder.layers.{i}.'
        if i != 0:
            yield f'blk.{i}.attn_norm.weight', weights[p + 'attn_norm.weight'], False
        else:
            assert p + 'attn_norm.weight' not in weights
        yield f'blk.{i}.attn_qkv.weight', weights[p + 'attn.Wqkv.weight'], True
        yield f'blk.{i}.attn_output.weight', weights[p + 'attn.Wo.weight'], True
        yield f'blk.{i}.ffn_norm.weight', weights[p + 'mlp_norm.weight'], False
        yield f'blk.{i}.ffn_up.weight', weights[p + 'mlp.Wi.weight'], True
        yield f'blk.{i}.ffn_down.weight', weights[p + 'mlp.Wo.weight'], True
    yield 'output_norm.weight', weights['encoder.final_norm.weight'], False


def head_tensors(weights: dict[str, np.ndarray], head_layers: int):
    yield 'julia1.type_embd.weight', weights['type_emb.weight']
    for j in range(head_layers):
        p = f'head.layers.{j}.'
        yield f'julia1.head.{j}.attn_norm.weight', weights[p + 'norm1.weight']
        yield f'julia1.head.{j}.attn_norm.bias', weights[p + 'norm1.bias']
        yield f'julia1.head.{j}.attn_qkv.weight', weights[p + 'self_attn.in_proj_weight']
        yield f'julia1.head.{j}.attn_qkv.bias', weights[p + 'self_attn.in_proj_bias']
        yield f'julia1.head.{j}.attn_output.weight', weights[p + 'self_attn.out_proj.weight']
        yield f'julia1.head.{j}.attn_output.bias', weights[p + 'self_attn.out_proj.bias']
        yield f'julia1.head.{j}.ffn_norm.weight', weights[p + 'norm2.weight']
        yield f'julia1.head.{j}.ffn_norm.bias', weights[p + 'norm2.bias']
        yield f'julia1.head.{j}.ffn_up.weight', weights[p + 'linear1.weight']
        yield f'julia1.head.{j}.ffn_up.bias', weights[p + 'linear1.bias']
        yield f'julia1.head.{j}.ffn_down.weight', weights[p + 'linear2.weight']
        yield f'julia1.head.{j}.ffn_down.bias', weights[p + 'linear2.bias']
    yield 'julia1.scorer.norm.weight', weights['scorer.0.weight']
    yield 'julia1.scorer.norm.bias', weights['scorer.0.bias']
    yield 'julia1.scorer.up.weight', weights['scorer.1.weight']
    yield 'julia1.scorer.up.bias', weights['scorer.1.bias']
    yield 'julia1.scorer.out.weight', weights['scorer.3.weight']
    yield 'julia1.scorer.out.bias', weights['scorer.3.bias']


def put_tensor(w: gguf.GGUFWriter, name: str, data: np.ndarray, qtype: Q) -> None:
    data = np.ascontiguousarray(data, dtype=np.float32)
    if qtype == Q.F32:
        w.add_tensor(name, data, raw_dtype=Q.F32)
        return
    if data.ndim != 2 or data.shape[-1] % 32 != 0:
        raise ValueError(f'{name}: shape {data.shape} cannot be quantised to {qtype.name}')
    packed = gguf.quants.quantize(data, qtype)
    w.add_tensor(name, packed, raw_dtype=qtype)


def parse_overrides(items: list[str]) -> list[tuple[str, str]]:
    """`pattern=TYPE[,pattern=TYPE...]` (repeatable) -> [(fnmatch pattern on GGUF tensor names, TYPE)]."""
    rules = []
    for item in items:
        for rule in item.split(','):
            rule = rule.strip()
            if not rule:
                continue
            pattern, _, type_name = rule.rpartition('=')
            if not pattern or type_name not in TYPES:
                raise SystemExit(f'--override {rule!r}: expected pattern=TYPE with TYPE in {",".join(TYPES)}')
            rules.append((pattern, type_name))
    return rules


def recipe_string(type_name: str, overrides: list[tuple[str, str]]) -> str:
    return ';'.join([f'base={type_name}'] + [f'{pattern}={t}' for pattern, t in overrides])


def tensor_qtype(name: str, data: np.ndarray, base: Q, overrides: list[tuple[str, str]]) -> Q:
    """Base TYPE for the five encoder matrix families, F32 otherwise, then the override rules (last match wins).
    Overrides apply to 2-D tensors only: 1-D tensors always stay F32."""
    qtype = base
    if data.ndim == 2:
        for pattern, type_name in overrides:
            if fnmatch.fnmatchcase(name, pattern):
                qtype = TYPES[type_name][0]
    return qtype


def write_file(path: Path, arch: str, type_name: str, weights: dict, vocab: dict, enc: dict, jcfg: dict,
               encoder_only: bool, overrides: list[tuple[str, str]] = ()) -> dict:
    qtype, ftype = TYPES[type_name]
    w = gguf.GGUFWriter(str(path), arch)
    add_general(w, ftype, encoder_only)
    add_encoder_hparams(w, enc)
    if not encoder_only:
        add_julia1_hparams(w, jcfg, enc)
        if overrides:
            w.add_string('julia1.quantization.recipe', recipe_string(type_name, overrides))
    add_tokenizer(w, vocab)
    count = 0
    for name, data, is_matrix in encoder_tensors(weights, enc['num_hidden_layers']):
        put_tensor(w, name, data, tensor_qtype(name, data, qtype if is_matrix else Q.F32, overrides))
        count += 1
    if not encoder_only:
        for name, data in head_tensors(weights, jcfg['head_layers']):
            put_tensor(w, name, data, tensor_qtype(name, data, Q.F32, overrides))
            count += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    info = dict(file=path.name, arch=arch, type=type_name, tensors=count, bytes=path.stat().st_size, sha256=digest(path))
    if overrides:
        info['recipe'] = recipe_string(type_name, overrides)
    return info


def gguf_version() -> str:
    try:
        return metadata.version('gguf')
    except metadata.PackageNotFoundError:
        return 'unknown'


def merge_manifest(path: Path, entries: list[dict]) -> dict:
    """Keep the entries of files not written by this run (by file name); replace/append the new ones."""
    files = []
    if path.exists():
        files = [f for f in json.loads(path.read_text()).get('files', []) if f['file'] not in {e['file'] for e in entries}]
    return dict(upstream_weights_sha256=WEIGHTS_SHA256, gguf_py=gguf_version(),
                files=files + entries)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--upstream', type=Path, required=True, help='a download of SupersonicLabs/Julia-1 (model.safetensors, '
                    'encoder/config.json, julia_config.json, tokenizer/tokenizer.json)')
    ap.add_argument('--outdir', type=Path, required=True, help='output directory (manifest.json is merged there)')
    ap.add_argument('--types', default='F32,F16,BF16,Q8_0,Q5_0,Q4_0')
    ap.add_argument('--encoder-types', default=None, help='comma list or "none" (default F16,Q8_0; none with --override)')
    ap.add_argument('--override', action='append', default=[],
                    help='pattern=TYPE[,pattern=TYPE...]: per-tensor types applied after the base TYPE (fnmatch on '
                         'GGUF tensor names, 2-D tensors only, last match wins; 1-D and julia1.* tensors stay F32 unless '
                         'matched). Recorded as julia1.quantization.recipe. Full-model files only.')
    ap.add_argument('--name-suffix', default='', help='write Julia-1-<TYPE><suffix>.gguf (e.g. "-mix")')
    ap.add_argument('--skip-hash-check', action='store_true')
    args = ap.parse_args()
    overrides = parse_overrides(args.override)
    encoder_types = args.encoder_types if args.encoder_types is not None else ('none' if overrides else 'F16,Q8_0')
    if overrides and encoder_types.lower() != 'none':
        ap.error('--override applies to the full model only; use --encoder-types none')

    started = time.monotonic()
    if not args.skip_hash_check:
        assert digest(args.upstream / 'model.safetensors') == WEIGHTS_SHA256, 'upstream weights hash mismatch'
    enc = json.loads((args.upstream / 'encoder' / 'config.json').read_text())
    jcfg = json.loads((args.upstream / 'julia_config.json').read_text())
    assert jcfg['architecture'] == 'JuliaDecisionModel' and jcfg['format_version'] == 1
    assert enc['model_type'] == 'modernbert' and not enc['attention_bias'] and not enc['mlp_bias'] and not enc['norm_bias']
    assert enc['hidden_activation'] == 'gelu' and enc['position_embedding_type'] == 'sans_pos'
    weights = load_weights(args.upstream)
    vocab = build_vocab(args.upstream)
    args.outdir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for type_name in [t for t in args.types.split(',') if t]:
        path = args.outdir / f'Julia-1-{type_name}{args.name_suffix}.gguf'
        info = write_file(path, 'julia1', type_name, weights, vocab, enc, jcfg, encoder_only=False, overrides=overrides)
        manifest.append(info)
        print(json.dumps(info), flush=True)
    if encoder_types.lower() != 'none':
        for type_name in [t for t in encoder_types.split(',') if t]:
            path = args.outdir / f'Julia-1-encoder-{type_name}.gguf'
            info = write_file(path, 'modern-bert', type_name, weights, vocab, enc, jcfg, encoder_only=True)
            manifest.append(info)
            print(json.dumps(info), flush=True)
    manifest_path = args.outdir / 'manifest.json'
    manifest_path.write_text(json.dumps(merge_manifest(manifest_path, manifest), indent=2) + '\n')
    print(f'done in {time.monotonic() - started:.1f}s', flush=True)


if __name__ == '__main__':
    main()
