"""SPEC §4 tokenizer, twice: an HF `tokenizers` object rebuilt from GGUF metadata, and a pure-Python twin.

Both take the metadata dict of `reader.GGUFModel` and expose `encode(text) -> list[int]`
(plain tokenisation, no CLS/SEP, i.e. HF `add_special_tokens=False`).
"""
from __future__ import annotations

import json
import re

META = '▁'
CONTROL_TYPE = 3  # gguf.TokenType.CONTROL: the 7 special added tokens (109 in the llama.cpp layout; encoding is the same)
# Rust `char::is_whitespace` (Unicode White_Space), used by the `lstrip` flag of <mask>.
WHITESPACE = frozenset('\t\n\x0b\x0c\r \x85\xa0     　'
                       + ''.join(chr(c) for c in range(0x2000, 0x200b)))


def _split_merge(merge):
    left, right = merge.split(' ')  # no token contains a space (converter asserts it)
    return left, right


def hf_tokenizer_json(meta) -> dict:
    """The tokenizer.json dict equivalent to upstream `tokenizer/tokenizer.json`."""
    tokens = meta['tokenizer.ggml.tokens']
    types = meta['tokenizer.ggml.token_type']
    mask = tokens[meta['tokenizer.ggml.mask_token_id']]
    added = [dict(id=i, content=t, single_word=False, lstrip=t == mask, rstrip=False, normalized=False,
                  special=types[i] == CONTROL_TYPE)
             for t, i in zip(meta['julia1.tokenizer.added_tokens'], meta['julia1.tokenizer.added_token_ids'])]
    return dict(
        version='1.0', truncation=None, padding=None, added_tokens=added,
        normalizer=dict(type='Replace', pattern=dict(String=' '), content=META),
        pre_tokenizer=dict(type='Metaspace', replacement=META, prepend_scheme='always', split=True),
        post_processor=None,
        decoder=dict(type='Sequence', decoders=[dict(type='Replace', pattern=dict(String=META), content=' '),
                                                dict(type='ByteFallback'), dict(type='Fuse')]),
        model=dict(type='BPE', dropout=None, unk_token=tokens[meta['tokenizer.ggml.unknown_token_id']],
                   continuing_subword_prefix=None, end_of_word_suffix=None, fuse_unk=True, byte_fallback=True,
                   ignore_merges=False, vocab={t: i for i, t in enumerate(tokens)},
                   merges=[list(_split_merge(m)) for m in meta['tokenizer.ggml.merges']]))


class HfTokenizer:
    def __init__(self, meta):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_str(json.dumps(hf_tokenizer_json(meta), ensure_ascii=False))

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids


class PureTokenizer:
    """Literal SPEC §4: added-token matching, Metaspace, byte fallback, rank-based BPE."""

    def __init__(self, meta):
        tokens = meta['tokenizer.ggml.tokens']
        self.vocab = {t: i for i, t in enumerate(tokens)}
        self.ranks = {}
        for rank, merge in enumerate(meta['tokenizer.ggml.merges']):
            self.ranks[_split_merge(merge)] = rank  # a repeated pair keeps the later rank, like HF's HashMap
        self.added = dict(zip(meta['julia1.tokenizer.added_tokens'], meta['julia1.tokenizer.added_token_ids']))
        # `re` alternation is leftmost-first; ordering alternatives longest-first makes it leftmost-longest.
        self.added_re = re.compile('|'.join(re.escape(t) for t in sorted(self.added, key=len, reverse=True)))
        self.lstrip = {tokens[meta['tokenizer.ggml.mask_token_id']]}
        # Byte fallback tokens; only <0x09> is absent from this vocab, and U+0009 itself is a vocab
        # token, so HF's <unk> fallback can never trigger on valid UTF-8 and is not implemented.
        self.bytes = {b: self.vocab[f'<0x{b:02X}>'] for b in range(256) if f'<0x{b:02X}>' in self.vocab}
        self.cache = {}

    def encode(self, text):
        ids = []
        pos = 0
        for match in self.added_re.finditer(text):
            start, end = match.span()
            if match.group() in self.lstrip:
                while start > pos and text[start - 1] in WHITESPACE:  # never before the previous match
                    start -= 1
            if start > pos:
                self._gap(text[pos:start], ids)
            ids.append(self.added[match.group()])
            pos = end
        if pos < len(text):
            self._gap(text[pos:], ids)
        return ids

    def _gap(self, gap, ids):
        gap = gap.replace(' ', META)
        if not gap.startswith(META):
            gap = META + gap
        for piece in gap.split(META)[1:]:  # every ▁ starts a chunk; consecutive ▁ give lone-▁ chunks
            ids.extend(self._bpe(META + piece))

    def _bpe(self, chunk):
        cached = self.cache.get(chunk)
        if cached is not None:
            return cached
        symbols = []
        for char in chunk:
            if char in self.vocab:
                symbols.append(char)
            else:
                symbols.extend(f'<0x{b:02X}>' for b in char.encode('utf-8'))
        while len(symbols) > 1:
            best = None
            for i in range(len(symbols) - 1):
                rank = self.ranks.get((symbols[i], symbols[i + 1]))
                if rank is not None and (best is None or rank < best[0]):  # strict <: leftmost wins ties
                    best = (rank, i)
            if best is None:
                break
            i = best[1]
            symbols[i:i + 2] = [symbols[i] + symbols[i + 1]]
        result = [self.vocab[s] for s in symbols]
        self.cache[chunk] = result
        return result


def build_tokenizer(meta, impl='auto'):
    if impl == 'auto':
        try:
            import tokenizers  # noqa: F401
            impl = 'hf'
        except ImportError:
            impl = 'pure'
    if impl == 'hf':
        return HfTokenizer(meta)
    if impl == 'pure':
        return PureTokenizer(meta)
    raise ValueError(f'unknown tokenizer implementation {impl!r} (auto, hf, pure)')
