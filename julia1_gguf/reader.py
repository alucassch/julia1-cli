"""GGUF access: metadata as plain Python values, tensors as float32 arrays in PyTorch [out, in] layout.

GGUF stores a PyTorch [out, in] matrix with ne = [in, out]; `GGUFReader` already
exposes `tensor.data` with the rows-first (numpy) shape, i.e. [out, bytes_per_row]
for quantised types and [out, in] for F32/F16, so `gguf.quants.dequantize`
returns [out, in] directly (checked for F32, F16, BF16, Q8_0, Q5_0, Q4_0).

Two layouts load: julia1 (tools/convert_julia1_to_gguf.py) as is, and llama.cpp's (convert_hf_to_gguf.py:
architecture modern-bert with a laya decision head and the Julia prompt template), which is presented with the julia1
metadata keys and tensor names the rest of the package reads. Same weights, same view, bit-identical logits.
"""
from __future__ import annotations

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

META = '▁'
# tokenizer.chat_template.systemone of Julia-1 (Laya, of the same family, has another template)
JULIA_TEMPLATE = ('<bos>{{ type }} question: {{ instructions if instructions is string else instructions | tojson }}<eos>'
                  '{% for o in options %}<mask> {% if o.description %}{{ o.description if o.description is string else '
                  'o.description | tojson }}{% else %}{{ o.key }}{% endif %}{% endfor %}<eos>'
                  '{{ state if state is string else state | tojson }}<eos>')


class GGUFModel:
    def __init__(self, path):
        self.path = str(path)
        self.reader = GGUFReader(self.path)
        self.metadata = {name: field.contents() for name, field in self.reader.fields.items()
                         if not name.startswith('GGUF.')}
        self._tensors = {t.name: t for t in self.reader.tensors}
        arch = self.metadata.get('general.architecture')
        if arch == 'modern-bert':
            self._llamacpp_view()
        elif arch != 'julia1':
            raise ValueError(f"general.architecture is '{arch}', expected 'julia1' or 'modern-bert' (a full Julia-1 "
                             'GGUF: tools/convert_julia1_to_gguf.py or llama.cpp convert_hf_to_gguf.py)')

    def _llamacpp_view(self):
        m = self.metadata
        decision = m.get('modern-bert.decision.type')
        if decision != 'laya':
            raise ValueError(f"modern-bert.decision.type is {decision!r}, expected 'laya' (a full Julia-1 GGUF with the "
                             'decision head, not an encoder-only one)')
        if m.get('tokenizer.chat_template.systemone') != JULIA_TEMPLATE:
            raise ValueError('tokenizer.chat_template.systemone is not the Julia-1 template (another model of the family?)')
        n_layer = m['modern-bert.block_count'] - m['modern-bert.decision.block_count']
        tokens = m['tokenizer.ggml.tokens']
        types = m['tokenizer.ggml.token_type']
        # The 249 added tokens of upstream tokenizer.json: CONTROL (3) and USER_DEFINED (4) tokens, plus the runs of
        # 2+ '▁' that llama.cpp's converter turned NORMAL (upstream still matches them in raw text: '▁▁' -> [139]).
        added = [i for i, t in enumerate(tokens) if types[i] in (3, 4) or (len(t) >= 2 and t == META * len(t))]
        mask = m['tokenizer.ggml.mask_token_id']
        m.update({
            'julia1.context_length': m['modern-bert.context_length'],
            'julia1.embedding_length': m['modern-bert.embedding_length'],
            'julia1.block_count': n_layer,
            'julia1.attention.head_count': m['modern-bert.attention.head_count'],
            'julia1.attention.layer_norm_epsilon': m['modern-bert.attention.layer_norm_epsilon'],
            'julia1.attention.sliding_window': m['modern-bert.attention.sliding_window'],
            'julia1.attention.sliding_window_pattern': m['modern-bert.attention.sliding_window_pattern'],
            'julia1.rope.freq_base': m['modern-bert.rope.freq_base'],
            'julia1.rope.dimension_count': m['modern-bert.embedding_length'] // m['modern-bert.attention.head_count'],
            'julia1.head.block_count': m['modern-bert.decision.block_count'],
            'julia1.head.attention.head_count': m['modern-bert.attention.head_count'],
            'julia1.head.layer_norm_epsilon': m['modern-bert.attention.layer_norm_epsilon'],
            # the template's sequence: <bos> head <eos> (<mask> option)* <eos> state <eos>
            'julia1.encoding.cls_token_id': m['tokenizer.ggml.bos_token_id'],
            'julia1.encoding.sep_token_id': m['tokenizer.ggml.eos_token_id'],
            'julia1.encoding.marker_token_id': mask,
            'julia1.encoding.marker_token_text': tokens[mask],
            'julia1.encoding.head_template': '{type} question: {question}',
            'julia1.encoding.option_prefix': ' ',
            'julia1.encoding.option_token_limit': 48,
            'julia1.tokenizer.added_tokens': [tokens[i] for i in added],
            'julia1.tokenizer.added_token_ids': added,
        })
        names = {'token_types.weight': 'julia1.type_embd.weight'}
        for k in ('weight', 'bias'):
            names.update({f'cls.norm.{k}': f'julia1.scorer.norm.{k}', f'cls.{k}': f'julia1.scorer.up.{k}',
                          f'cls.output.{k}': f'julia1.scorer.out.{k}'})
            for j in range(m['modern-bert.decision.block_count']):
                for part in ('attn_norm', 'attn_qkv', 'attn_output', 'ffn_norm', 'ffn_up', 'ffn_down'):
                    names[f'blk.{n_layer + j}.{part}.{k}'] = f'julia1.head.{j}.{part}.{k}'
        self._tensors = {names.get(name, name): t for name, t in self._tensors.items()}

    def __getitem__(self, key):
        return self.metadata[key]

    def get(self, key, default=None):
        return self.metadata.get(key, default)

    @property
    def tensor_names(self):
        return list(self._tensors)

    def tensor_type(self, name):
        return self._tensors[name].tensor_type

    def tensor(self, name) -> np.ndarray:
        """Whole tensor dequantised to float32, PyTorch-shaped [out, in] (1-D tensors unchanged)."""
        t = self._tensors[name]
        return np.ascontiguousarray(dequantize(np.asarray(t.data), t.tensor_type), dtype=np.float32)

    def rows(self, name, ids) -> np.ndarray:
        """Rows `ids` of a 2-D tensor, dequantised to float32 [len(ids), in] without touching the rest."""
        t = self._tensors[name]
        picked = np.ascontiguousarray(t.data[np.asarray(ids, dtype=np.int64)])
        return np.ascontiguousarray(dequantize(picked, t.tensor_type), dtype=np.float32)
