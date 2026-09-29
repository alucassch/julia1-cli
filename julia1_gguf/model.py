"""SPEC §6 forward pass in float32 numpy for one unpadded sequence.

GELU(erf) note: numpy has no erf. `erf` below is a cubic Taylor expansion of
erf around the nearest node of a 1/256-spaced table on [0, 6] (values and
derivatives from math.erf/exp at import time), evaluated in float64: its max abs
error against math.erf over [-12, 12] is below 3e-12, so gelu is within one
float32 ulp of the exact value after the cast. No scipy needed.
"""
from __future__ import annotations

import math

import numpy as np

F32 = np.float32

_ERF_STEP = 1.0 / 256
_ERF_X = np.arange(6 * 256 + 1) * _ERF_STEP
_ERF_Y = np.array([math.erf(v) for v in _ERF_X])
_ERF_D1 = 2.0 / math.sqrt(math.pi) * np.exp(-np.square(_ERF_X))  # first derivative
_ERF_D2 = -_ERF_X * _ERF_D1                                       # second derivative / 2
_ERF_D3 = (2.0 * np.square(_ERF_X) - 1.0) / 3.0 * _ERF_D1         # third derivative / 6


def erf(x):
    x = np.asarray(x, dtype=np.float64)
    ax = np.minimum(np.abs(x), 6.0)  # erf(6) == 1.0 in float64
    idx = np.rint(ax * 256).astype(np.intp)
    d = ax - idx * _ERF_STEP
    return np.copysign(_ERF_Y[idx] + d * (_ERF_D1[idx] + d * (_ERF_D2[idx] + d * _ERF_D3[idx])), x)


def gelu(x):
    x64 = x.astype(np.float64)
    return (0.5 * x64 * (1.0 + erf(x64 * 0.7071067811865476))).astype(F32)


def layer_norm(x, weight, bias=None, eps=1e-5):
    centred = x - x.mean(-1, keepdims=True)
    var = np.square(centred).mean(-1, keepdims=True)
    out = centred / np.sqrt(var + F32(eps)) * weight
    return out if bias is None else out + bias


def attention(q, k, v, mask=None):
    """q, k, v: [n, heads, dim] -> [n, heads * dim]; softmax(q.k / sqrt(dim)) v per head."""
    n, heads, dim = q.shape
    q, k, v = (x.transpose(1, 0, 2) for x in (q, k, v))  # [heads, n, dim]
    scores = (q @ k.transpose(0, 2, 1)) * F32(dim ** -0.5)
    if mask is not None:
        scores = scores + mask
    scores -= scores.max(-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(-1, keepdims=True)
    return (probs @ v).transpose(1, 0, 2).reshape(n, heads * dim)


class JuliaModel:
    def __init__(self, gguf):
        m = gguf.metadata
        self.gguf = gguf
        self.n_embd = m['julia1.embedding_length']
        self.n_head = m['julia1.attention.head_count']
        self.n_layer = m['julia1.block_count']
        self.eps = m['julia1.attention.layer_norm_epsilon']
        self.head_eps = m['julia1.head.layer_norm_epsilon']
        self.window = m['julia1.attention.sliding_window'] // 2
        self.pattern = m['julia1.attention.sliding_window_pattern']
        self.rope_theta = m['julia1.rope.freq_base']
        self.rope_dims = m['julia1.rope.dimension_count']
        self.n_head_layer = m['julia1.head.block_count']
        self.head_heads = m['julia1.head.attention.head_count']
        t = gguf.tensor
        self.token_embd_norm = t('token_embd_norm.weight')
        self.layers = []
        for il in range(self.n_layer):
            p = f'blk.{il}.'
            self.layers.append(dict(
                attn_norm=t(p + 'attn_norm.weight') if il != 0 else None,
                qkv=t(p + 'attn_qkv.weight'), wo=t(p + 'attn_output.weight'),
                ffn_norm=t(p + 'ffn_norm.weight'), up=t(p + 'ffn_up.weight'), down=t(p + 'ffn_down.weight')))
        self.output_norm = t('output_norm.weight')
        self.type_embd = t('julia1.type_embd.weight')
        self.head_layers = []
        for j in range(self.n_head_layer):
            p = f'julia1.head.{j}.'
            self.head_layers.append({k: t(p + k) for k in (
                'attn_norm.weight', 'attn_norm.bias', 'attn_qkv.weight', 'attn_qkv.bias',
                'attn_output.weight', 'attn_output.bias', 'ffn_norm.weight', 'ffn_norm.bias',
                'ffn_up.weight', 'ffn_up.bias', 'ffn_down.weight', 'ffn_down.bias')})
        self.scorer = {k: t('julia1.scorer.' + k) for k in (
            'norm.weight', 'norm.bias', 'up.weight', 'up.bias', 'out.weight', 'out.bias')}
        # inv_freq exactly as HF computes it (float32 pow, float32 reciprocal)
        exponent = np.arange(0, self.rope_dims, 2, dtype=np.int64).astype(F32) / F32(self.rope_dims)
        self.inv_freq = F32(1.0) / (F32(self.rope_theta) ** exponent)

    def _rope(self, positions):
        freqs = positions[:, None].astype(F32) * self.inv_freq[None, :]  # [n, dims/2], float32 like HF
        return np.cos(freqs)[:, None, :], np.sin(freqs)[:, None, :]

    @staticmethod
    def _rotate(x, cos, sin):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    def encode(self, ids):
        n = len(ids)
        d, heads = self.n_embd, self.n_head
        x = layer_norm(self.gguf.rows('token_embd.weight', ids), self.token_embd_norm, eps=self.eps)
        positions = np.arange(n)
        cos, sin = self._rope(positions)
        distance = np.abs(positions[:, None] - positions[None, :])
        window_mask = np.where(distance <= self.window, F32(0), F32(-np.inf))[None]  # [1, n, n]
        for il, layer in enumerate(self.layers):
            h = x if il == 0 else layer_norm(x, layer['attn_norm'], eps=self.eps)
            qkv = h @ layer['qkv'].T
            q = self._rotate(qkv[:, :d].reshape(n, heads, -1), cos, sin)
            k = self._rotate(qkv[:, d:2 * d].reshape(n, heads, -1), cos, sin)
            v = qkv[:, 2 * d:].reshape(n, heads, -1)
            mask = None if il % self.pattern == 0 else window_mask
            x = x + attention(q, k, v, mask) @ layer['wo'].T
            h = layer_norm(x, layer['ffn_norm'], eps=self.eps)
            u = h @ layer['up'].T
            half = u.shape[1] // 2
            x = x + (gelu(u[:, :half]) * u[:, half:]) @ layer['down'].T
        return layer_norm(x, self.output_norm, eps=self.eps)

    def head(self, hidden, markers, qtype):
        n, d = hidden.shape
        heads = self.head_heads
        h = hidden + self.type_embd[qtype]
        for layer in self.head_layers:
            a = layer_norm(h, layer['attn_norm.weight'], layer['attn_norm.bias'], eps=self.head_eps)
            qkv = a @ layer['attn_qkv.weight'].T + layer['attn_qkv.bias']
            q, k, v = (qkv[:, i * d:(i + 1) * d].reshape(n, heads, -1) for i in range(3))
            h = h + attention(q, k, v) @ layer['attn_output.weight'].T + layer['attn_output.bias']
            f = layer_norm(h, layer['ffn_norm.weight'], layer['ffn_norm.bias'], eps=self.head_eps)
            f = np.maximum(f @ layer['ffn_up.weight'].T + layer['ffn_up.bias'], F32(0))
            h = h + f @ layer['ffn_down.weight'].T + layer['ffn_down.bias']
        s = self.scorer
        m = layer_norm(h[np.asarray(markers)], s['norm.weight'], s['norm.bias'], eps=self.head_eps)
        m = gelu(m @ s['up.weight'].T + s['up.bias'])
        return (m @ s['out.weight'].T + s['out.bias'])[:, 0]

    def forward(self, ids, markers, qtype):
        """Logits (float32 [len(markers)]) for one encoded request."""
        return self.head(self.encode(ids), markers, qtype)
