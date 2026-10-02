"""Engine mirroring upstream `julia.inference.TransformerEngine`, `julia.typed`, `julia.probabilities`."""
from __future__ import annotations

import math

import numpy as np

from .encoding import RequestEncoder, typed_rows, validate_row
from .model import JuliaModel
from .reader import GGUFModel
from .tokenizer import build_tokenizer


def display_probabilities(probabilities):
    """Upstream julia/probabilities.py, verbatim semantics."""
    values = list(probabilities)
    if not values:
        return values
    winner = max(range(len(values)), key=values.__getitem__)
    if values[winner] > 0.95 and all(value < 0.045 for i, value in enumerate(values) if i != winner):
        return [1.0 if i == winner else 0.0 for i in range(len(values))]
    visible = [value if value >= 0.01 else 0.0 for value in values]
    total = sum(visible)
    return [value / total for value in visible]


def predict_typed(engine, state, questions):
    """Upstream julia/typed.py, verbatim semantics (float64 softmax via math.exp)."""
    rows, metadata = typed_rows(state, questions)
    scores = engine.logits(rows)
    if len(scores) != len(rows):
        raise ValueError('Model returned an incorrect answer count')
    answers = {}
    for (qid, kind, keys), z in zip(metadata, scores):
        if len(z) != len(keys) or not all(math.isfinite(x) for x in z):
            raise ValueError('Invalid model scores')
        maximum = max(z)
        p = [math.exp(x - maximum) for x in z]
        total = sum(p)
        p = [x / total for x in p]
        result = dict(type=kind, probabilities=dict(zip(keys, p)))
        if kind == 'choice':
            result['choice'] = keys[max(range(len(p)), key=p.__getitem__)]
        elif kind == 'score':
            result['score'] = sum(i * x for i, x in enumerate(p))
        else:
            result['noul'] = p[1]
        if kind != 'noul':
            result['max_probability'] = max(p)
        answers[qid] = result
    return dict(answers=answers)


def softmax32(values):
    """float32 softmax, as upstream `torch.tensor(values).softmax(-1)` in the legacy list API."""
    z = np.asarray(values, dtype=np.float32)
    p = np.exp(z - z.max())
    return p / p.sum()


class Engine:
    backend = 'numpy'

    def __init__(self, path, max_length=None, head_length=256, strict_encoding=False, *, tokenizer='auto'):
        self.gguf = GGUFModel(path)  # rejects any GGUF but a full Julia-1 one (julia1 or llama.cpp layout)
        self.metadata = self.gguf.metadata
        limit = self.metadata['julia1.context_length']
        if max_length is None:  # upstream julia/inference.py::context_length: None -> checkpoint limit
            max_length = limit
        if type(max_length) is not int or not 1 <= max_length <= limit:
            raise ValueError(f'max_length must be an integer between 1 and {limit}')
        self.max_length = max_length
        self.head_length = head_length
        self.strict = strict_encoding
        self.tokenizer = build_tokenizer(self.metadata, tokenizer)
        self.encoder = RequestEncoder(self.tokenizer, self.metadata)
        self.model = JuliaModel(self.gguf)

    def encode(self, row):
        """SPEC §5 encoding of one validated request -> {ids, markers, qtype}."""
        return self.encoder.sequence(row, self.max_length, self.head_length, strict=self.strict)

    def tokenize(self, text):
        """SPEC §4 token ids of a text, without CLS/SEP."""
        return self.tokenizer.encode(text)

    def replay(self, ids, markers, qtype):
        """Logits for an already-encoded sequence (the harness' replay mode)."""
        logits = self.model.forward(ids, markers, qtype)
        if not np.isfinite(logits).all():
            raise FloatingPointError('Inference returned nonfinite logits')
        return logits.tolist()

    def logits(self, rows):
        if not rows:
            return []
        for i, row in enumerate(rows):
            validate_row(row, i + 1)
        result = []
        for row in rows:
            encoded = self.encode(row)
            result.append(self.replay(encoded['ids'], encoded['markers'], encoded['qtype']))
        return result

    def predict(self, rows=None, questions=None, *, state=None):
        if questions is not None:
            if rows is not None and state is not None:
                raise ValueError('Pass state either positionally or by keyword, not both')
            return predict_typed(self, state if rows is None else rows, questions)
        if state is not None or rows is None:
            raise ValueError('Provide legacy rows or state with questions')
        result = []
        for values in self.logits(rows):
            probabilities = softmax32(values)
            result.append(dict(index=int(probabilities.argmax()),
                               probabilities=display_probabilities(probabilities.tolist())))
        return result


def load_model(path, max_length=None, head_length=256, strict_encoding=False, *, tokenizer='auto'):
    """Same signature defaults as upstream ``julia.load_model`` (max_length=None -> 8192,
    head_length=256, strict off). The evaluation protocol uses max_length=1024,
    head_length=512, strict_encoding=True; pass them explicitly to reproduce it."""
    if path is None:
        raise ValueError('path is required; pass a Julia-1 GGUF file')
    return Engine(path, max_length, head_length, strict_encoding, tokenizer=tokenizer)
