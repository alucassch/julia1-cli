"""SPEC §5: request validation and marker-sequence encoding (port of upstream `julia/data.py`, and the request half of
`julia/typed.py`; no torch, no numpy)."""
from __future__ import annotations

import json
import math

QTYPES = {'choice': 0, 'score': 1, 'noul': 2}


def validate_row(row, line):
    prefix = f'JSONL line {line}: '
    if not isinstance(row, dict):
        raise ValueError(prefix + 'request must be a JSON object')
    if not isinstance(row.get('state'), (str, dict, list)) or not isinstance(row.get('question'), str):
        raise ValueError(prefix + 'state must be text/JSON and question must be text')
    options = row.get('options')
    if not isinstance(options, list) or not 2 <= len(options) <= 20 or not all(isinstance(x, str) and x for x in options):
        raise ValueError(prefix + 'options must contain 2–20 nonempty rendered descriptions')
    if row.get('type', 'choice') not in QTYPES:
        raise ValueError(prefix + 'type must be choice, score, or noul')
    if row.get('type') == 'noul' and len(options) != 2:
        raise ValueError(prefix + 'noul options must be ordered [false, true]')
    if 'target' in row and (type(row['target']) is not int or not 0 <= row['target'] < len(options)):
        raise ValueError(prefix + 'target must index the supplied option list')
    teacher = row.get('teacher_logits')
    if teacher is not None and (
        not isinstance(teacher, list)
        or len(teacher) != len(options)
        or not all(type(x) in (int, float) and math.isfinite(x) for x in teacher)
    ):
        raise ValueError(prefix + 'teacher logits must be finite and match option count/order')


def typed_rows(state, questions):
    """Upstream julia/typed.py's validation of named questions: one validated request per question, and its
    (id, type, keys)."""
    if not isinstance(questions, dict) or not questions:
        raise ValueError('questions must be a nonempty mapping')
    rows = []
    metadata = []
    for qid, q in questions.items():
        if not isinstance(qid, str) or not qid or not isinstance(q, dict):
            raise ValueError('Questions require nonempty string IDs and question objects')
        kind = q.get('type')
        criteria = q.get('criteria')
        if kind == 'choice':
            if not isinstance(criteria, dict) or any(not isinstance(k, str) or not k for k in criteria):
                raise ValueError('Choice criteria must map nonempty IDs to descriptions')
            keys = list(criteria)
            labels = list(criteria.values())
        elif kind == 'score':
            if not isinstance(criteria, list):
                raise ValueError('Score requires an ordered rubric')
            labels = criteria
            keys = [str(i) for i in range(len(labels))]
        elif kind == 'noul':
            keys = ['false', 'true']
            if criteria is None:
                labels = list(keys)
            else:
                if not isinstance(criteria, dict) or set(criteria) != set(keys):
                    raise ValueError('Noul criteria must map false and true to descriptions')
                labels = [criteria[key] for key in keys]
        else:
            raise ValueError('Unsupported question type')
        row = dict(state=state, question=q.get('instructions'), type=kind, options=labels)
        validate_row(row, len(rows) + 1)
        rows.append(row)
        metadata.append((qid, kind, keys))
    return rows, metadata


class RequestEncoder:
    def __init__(self, tokenizer, meta):
        self.encode = tokenizer.encode
        self.mask_token = meta['julia1.encoding.marker_token_text']
        self.mask_id = meta['julia1.encoding.marker_token_id']
        self.cls_id = meta['julia1.encoding.cls_token_id']
        self.sep_id = meta['julia1.encoding.sep_token_id']
        self.template = meta['julia1.encoding.head_template']
        self.option_prefix = meta['julia1.encoding.option_prefix']
        self.option_limit = meta['julia1.encoding.option_token_limit']

    def sequence(self, row, max_length=8192, head_length=256, *, strict=False):
        if head_length + 4 >= max_length:
            raise ValueError('max_length must leave room beyond the question head')
        # allow_nan=False: upstream FastEngine rejects NaN/inf floats in a JSON state (its encoding-cache key)
        state = row['state'] if isinstance(row['state'], str) else json.dumps(row['state'], ensure_ascii=False, allow_nan=False)
        if strict and any(self.mask_token in text for text in [state, row['question'], *row['options']]):
            raise ValueError('Reserved model marker in request')
        clean = lambda text: text.replace(self.mask_token, ' ')  # noqa: E731
        encode = self.encode
        qtype = row.get('type', 'choice')
        head = encode(self.template.format(type=qtype, question=clean(row['question'])))
        option_ids = [encode(self.option_prefix + clean(x)) for x in row['options']]
        if strict and any(len(x) > self.option_limit for x in option_ids):
            raise ValueError(f'Option exceeds {self.option_limit}-token model contract')
        options = [[self.mask_id] + x[:self.option_limit] for x in option_ids]
        budget = head_length - sum(map(len, options))
        if budget < 16:
            per_option = max(4, (head_length - 16) // len(options))
            options = [x[:per_option] for x in options]
            budget = head_length - sum(map(len, options))
        if strict and (len(head) > budget or any(len(x) != len(y) + 1 for x, y in zip(options, option_ids))):
            raise ValueError('Question/options exceed lossless head budget')
        ids = [self.cls_id] + head[:max(8, budget)] + [self.sep_id]
        markers = []
        for option in options:
            markers.append(len(ids))
            ids.extend(option)
        ids.append(self.sep_id)
        state_ids = encode(clean(state))
        room = max_length - len(ids) - 1
        if room < 1:
            raise ValueError('Question/options exceed sequence budget; shorten descriptions')
        if strict and len(state_ids) > room:
            raise ValueError('Game state exceeds lossless context budget')
        return dict(ids=ids + state_ids[:room] + [self.sep_id], markers=markers, qtype=QTYPES[qtype])
