#!/usr/bin/env python3
"""Parity of a typed-question HTTP server against the upstream PyTorch golden reference: llama.cpp's /v1/systemone
(PR #29818) or julia1-cli serve's /v1/predict.

Each of the 2000 rows of data/reference/typed.jsonl is sent as one question (state + one named question), so that
usage.input_tokens is the token count of that row's prompt. The golden rows fit both encodings (question + options
<= 75 tokens, sequence <= 607 tokens), so llama.cpp's defaults (head 256, context 8192) and the reference protocol
(head 512, max_length 1024) encode them the same way.

Compared per row: decision (argmax), token count vs the upstream ids, and the logits recovered from the returned
probabilities (log p, centred; temperature is 1 for Julia-1, the model has none) vs the centred golden logits.

  llama-server -m Julia-1-F32.gguf -b 1024 -ub 1024 --port 8080
  python3 tools/systemone_parity.py --url http://127.0.0.1:8080 --state-as-string --out f32-metal.jsonl
"""
import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / 'data' / 'reference' / 'typed.jsonl'


def question(row):
    req = row['request']
    kind, options, keys = row['type'], req['options'], row['keys']
    q = dict(type=kind, instructions=req['question'])
    if kind == 'choice':
        q['criteria'] = dict(zip(keys, options))
    elif kind == 'score':
        q['criteria'] = list(options)
    elif options != ['false', 'true']:
        q['criteria'] = dict(zip(['false', 'true'], options))
    return q


def post(url, body, endpoint):
    data = json.dumps(body, ensure_ascii=False).encode()
    request = urllib.request.Request(url + endpoint, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def centred(xs):
    mean = sum(xs) / len(xs)
    return [x - mean for x in xs]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', default='http://127.0.0.1:18191')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--endpoint', default='/v1/systemone',
                    help='/v1/systemone for llama-server, /v1/predict for julia1-cli serve (same request and answer shape)')
    ap.add_argument('--state-as-string', action='store_true',
                    help='send the state pre-serialized with json.dumps, as upstream encodes it (llama.cpp renders '
                         'an integral float such as 532.0 as 532)')
    args = ap.parse_args()

    rows = [json.loads(line) for line in GOLDEN.open(encoding='utf-8')][:args.limit]
    records, latencies = [], []
    for row in rows:
        state = row['request']['state']
        if args.state_as_string and not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False)
        body = dict(state=state, questions={'q': question(row)})
        started = time.perf_counter()
        try:
            answer = post(args.url, body, args.endpoint)
        except urllib.error.HTTPError as e:
            records.append(dict(id=row['id'], type=row['type'], error=f'{e.code}: {e.read().decode()[:200]}',
                                ref_tokens=len(row['ids'])))
            continue
        latencies.append((time.perf_counter() - started) * 1e3)
        a = answer['answers']['q']
        gold = row['logits']
        if row['type'] == 'noul':
            p = min(max(a['noul'], 1e-300), 1 - 1e-16)
            # two options: only the difference of the logits is observable
            got_diff = math.log(p) - math.log1p(-p)
            err = abs(got_diff - (gold[1] - gold[0]))
            pred = 'true' if p > 0.5 else 'false'
        else:
            probs = [a['probabilities'][k] for k in row['keys']]
            got = centred([math.log(max(x, 1e-300)) for x in probs])
            err = max(abs(x - y) for x, y in zip(got, centred(gold)))
            pred = row['keys'][max(range(len(probs)), key=probs.__getitem__)]
        ref_pred = row['keys'][max(range(len(gold)), key=gold.__getitem__)]
        records.append(dict(id=row['id'], type=row['type'], pred=pred, ref_pred=ref_pred, gold=row['gold'],
                            agree=pred == ref_pred, correct=pred == row['gold'], err=err,
                            tokens=answer.get('usage', {}).get('input_tokens'), ref_tokens=len(row['ids'])))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    errors = [r for r in records if 'error' in r]
    records = [r for r in records if 'error' not in r]
    by_type = {}
    for r in records:
        b = by_type.setdefault(r['type'], dict(n=0, agree=0, correct=0))
        b['n'] += 1
        b['agree'] += r['agree']
        b['correct'] += r['correct']
    errs = sorted(r['err'] for r in records)
    summary = dict(
        url=args.url, endpoint=args.endpoint, state_as_string=args.state_as_string, rows=len(records) + len(errors), errors=len(errors),
        first_error=errors[0]['error'] if errors else None,
        agree=sum(r['agree'] for r in records), correct=sum(r['correct'] for r in records), by_type=by_type,
        token_count_mismatches=sum(r['tokens'] is not None and r['tokens'] != r['ref_tokens'] for r in records),
        max_abs_logit_err=errs[-1], p99_logit_err=errs[int(0.99 * (len(errs) - 1))],
        median_logit_err=statistics.median(errs),
        rows_err_gt_1e_2=sum(e > 1e-2 for e in errs), rows_err_gt_1e_3=sum(e > 1e-3 for e in errs),
        latency_ms_median=statistics.median(latencies), latency_ms_p95=sorted(latencies)[int(0.95 * (len(latencies) - 1))],
    )
    args.out.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
