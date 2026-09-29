#!/usr/bin/env python
"""Compare julia1-cli output (replay or decide) against a reference JSONL (SPEC §8).

Reports max/mean abs logit delta, argmax agreement, accuracy vs gold (keys[argmax] == gold), and for
`decide` output also exact equality of ids/markers/qtype with the reference encoding. Optionally writes
a JSON report. Exit code 1 if any threshold given on the command line is violated.
"""
import argparse
import json
import sys
from pathlib import Path


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def load(path):
    return [json.loads(l) for l in Path(path).open(encoding='utf-8') if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', required=True)
    ap.add_argument('--output', required=True, help='julia1-cli output JSONL (same row order as the reference)')
    ap.add_argument('--report', help='write a JSON report here')
    ap.add_argument('--label', default='')
    ap.add_argument('--max-abs', type=float, help='fail if max abs delta exceeds this')
    ap.add_argument('--min-argmax', type=int, help='fail if argmax agreement is below this')
    ap.add_argument('--accuracy', type=int, help='expected accuracy (with --accuracy-tol)')
    ap.add_argument('--accuracy-tol', type=int, default=1)
    args = ap.parse_args()

    ref = load(args.reference)
    out = load(args.output)
    if len(ref) != len(out):
        print(f'row count differs: reference {len(ref)} vs output {len(out)}')
        sys.exit(1)

    ref_correct = 0
    max_abs = 0.0
    sum_abs = 0.0
    n_logits = 0
    argmax_agree = 0
    correct = 0
    max_abs_upstream = 0.0
    encoding_mismatches = 0
    have_upstream = False
    have_encoding = False
    worst = None
    for r, o in zip(ref, out):
        if 'id' in o and o['id'] != r['id']:
            print(f"id mismatch: {o['id']} vs {r['id']}")
            sys.exit(1)
        a, b = r['logits'], o['logits']
        if len(a) != len(b):
            print(f"logit count differs for {r['id']}: {len(a)} vs {len(b)}")
            sys.exit(1)
        deltas = [abs(x - y) for x, y in zip(a, b)]
        row_max = max(deltas)
        if row_max > max_abs:
            max_abs, worst = row_max, r['id']
        sum_abs += sum(deltas)
        n_logits += len(deltas)
        ia, ib = argmax(a), argmax(b)
        argmax_agree += ia == ib
        if r.get('gold') is not None:
            ref_correct += r['keys'][ia] == r['gold']
            correct += r['keys'][ib] == r['gold']
        if 'upstream_logits' in r:
            have_upstream = True
            max_abs_upstream = max(max_abs_upstream, max(abs(x - y) for x, y in zip(r['upstream_logits'], b)))
        if 'ids' in o:
            have_encoding = True
            if o['ids'] != r['ids'] or o['markers'] != r['markers'] or o.get('qtype') != r['qtype']:
                encoding_mismatches += 1

    n = len(ref)
    has_gold = any(r.get('gold') is not None for r in ref)
    by_type = {}
    if has_gold:
        for r, o in zip(ref, out):
            t = r['type']
            d = by_type.setdefault(t, {'count': 0, 'correct': 0})
            d['count'] += 1
            d['correct'] += r['keys'][argmax(o['logits'])] == r['gold']
    report = {
        'label': args.label, 'reference': str(args.reference), 'output': str(args.output), 'rows': n,
        'max_abs_logit_delta': max_abs, 'mean_abs_logit_delta': sum_abs / max(1, n_logits), 'worst_row': worst,
        'argmax_agreement': argmax_agree,
        'accuracy': correct if has_gold else None, 'reference_accuracy': ref_correct if has_gold else None,
        'by_type': by_type or None,
        'max_abs_vs_upstream_logits': max_abs_upstream if have_upstream else None,
        'encoding_mismatches': encoding_mismatches if have_encoding else None,
    }
    print(json.dumps(report, indent=2))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2) + '\n')

    failed = []
    if args.max_abs is not None and max_abs > args.max_abs:
        failed.append(f'max abs {max_abs:.3e} > {args.max_abs:.1e}')
    if args.min_argmax is not None and argmax_agree < args.min_argmax:
        failed.append(f'argmax agreement {argmax_agree} < {args.min_argmax}')
    if args.accuracy is not None and abs(correct - args.accuracy) > args.accuracy_tol:
        failed.append(f'accuracy {correct} not within {args.accuracy} +- {args.accuracy_tol}')
    if have_encoding and encoding_mismatches:
        failed.append(f'{encoding_mismatches} encoding mismatches')
    for f in failed:
        print('FAIL:', f)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
