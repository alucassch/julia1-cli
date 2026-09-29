"""Evaluate a Julia-1 GGUF through a runtime that implements the julia1-cli contract (SPEC §8).

  --runtime cli     drives --cli <julia1-cli>         --runtime numpy  drives `python -m julia1_gguf` (this repository)
  --suite parity|typed   reference rows from data/reference/<suite>.jsonl
  --mode replay          feed reference ids/markers/qtype, compare logits
  --mode e2e             feed the raw request (max_length 1024, head_length 256 parity / 512 typed, strict);
                         compare logits AND the produced ids/markers/qtype with the reference encoding

Reports max/mean abs logit diff, argmax agreement, accuracy vs gold (typed, per type), wall time per
request (from the runtime's stderr timing line when present, else subprocess wall / N) and, for
parity, max abs diff vs the upstream PyTorch logits. Prints a one-line summary, writes the full report to
--output when given, and exits 1 when a guard given on the command line fails (--max-abs, --min-argmax,
--accuracy, 0 encoding mismatches in e2e mode).

  python tools/eval_gguf.py --runtime cli --cli build/julia1-cli --model Julia-1-F32.gguf --suite parity --mode e2e \
      --device cpu --min-argmax 100 --max-abs 2e-3
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEAD_LENGTH = dict(parity=256, typed=512)
MAX_LENGTH = 1024


def read_jsonl(path):
    with open(path, encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + '\n')


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--runtime', choices=['numpy', 'cli'], required=True)
    ap.add_argument('--cli', type=Path, help='path to julia1-cli (runtime=cli)')
    ap.add_argument('--python', default=sys.executable, help='interpreter for the numpy runtime')
    ap.add_argument('--suite', choices=['parity', 'typed'], required=True)
    ap.add_argument('--mode', choices=['replay', 'e2e'], required=True)
    ap.add_argument('--output', type=Path, help='write the JSON report here')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--device', default=None, help='forwarded as --device')
    ap.add_argument('--threads', type=int, default=None, help='forwarded as --threads')
    ap.add_argument('--tokenizer', choices=['auto', 'hf', 'pure'], default=None, help='numpy runtime only')
    ap.add_argument('--fast', action='store_true', help='forwarded as --fast (runtime=cli)')
    ap.add_argument('--batch', type=int, default=None, help='forwarded as --batch (runtime=cli)')
    ap.add_argument('--max-abs', type=float, help='fail if the max abs logit diff exceeds this')
    ap.add_argument('--min-argmax', type=int, help='fail if the argmax agreement is below this')
    ap.add_argument('--accuracy', type=int, help='expected accuracy (typed, with --accuracy-tol)')
    ap.add_argument('--accuracy-tol', type=int, default=1)
    args = ap.parse_args()

    reference = read_jsonl(ROOT / 'data' / 'reference' / f'{args.suite}.jsonl')
    if args.limit:
        reference = reference[:args.limit]

    if args.runtime == 'cli':
        if not args.cli:
            ap.error('--cli is required for runtime=cli')
        command = [str(args.cli.resolve()) if args.cli.exists() else str(args.cli)]  # the runtime runs in ROOT
        if args.fast:
            command += ['--fast']
        if args.batch:
            command += ['--batch', str(args.batch)]
    else:
        command = [args.python, '-m', 'julia1_gguf']
        if args.tokenizer:
            command += ['--tokenizer', args.tokenizer]
    command += ['--model', str(args.model.resolve())]
    if args.device:
        command += ['--device', args.device]
    if args.threads:
        command += ['--threads', str(args.threads)]

    with tempfile.TemporaryDirectory(prefix='julia1-eval-') as tmp:
        inp = Path(tmp) / 'input.jsonl'
        outp = Path(tmp) / 'output.jsonl'
        if args.mode == 'replay':
            write_jsonl(inp, [dict(id=r['id'], ids=r['ids'], markers=r['markers'], qtype=r['qtype']) for r in reference])
            command += ['replay', '--input', str(inp), '--output', str(outp)]
        else:
            write_jsonl(inp, [r['request'] for r in reference])
            command += ['decide', '--input', str(inp), '--output', str(outp), '--max-length', str(MAX_LENGTH),
                        '--head-length', str(HEAD_LENGTH[args.suite]), '--strict']
        env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', ''))
        started = time.monotonic()
        proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        wall = time.monotonic() - started
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            raise SystemExit(f'runtime failed with exit code {proc.returncode}: {" ".join(command)}')
        outputs = read_jsonl(outp)
    if len(outputs) != len(reference):
        raise SystemExit(f'runtime wrote {len(outputs)} rows, expected {len(reference)}')
    timing = None
    for line in reversed(proc.stderr.splitlines()):
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict) and 'elapsed_s' in candidate:
            timing = candidate
            break

    max_abs = mean_abs = 0.0
    agree = 0
    max_abs_upstream = 0.0
    encoding_mismatches = 0
    correct = {}
    worst = []
    count = 0
    for ref, out in zip(reference, outputs):
        logits = out['logits']
        if len(logits) != len(ref['logits']):
            raise SystemExit(f'{ref["id"]}: runtime returned {len(logits)} logits, expected {len(ref["logits"])}')
        diffs = [abs(a - b) for a, b in zip(logits, ref['logits'])]
        row_max = max(diffs)
        max_abs = max(max_abs, row_max)
        mean_abs += sum(diffs)
        count += len(diffs)
        agree += int(argmax(logits) == argmax(ref['logits']))
        worst.append((row_max, ref['id']))
        if 'upstream_logits' in ref:
            max_abs_upstream = max(max_abs_upstream, max(abs(a - b) for a, b in zip(logits, ref['upstream_logits'])))
        if ref.get('gold') is not None:
            bucket = correct.setdefault(ref['type'], [0, 0])
            bucket[0] += int(ref['keys'][argmax(logits)] == ref['gold'])
            bucket[1] += 1
        if args.mode == 'e2e':
            if out['ids'] != ref['ids'] or out['markers'] != ref['markers'] or out['qtype'] != ref['qtype']:
                encoding_mismatches += 1
    worst.sort(reverse=True)
    n = len(reference)
    report = dict(
        model=str(args.model), runtime=args.runtime, cli=str(args.cli) if args.cli else None, suite=args.suite,
        mode=args.mode, rows=n, max_abs_logit_diff=max_abs, mean_abs_logit_diff=mean_abs / max(1, count),
        argmax_agreement=agree, argmax_agreement_fraction=agree / n,
        accuracy=sum(v[0] for v in correct.values()) if correct else None,
        by_type={k: dict(correct=v[0], count=v[1], accuracy=v[0] / v[1]) for k, v in correct.items()},
        max_abs_vs_upstream_logits=max_abs_upstream if args.suite == 'parity' else None,
        encoding_mismatches=encoding_mismatches if args.mode == 'e2e' else None,
        wall_s=round(wall, 3), ms_per_request_incl_load=round(1000 * wall / n, 3),
        runtime_timing=timing, ms_per_request=timing['ms_per_request'] if timing else round(1000 * wall / n, 3),
        worst_rows=[dict(id=i, max_abs=d) for d, i in worst[:10]], command=command)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    summary = {k: report[k] for k in ('suite', 'mode', 'runtime', 'rows', 'max_abs_logit_diff', 'mean_abs_logit_diff',
                                      'argmax_agreement', 'accuracy', 'max_abs_vs_upstream_logits',
                                      'encoding_mismatches', 'ms_per_request')}
    summary['by_type'] = {k: v['correct'] for k, v in report['by_type'].items()}
    print(json.dumps(summary))

    failed = []
    if args.max_abs is not None and max_abs > args.max_abs:
        failed.append(f'max abs {max_abs:.3e} > {args.max_abs:.1e}')
    if args.min_argmax is not None and agree < args.min_argmax:
        failed.append(f'argmax agreement {agree} < {args.min_argmax}')
    if args.accuracy is not None and abs((report['accuracy'] or 0) - args.accuracy) > args.accuracy_tol:
        failed.append(f'accuracy {report["accuracy"]} not within {args.accuracy} +- {args.accuracy_tol}')
    if args.mode == 'e2e' and encoding_mismatches:
        failed.append(f'{encoding_mismatches} encoding mismatches')
    for f in failed:
        print('FAIL:', f)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
