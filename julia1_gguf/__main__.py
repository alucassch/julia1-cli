"""CLI with the same contract as julia1-cli:

  python -m julia1_gguf --model <gguf> [--device auto|cpu|metal] [--threads N] [--tokenizer auto|hf|pure] \\
      replay   --input <ref.jsonl> --output <out.jsonl>
      decide   --input <requests.jsonl> --output <out.jsonl> [--max-length 1024] [--head-length 512] [--strict]
      tokenize --input <texts.jsonl> --output <out.jsonl>

Prints one JSON line with load/elapsed timing to stderr. Non-zero exit and a
message on stderr on any error. --device is accepted for contract parity and ignored.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def parse_args(argv):
    ap = argparse.ArgumentParser(prog='julia1_gguf', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'metal'])
    ap.add_argument('--threads', type=int, default=None)
    ap.add_argument('--tokenizer', default='auto', choices=['auto', 'hf', 'pure'])
    sub = ap.add_subparsers(dest='command', required=True)
    for name in ('replay', 'decide', 'tokenize'):
        p = sub.add_parser(name)
        p.add_argument('--input', required=True)
        p.add_argument('--output', required=True)
        if name == 'decide':
            p.add_argument('--max-length', type=int, default=8192, help='upstream default (checkpoint limit); evaluation protocol: 1024')
            p.add_argument('--head-length', type=int, default=256, help='upstream default; evaluation protocol: 512')
            p.add_argument('--strict', action='store_true')
    return ap.parse_args(argv)


def read_jsonl(path):
    with open(path, encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def run(args):
    from .encoding import validate_row
    from .engine import load_model, softmax32

    started = time.monotonic()
    decide = args.command == 'decide'
    engine = load_model(args.model, max_length=args.max_length if decide else 8192,
                        head_length=args.head_length if decide else 512,
                        strict_encoding=args.strict if decide else True, tokenizer=args.tokenizer)
    loaded = time.monotonic()
    rows = read_jsonl(args.input)
    with open(args.output, 'w', encoding='utf-8') as out:
        for index, row in enumerate(rows):
            if args.command == 'replay':
                record = dict(id=row.get('id', index), logits=engine.replay(row['ids'], row['markers'], row['qtype']))
            elif args.command == 'decide':
                validate_row(row, index + 1)
                encoded = engine.encode(row)
                logits = engine.replay(encoded['ids'], encoded['markers'], encoded['qtype'])
                probabilities = softmax32(logits)
                record = dict(logits=logits, index=int(probabilities.argmax()), probabilities=probabilities.tolist(),
                              ids=encoded['ids'], markers=encoded['markers'], qtype=encoded['qtype'])
                if 'id' in row:
                    record['id'] = row['id']
            else:
                record = dict(ids=engine.tokenizer.encode(row['text']))
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
    elapsed = time.monotonic() - loaded
    print(json.dumps(dict(command=args.command, requests=len(rows), loaded_s=round(loaded - started, 3),
                          elapsed_s=round(elapsed, 3), ms_per_request=round(1000 * elapsed / max(1, len(rows)), 3))),
          file=sys.stderr, flush=True)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.threads:  # must precede the numpy import
        for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            os.environ[var] = str(args.threads)
    try:
        run(args)
    except Exception as error:  # noqa: BLE001 - contract: message on stderr, non-zero exit
        print(f'julia1_gguf: {type(error).__name__}: {error}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
