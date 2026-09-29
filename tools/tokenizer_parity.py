"""Tokenizer parity: upstream HF `tokenizers` (tokenizer.json, add_special_tokens=False) vs a candidate (SPEC §4).

  --candidate hf-from-gguf   julia1_gguf.tokenizer.HfTokenizer rebuilt from GGUF metadata
  --candidate pure           julia1_gguf.tokenizer.PureTokenizer (dependency-free)
  --candidate cpp-jsonl      a file produced by `julia1-cli tokenize --input <corpus> --output <file>` (--cpp-file)

--tokenizer-json is upstream's tokenizer/tokenizer.json (SupersonicLabs/Julia-1). Prints the first mismatches and a
one-line summary, writes the report to --output when given, and exits 1 on any mismatch.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_jsonl(path):
    with open(path, encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--corpus', type=Path, default=ROOT / 'data' / 'tokenizer-corpus.jsonl')
    ap.add_argument('--tokenizer-json', type=Path, required=True, help="upstream's tokenizer/tokenizer.json")
    ap.add_argument('--model', type=Path, help='Julia-1 GGUF providing the metadata (hf-from-gguf, pure)')
    ap.add_argument('--candidate', choices=['hf-from-gguf', 'pure', 'cpp-jsonl'], required=True)
    ap.add_argument('--cpp-file', type=Path, help='output of julia1-cli tokenize over --corpus (cpp-jsonl)')
    ap.add_argument('--output', type=Path, help='write the JSON report here')
    args = ap.parse_args()
    if args.candidate != 'cpp-jsonl' and not args.model:
        ap.error('--model is required for hf-from-gguf and pure')

    from tokenizers import Tokenizer
    reference = Tokenizer.from_file(str(args.tokenizer_json))
    corpus = read_jsonl(args.corpus)
    texts = [row['text'] for row in corpus]
    expected = [reference.encode(text, add_special_tokens=False).ids for text in texts]

    started = time.monotonic()
    if args.candidate == 'cpp-jsonl':
        if not args.cpp_file:
            ap.error('--cpp-file is required for cpp-jsonl')
        rows = read_jsonl(args.cpp_file)
        if len(rows) != len(texts):
            raise SystemExit(f'{args.cpp_file}: {len(rows)} lines, corpus has {len(texts)}')
        got = [row['ids'] for row in rows]
        load_s = 0.0
    else:
        from julia1_gguf.reader import GGUFModel
        from julia1_gguf.tokenizer import HfTokenizer, PureTokenizer
        meta = GGUFModel(args.model).metadata
        candidate = HfTokenizer(meta) if args.candidate == 'hf-from-gguf' else PureTokenizer(meta)
        load_s = time.monotonic() - started
        started = time.monotonic()
        got = [candidate.encode(text) for text in texts]
    encode_s = time.monotonic() - started

    diffs = [dict(line=i + 1, section=corpus[i].get('section'), text=texts[i], expected=e, got=g)
             for i, (e, g) in enumerate(zip(expected, got)) if e != g]
    by_section = {}
    for i, row in enumerate(corpus):
        bucket = by_section.setdefault(row.get('section', '?'), [0, 0])
        bucket[1] += 1
        bucket[0] += int(expected[i] != got[i])
    report = dict(candidate=args.candidate, corpus=str(args.corpus), total=len(texts), mismatches=len(diffs),
                  tokens=sum(map(len, expected)), by_section={k: dict(mismatches=v[0], total=v[1]) for k, v in by_section.items()},
                  load_s=round(load_s, 3), encode_s=round(encode_s, 3), first_diffs=diffs[:20])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    for diff in diffs[:20]:
        print(json.dumps(diff, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != 'first_diffs'}))
    return 1 if diffs else 0


if __name__ == '__main__':
    sys.exit(main())
