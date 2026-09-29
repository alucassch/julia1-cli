#!/usr/bin/env python
"""Compare `julia1-cli tokenize` against HF tokenizers (SPEC §4 acceptance: 0 mismatches).

--tokenizer-json is upstream's tokenizer/tokenizer.json (SupersonicLabs/Julia-1). Builds the corpus
(data/tokenizer-corpus.jsonl, one {"text": ...} per line) when it is missing or empty: every distinct
state/question/option string of the reference requests, plus " "+option and "{type} question: {question}",
plus deterministic fuzz strings. Exits 1 on any mismatch.
"""
import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def reference_strings():
    out = []
    for name in ('parity.jsonl', 'typed.jsonl'):
        for line in (ROOT / 'data' / 'reference' / name).open(encoding='utf-8'):
            req = json.loads(line)['request']
            state = req['state'] if isinstance(req['state'], str) else json.dumps(req['state'], ensure_ascii=False)
            out.append(state)
            out.append(req['question'])
            out.append(f"{req.get('type', 'choice')} question: {req['question']}")
            for o in req['options']:
                out.append(o)
                out.append(' ' + o)
    return out


def fuzz_strings(tokenizer_path, count=2400, seed=20260927):
    rng = random.Random(seed)
    tokenizer_json = json.loads(Path(tokenizer_path).read_text(encoding='utf-8'))
    added = [a['content'] for a in tokenizer_json['added_tokens']]
    tags = [a for a in added if a.startswith('<') and a[1:2].isalpha() and 'unused' not in a and a not in ('<mask>',)]
    unused = [a for a in added if 'unused' in a]
    words = ['hello', 'world', 'Julia', 'decision', 'model', 'state', 'option', 'the', 'a', 'of', 'and', 'ação',
             'coração', 'naïve', 'straße', 'Ünïcödé', 'año', 'ça', 'œuvre', 'Ωmega', 'Ελληνικά', 'русский', 'עברית',
             'مرحبا', 'नमस्ते', 'বাংলা', 'ไทย', '日本語', '中文', '한국어', '😀', '🚀🌍', '👨‍👩‍👧', '🇧🇷', '∑∫√', '→←↑',
             '€$£¥', '1234567', '3.14159', '-42', '0x1F', 'a1b2c3', 'CamelCase', 'snake_case', 'kebab-case',
             'http://x.y/z?q=1&r=2', 'user@example.com', '{"k": [1, 2.5, null, true]}', "it's", '"quoted"',
             '(parens)', '[brackets]', '<angle>', '#hash', '@at', '%pct', '^caret', '~tilde', '`tick`', '|pipe|',
             '\\backslash', '/slash/', ';;', '::', '..', ',,', '!!', '??', '­', '​', '﻿', ' ',
             '　', ' ', '\U0001F9E0', '\U00010348', '\U0002A6A5', '', '�', '́',
             'é', 'ﬁ', 'ǅ', 'ß', 'İ', 'ı']
    ws = [' ', '  ', '   ', '\t', '\t\t', '\n', '\n\n', '\r\n', '\r', ' \n ', '\n\t', '\t \t', '▁', '▁▁',
          ' ' * 35, '\n' * 33, '\t' * 33, '▁' * 33, '\x0b', '\x0c', '\x1f', '\x7f', '\x85']
    out = []
    for _ in range(count):
        parts = []
        for _ in range(rng.randint(1, 12)):
            r = rng.random()
            if r < 0.45:
                parts.append(rng.choice(words))
            elif r < 0.75:
                parts.append(rng.choice(ws))
            elif r < 0.85:
                parts.append(rng.choice(tags))
            elif r < 0.9:
                parts.append(rng.choice(unused))
            elif r < 0.95:
                parts.append(chr(rng.choice([rng.randint(0x20, 0x7e), rng.randint(0x80, 0x7ff), rng.randint(0x800, 0xd7ff),
                                             rng.randint(0xe000, 0xffff), rng.randint(0x10000, 0x10ffff)])))
            else:
                parts.append(''.join(chr(rng.randint(0x4e00, 0x9fff)) for _ in range(rng.randint(1, 40))))
        out.append(''.join(parts))
    # long space-free strings and pure whitespace runs
    out.append(''.join(chr(rng.randint(0x4e00, 0x9fff)) for _ in range(3000)))
    out.append('a' * 3000)
    out.append(' ' * 200)
    out.append('\n' * 100)
    out.append('\t' * 100)
    out.append('')
    return out


def build_corpus(path: Path, tokenizer_path):
    texts = []
    seen = set()
    for t in reference_strings() + fuzz_strings(tokenizer_path):
        if t not in seen:
            seen.add(t)
            texts.append(t)
    with path.open('w', encoding='utf-8') as f:
        for t in texts:
            f.write(json.dumps({'text': t}, ensure_ascii=False) + '\n')
    return len(texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cli', default=str(ROOT / 'build' / 'julia1-cli'))
    ap.add_argument('--model', required=True, help='a full Julia-1 GGUF')
    ap.add_argument('--tokenizer-json', required=True, help="upstream's tokenizer/tokenizer.json")
    ap.add_argument('--corpus', default=str(ROOT / 'data' / 'tokenizer-corpus.jsonl'))
    ap.add_argument('--report', help='write the JSON report here')
    args = ap.parse_args()

    corpus = Path(args.corpus)
    if not corpus.exists() or corpus.stat().st_size == 0:
        try:
            n = build_corpus(corpus, args.tokenizer_json)
        except OSError:
            corpus = Path(tempfile.gettempdir()) / 'tokenizer-corpus.jsonl'
            n = build_corpus(corpus, args.tokenizer_json)
        print(f'built corpus {corpus} with {n} strings')
    texts = [json.loads(l)['text'] for l in corpus.open(encoding='utf-8') if l.strip()]

    out_path = Path(tempfile.gettempdir()) / 'tokenizer-cpp.out.jsonl'
    subprocess.run([args.cli, '--model', args.model, '--device', 'cpu', 'tokenize', '--input', str(corpus), '--output', str(out_path)],
                   check=True)
    cpp = [json.loads(l)['ids'] for l in out_path.open(encoding='utf-8')]
    assert len(cpp) == len(texts), (len(cpp), len(texts))

    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(args.tokenizer_json))
    mismatches = []
    for text, ids in zip(texts, cpp):
        ref = tk.encode(text, add_special_tokens=False).ids
        if ref != ids:
            mismatches.append({'text': text, 'hf': ref, 'cpp': ids})
    for m in mismatches[:10]:
        print('MISMATCH', repr(m['text'])[:200], '\n  hf ', m['hf'][:40], '\n  cpp', m['cpp'][:40])
    report = {'corpus': str(corpus), 'corpus_size': len(texts), 'mismatches': len(mismatches),
              'examples': mismatches[:20]}
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'corpus {len(texts)} strings, mismatches {len(mismatches)}')
    sys.exit(1 if mismatches else 0)


if __name__ == '__main__':
    main()
