#!/bin/bash
# Release packaging for macOS arm64 (the only platform packaged here; Linux builds from source).
#  1. version check: CMakeLists.txt project() = pyproject.toml = julia1_gguf.__version__, and a CHANGELOG.md entry
#  2. clean Release build in build-release/ (configure downloads llama.cpp v0.5.0, SHA-256 pinned, and patches a copy of
#     its ggml); MACOSX_DEPLOYMENT_TARGET (default 14.0) is the minimum macOS of the binaries
#  3. regression smoke on the 100 parity cases with the F32 model: decide and replay on Metal (exact kernels) and decide
#     on the CPU must pass the exact guard (100/100 argmax, max abs <= 1e-3 Metal / 2e-3 CPU, 0 encoding mismatches)
#  4. cmake --install into a staging prefix -> dist/julia1-cli-<version>-macos-arm64.tar.gz (+ .sha256)
#  5. the Python wheel, uv build --wheel -> dist/julia1_gguf-<version>-py3-none-macosx_<target>_arm64.whl (+ .sha256)
#  6. checks outside the build tree, in build-release/relcheck (deleted at exit; KEEP_RELCHECK=1 keeps it): the
#     tarball's file list, otool -L (system libraries only), minos and no build-tree paths in the binaries, --version,
#     decide on parity (exact guard), serve answering /v1/predict, the quickstart's C API example, and the wheel installed
#     in a fresh venv
# usage: tools/package_release.sh --model <Julia-1-F32.gguf> [--no-wheel]
# env: MACOSX_DEPLOYMENT_TARGET (14.0), JOBS (CPU count), PYTHON (python3; stdlib only), CMAKE_ARGS (extra configure
#      arguments, also read by the wheel build; offline: CMAKE_ARGS=-DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=<unpacked v0.5.0>)
# needs: cmake >= 3.18, a C++17 compiler, patch, curl, uv (wheel), network (github.com; pypi.org for the wheel), Metal
set -euo pipefail

MODEL=; WHEEL=1; WHL=
while [ $# -gt 0 ]; do
  case $1 in
    --model) MODEL=$2; shift 2 ;;
    --no-wheel) WHEEL=0; shift ;;
    *) echo "usage: $0 --model <Julia-1-F32.gguf> [--no-wheel]" >&2; exit 2 ;;
  esac
done
die() { echo "package_release: error: $*" >&2; exit 1; }
step() { echo; echo "== $*"; }
PY=${PYTHON:-python3}
[ "$(uname -s)-$(uname -m)" = Darwin-arm64 ] || die "only macOS arm64 is packaged by this script"
[ -n "$MODEL" ] && [ -f "$MODEL" ] || die "model not found: '$MODEL' (--model <path of Julia-1-F32.gguf>)"
MODEL=$(cd "$(dirname "$MODEL")" && pwd)/$(basename "$MODEL")
cd "$(dirname "$0")/.."
ROOT=$PWD
[ $WHEEL = 0 ] || command -v uv > /dev/null || die "uv not found (install it, or pass --no-wheel)"

step "1. version"
VERSION=$(sed -nE 's/^project\(julia1-gguf VERSION ([0-9][0-9.]*) .*/\1/p' CMakeLists.txt)
PYPROJECT=$(sed -nE 's/^version = "(.*)"$/\1/p' pyproject.toml)
PYPACKAGE=$(sed -nE "s/^__version__ = '(.*)'$/\1/p" julia1_gguf/__init__.py)
[ -n "$VERSION" ] && [ "$VERSION" = "$PYPROJECT" ] && [ "$VERSION" = "$PYPACKAGE" ] ||
  die "versions differ: CMakeLists.txt '$VERSION', pyproject.toml '$PYPROJECT', julia1_gguf/__init__.py '$PYPACKAGE'"
grep -q "^## \[$VERSION\]" CHANGELOG.md || die "CHANGELOG.md has no '## [$VERSION]' entry"
export MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-14.0}
NAME=julia1-cli-$VERSION-macos-arm64
echo "version $VERSION, minimum macOS $MACOSX_DEPLOYMENT_TARGET, model $MODEL"

step "2. clean Release build (build-release/)"
B=build-release
rm -rf $B
GEN=(); command -v ninja > /dev/null && GEN=(-G Ninja)
# shellcheck disable=SC2086 # CMAKE_ARGS is a list of arguments
cmake -S . -B $B "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release ${CMAKE_ARGS:-} > $B.configure.log 2>&1 || { tail -30 $B.configure.log; die "configure failed"; }
mv $B.configure.log $B/configure.log
grep "julia1-cli: ggml from" $B/configure.log
cmake --build $B -j "${JOBS:-$(sysctl -n hw.ncpu 2> /dev/null || echo 8)}" > $B/build.log 2>&1 || { tail -30 $B/build.log; die "build failed"; }
CLI=$B/julia1-cli
"$CLI" --version
"$CLI" --version | grep -q "^julia1-cli $VERSION (ggml .*, exact kernels)$" || die "unexpected julia1-cli --version"

step "3. regression smoke (parity, F32)"
S=$B/smoke; mkdir -p $S
REF=data/reference/parity.jsonl; REQ=$S/parity-requests.jsonl
"$PY" -c 'import json, sys
with open(sys.argv[1], encoding="utf-8") as f, open(sys.argv[2], "w", encoding="utf-8") as o:
    for line in f:
        if line.strip():
            o.write(json.dumps(json.loads(line)["request"], ensure_ascii=False) + "\n")' $REF $REQ
guard() { # guard <label> <output> <max abs>: exact guard against the reference, one summary line
  "$PY" tools/compare_replay.py --reference $REF --output "$2" --report "$2.report.json" --min-argmax "$(wc -l < $REF)" \
    --max-abs "$3" > "$2.guard.txt" || { cat "$2.guard.txt"; die "guard failed: $1"; }
  "$PY" -c 'import json, sys; r = json.load(open(sys.argv[1]))
e = r["encoding_mismatches"]
print("  %s: argmax %d/%d, max abs %.3e%s" % (sys.argv[2], r["argmax_agreement"], r["rows"], r["max_abs_logit_delta"],
      "" if e is None else ", encoding mismatches %d" % e))' "$2.report.json" "$1"
}
ENC=(--max-length 1024 --head-length 256 --strict)
"$CLI" --model "$MODEL" --device metal decide --input $REQ --output $S/metal-decide.jsonl "${ENC[@]}" 2> $S/metal-decide.err
guard "Metal exact, decide" $S/metal-decide.jsonl 1e-3
"$CLI" --model "$MODEL" --device metal replay --input $REF --output $S/metal-replay.jsonl 2> $S/metal-replay.err
guard "Metal exact, replay" $S/metal-replay.jsonl 1e-3
"$CLI" --model "$MODEL" --device cpu --threads 4 decide --input $REQ --output $S/cpu-decide.jsonl "${ENC[@]}" 2> $S/cpu-decide.err
guard "CPU, decide" $S/cpu-decide.jsonl 2e-3

step "4. install + tarball"
mkdir -p dist
rm -f dist/$NAME.tar.gz dist/$NAME.tar.gz.sha256 dist/julia1_gguf-$VERSION-*.whl dist/julia1_gguf-$VERSION-*.whl.sha256
cmake --install $B --prefix "$ROOT/$B/stage/$NAME" > $B/install.log
# no local user names, no macOS extended attributes
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata --uid 0 --gid 0 --numeric-owner -C $B/stage -czf dist/$NAME.tar.gz $NAME
(cd dist && shasum -a 256 $NAME.tar.gz > $NAME.tar.gz.sha256 && cat $NAME.tar.gz.sha256)

if [ $WHEEL = 1 ]; then
  step "5. Python wheel (uv build --wheel)"
  uv build --wheel --out-dir dist > $B/wheel.log 2>&1 || { tail -30 $B/wheel.log; die "wheel build failed"; }
  WHL=$(cd dist && ls julia1_gguf-$VERSION-*.whl)
  (cd dist && shasum -a 256 "$WHL" > "$WHL.sha256" && cat "$WHL.sha256")
fi

step "6. checks outside the build tree ($B/relcheck)"
R=$ROOT/$B/relcheck; SERVER=
cleanup() {
  [ -z "$SERVER" ] || kill "$SERVER" 2> /dev/null || true
  [ "${KEEP_RELCHECK:-0}" = 1 ] || rm -rf "$R"
}
trap cleanup EXIT
rm -rf "$R"; mkdir -p "$R"
tar -xzf dist/$NAME.tar.gz -C "$R"
X=$R/$NAME
FILES=$(cd "$X" && find . -type f -o -type l | sort | tr '\n' ' ')
EXPECTED="./bin/julia1-cli ./include/julia1_gguf.h ./lib/libjulia1_gguf_native.dylib ./share/julia1-cli/LICENSE ./share/julia1-cli/NOTICE ./share/julia1-cli/README-quickstart.md ./share/julia1-cli/THIRD_PARTY_LICENSES "
[ "$FILES" = "$EXPECTED" ] || die "unexpected tarball contents: $FILES"
echo "  contents: $FILES(no wheel, no Python files)"
libs() { # the non-system libraries a Mach-O file loads (its own install name excluded)
  local own; own=$(otool -D "$1" | sed -n 2p)
  otool -L "$1" | tail -n +2 | awk '{print $1}' | grep -v -x -F "${own:-/}" | grep -v -E '^(/usr/lib/|/System/Library/)' || true
}
for f in "$X/bin/julia1-cli" "$X/lib/libjulia1_gguf_native.dylib"; do
  [ -z "$(libs "$f")" ] || die "$f loads non-system libraries: $(libs "$f")"
  ! strings -a "$f" | grep -q -F "$ROOT" || die "$f contains paths of the build tree ($ROOT)"
  echo "  ${f#"$X"/}: system libraries only ($(otool -L "$f" | tail -n +2 | awk '{print $1}' | grep -v -x -F "$(otool -D "$f" | sed -n 2p)" | sed -E 's#.*/##' | tr '\n' ' ')); minos $(otool -l "$f" | awk '/LC_BUILD_VERSION/ {b = 1} b && /minos/ {print $2; exit}')"
done
cd "$R"
ln -s "$MODEL" Julia-1-F32.gguf # the file name the quickstart uses
"$X/bin/julia1-cli" --version
"$X/bin/julia1-cli" --model Julia-1-F32.gguf decide --input "$ROOT/$REQ" --output decide.jsonl "${ENC[@]}" 2> decide.err
grep -q "using embedded metal library" decide.err || die "the Metal library was not loaded from the binary"
echo "  Metal library: embedded"
(cd "$ROOT" && guard "tarball julia1-cli, Metal exact, decide" "$R/decide.jsonl" 1e-3)

# the quickstart's predict line, through the CLI, the HTTP server and the C API
QUESTIONS='{"state": "I was charged twice for the same order.", "questions": {"team": {"type": "choice", "instructions": "Which team should handle this request?", "criteria": {"billing": "Billing and payment disputes", "shipping": "Shipping and delivery", "access": "Account access and login"}}}}'
printf '%s\n' "$QUESTIONS" > questions.jsonl
"$X/bin/julia1-cli" --model Julia-1-F32.gguf predict --input questions.jsonl --output answers.jsonl --max-length 8192 --head-length 512 --strict 2> predict.err
grep -q '"choice":"billing"' answers.jsonl || die "predict: unexpected answer $(cat answers.jsonl)"
echo "  CLI predict: $(cat answers.jsonl)"

"$X/bin/julia1-cli" --model Julia-1-F32.gguf serve --port 0 2> serve.log & SERVER=$!
URL=
for _ in $(seq 1 600); do
  URL=$(sed -nE 's#^julia1-cli: serving on (http://[^ ]+) .*#\1#p' serve.log)
  [ -n "$URL" ] && break
  kill -0 $SERVER 2> /dev/null || { cat serve.log; die "serve exited"; }
  sleep 0.1
done
[ -n "$URL" ] || { cat serve.log; die "serve did not start"; }
curl -sf "$URL/v1/predict" -H 'Content-Type: application/json' -d "$QUESTIONS" > served.json || die "POST /v1/predict failed"
kill -INT $SERVER; RC=0; wait $SERVER || RC=$?; SERVER=
[ $RC = 0 ] || die "serve exited with $RC after SIGINT"
cmp -s served.json <(tr -d '\n' < answers.jsonl) || die "/v1/predict body differs from julia1-cli predict: $(cat served.json)"
echo "  serve: $URL/v1/predict body identical to the CLI answer; stopped with exit code 0 on SIGINT"

awk '/^```c$/ {c = 1; next} /^```/ {c = 0} c' "$X/share/julia1-cli/README-quickstart.md" > example.c
cc -I "$X/include" example.c -L "$X/lib" -ljulia1_gguf_native -Wl,-rpath,"$X/lib" -o example
./example > example.out 2> example.err
sed -n 1p example.out | grep -q "^julia1-engine $VERSION " || die "julia1_version(): $(sed -n 1p example.out)"
cmp -s <(sed -n 2p example.out) answers.jsonl || die "C API result differs from julia1-cli predict: $(sed -n 2p example.out)"
echo "  C API (quickstart example.c): $(sed -n 1p example.out); predict_typed result identical to the CLI answer"

if [ $WHEEL = 1 ]; then
  unzip -Z1 "$ROOT/dist/$WHL" | sort | tr '\n' ' ' | sed 's/^/  wheel contents: /'; echo
  uv venv -q --python '>=3.10' venv
  uv pip install -q --python venv/bin/python "$ROOT/dist/$WHL"
  env -u JULIA1_GGUF_LIB venv/bin/python - questions.jsonl answers.jsonl 2> wheel.err <<'PY' || { cat wheel.err; die "wheel check failed"; }
import json, sys
import julia1_gguf
from julia1_gguf import native
engine = julia1_gguf.load_model('Julia-1-F32.gguf', max_length=8192, head_length=512, strict_encoding=True)
line = json.loads(open(sys.argv[1]).read())
result = engine.predict(state=line['state'], questions=line['questions'])
assert engine.backend == 'native', engine.backend
assert native.library_path().startswith(sys.prefix), native.library_path()
assert result == json.loads(open(sys.argv[2]).read()), result
print(f'  wheel in a fresh venv: julia1_gguf {julia1_gguf.__version__}, {native.version()}, backend native, '
      f'library {native.library_path()[len(sys.prefix) + 1:]}; predict equal to the CLI answer')
PY
  dylib=$(find venv -name 'libjulia1_gguf_native.dylib' | head -1)
  [ -z "$(libs "$dylib")" ] || die "the wheel's library loads non-system libraries: $(libs "$dylib")"
  ! strings -a "$dylib" | grep -q -F "$ROOT" || die "the wheel's library contains paths of the build tree ($ROOT)"
  echo "  wheel library: system libraries only; minos $(otool -l "$dylib" | awk '/LC_BUILD_VERSION/ {b = 1} b && /minos/ {print $2; exit}')"
fi
cd "$ROOT"

step "done"
ls -l dist/$NAME.tar.gz* ${WHL:+dist/$WHL*}
