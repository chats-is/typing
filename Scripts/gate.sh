#!/bin/bash
# The gate before anything is installed: word tests, sentence sets, fresh sets, a learning session, per-key
# timing, pacing, concurrency, key handling. Nothing may come out lower than docs/benchmarks.md:
# Scripts/gate_check.py compares against Scripts/gate_baseline.tsv (the same numbers) at the end and
# the gate fails (exit 1) when anything is lower, unkept or slow enough to notice.
# Usage: Scripts/gate.sh [extra eval file ...]   (e.g. a private replay file; shown, not judged)
set -euo pipefail
cd "$(dirname "$0")/.."
LOG="$(mktemp)"
exec > >(tee "$LOG") 2>&1
swift build -c release >/dev/null 2>&1 || swift build -c release
E="$(swift build -c release --show-bin-path)/enginecli"
R=Resources
C=(--model "$R/LanguageModel.mlmodelc" --vocab "$R/vocab.json" --lexicon "$R/lexicon.sqlite" --bigram "$R/bigram.bin")

echo "== word tests (first choice)"
for f in Data/eval_word_full.tsv Data/eval_word_mixed.tsv Data/eval_word_initials.tsv Data/eval_word_english.tsv "$@"; do
    printf '%-32s ' "$(basename "$f")"
    "$E" "${C[@]}" --eval "$f" 2>/dev/null | grep "sentences:" | sed -E 's/ +among best.*//'
done
echo "== sentence sets (300 each)"
for f in Data/eval_dialogue.tsv Data/eval_news.tsv Data/eval_forum.tsv Data/eval_touchpal.tsv Data/eval_short.tsv; do
    printf '%-32s ' "$(basename "$f")"
    "$E" "${C[@]}" --eval "$f" --limit 300 2>/dev/null | grep "sentences:" | sed -E 's/ +among best.*//'
done
echo "== fresh sets (cleaned held-out text, nothing seen in training, never tuned on: these decide between models)"
for f in Data/fresh/eval_*.tsv; do
    printf '%-32s ' "fresh/$(basename "$f")"
    "$E" "${C[@]}" --eval "$f" 2>/dev/null | grep "sentences:" | sed -E 's/ +among best.*//'
done
echo "== learning session (GPU)"
"$E" "${C[@]}" --graph-scorer "$R/LanguageModel.mlmodelc" --learn-demo forum2 2>/dev/null | grep "session:"
# Speed depends on what else the Mac is doing: the load is shown, pacing is run three times and the
# median judged.
echo "== load: $(sysctl -n vm.loadavg) on $(sysctl -n hw.ncpu) cores"
echo "== per key"
"$E" "${C[@]}" --eval Data/eval_word_mixed.tsv --typing --limit 300 2>/dev/null | grep "per key"
echo "== pace 120 ms (three runs)"
for run in 1 2 3; do
    "$E" "${C[@]}" --eval Data/eval_word_mixed.tsv --pace 120 --limit 60 2>/dev/null | grep -E "pace|reorder"
done
echo "== stress"
"$E" "${C[@]}" --stress 15 2>/dev/null | tail -1
echo "== keys"
Scripts/symtest.sh 2>/dev/null | tail -1
sleep 1  # let tee write everything out
python3 Scripts/gate_check.py Scripts/gate_baseline.tsv "$LOG"
