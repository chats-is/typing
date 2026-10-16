#!/bin/bash
# Packs the large resources into one release archive with checksums, for people who clone
# without Git LFS: Scripts/package_resources.sh [out dir, default build]
# → <out>/typing-resources-<date>.tar.zst and .sha256
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-build}"
mkdir -p "$OUT"
NAME="typing-resources-$(date +%Y%m%d)"
FILES=(Resources/lexicon.sqlite Resources/bigram.bin Resources/LanguageModel.mlmodelc Resources/weights.json Resources/positions.bin Resources/vocab.json)
for f in "${FILES[@]}"; do [ -e "$f" ] || { echo "missing $f" >&2; exit 1; }; done
tar --zstd -cf "$OUT/$NAME.tar.zst" "${FILES[@]}"
(cd "$OUT" && shasum -a 256 "$NAME.tar.zst" > "$NAME.sha256")
ls -la "$OUT/$NAME.tar.zst" "$OUT/$NAME.sha256"
