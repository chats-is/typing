#!/bin/bash
# Unpacks a resources archive made by Scripts/package_resources.sh into Resources/, verifying
# its checksum: Scripts/fetch_resources.sh <archive.tar.zst> [archive.sha256]
# (With Git LFS installed the files come with `git clone`; this is for everyone else.)
set -euo pipefail
cd "$(dirname "$0")/.."
ARCHIVE="${1:?usage: Scripts/fetch_resources.sh <typing-resources-*.tar.zst> [.sha256]}"
SUM="${2:-${ARCHIVE%.tar.zst}.sha256}"
if [ -f "$SUM" ]; then
    (cd "$(dirname "$ARCHIVE")" && shasum -a 256 -c "$(basename "$SUM")")
else
    echo "no checksum file beside the archive; unpacking unverified" >&2
fi
tar --zstd -xf "$ARCHIVE"
ls -la Resources/lexicon.sqlite Resources/bigram.bin Resources/positions.bin Resources/LanguageModel.mlmodelc/weights/weight.bin
