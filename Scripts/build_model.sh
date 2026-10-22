#!/bin/bash
# Turns a trained checkpoint into the Core ML model the input method bundles.
# Usage: Scripts/build_model.sh Data/clean2/model.pt [output directory, default Resources]
# The vocabulary is taken from vocab.json next to the checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
CHECKPOINT="${1:?usage: Scripts/build_model.sh <checkpoint.pt> [output directory]}"
OUT="${2:-Resources}"
PACKAGE="$(dirname "$CHECKPOINT")/LanguageModel${LM_ROWS:-64}.mlpackage"  # e.g. LanguageModel32.mlpackage beside the checkpoint
mkdir -p "$OUT"

Data/venv-train/bin/python Training/export_coreml.py "$CHECKPOINT" "$PACKAGE"
mkdir -p build/tools
swiftc -O Tools/compile_model.swift -o build/tools/compile_model
build/tools/compile_model "$PACKAGE" "$OUT/LanguageModel.mlmodelc"
cp "$(dirname "$CHECKPOINT")/vocab.json" "$OUT/vocab.json"
