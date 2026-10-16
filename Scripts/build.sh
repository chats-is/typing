#!/bin/bash
# Builds build/Zhiyin.app (sandboxed, ad-hoc signed).
set -euo pipefail
cd "$(dirname "$0")/.."

swift build -c release --product Zhiyin
BIN="$(swift build -c release --show-bin-path)"

mkdir -p build/tools
if [ ! -f Resources/AppIcon.icns ] || [ Tools/make_icons.swift -nt Resources/AppIcon.icns ]; then
    swiftc -O Tools/make_icons.swift -o build/tools/make_icons
    build/tools/make_icons Resources
    iconutil -c icns Resources/AppIcon.iconset -o Resources/AppIcon.icns
    rm -rf Resources/AppIcon.iconset
fi

APP=build/Zhiyin.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp Bundle/Info.plist "$APP/Contents/Info.plist"
cp -R Bundle/*.lproj "$APP/Contents/Resources/"
cp "$BIN/Zhiyin" "$APP/Contents/MacOS/Zhiyin"
cp Resources/icon.pdf Resources/menu.pdf Resources/palette.png Resources/palette@2x.png Resources/AppIcon.icns "$APP/Contents/Resources/"
# The engine: the lexicon (what the keys can be) and the language model (what fits the context).
if [ -d Resources/LanguageModel.mlmodelc ] && [ -f Resources/lexicon.sqlite ]; then
    cp -R Resources/LanguageModel.mlmodelc Resources/vocab.json Resources/lexicon.sqlite "$APP/Contents/Resources/"
else
    echo "error: Resources/LanguageModel.mlmodelc or Resources/lexicon.sqlite is missing (Scripts/build_model.sh, Tools/build_lexicon.py)" >&2
    exit 1
fi
if [ -f Resources/bigram.bin ]; then cp Resources/bigram.bin "$APP/Contents/Resources/"; fi
# Learning on this machine: the model's header and full position table (the weights themselves are
# read out of the Core ML model), and the general check sets.
for f in Resources/weights.json Resources/positions.bin Resources/general_lattices.tsv Resources/general_words.tsv; do [ -f "$f" ] && cp "$f" "$APP/Contents/Resources/"; done
codesign --force --sign - --entitlements Bundle/Zhiyin.entitlements "$APP"
echo "built $APP"
