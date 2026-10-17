#!/bin/bash
# Builds build/Typing.app (sandboxed, ad-hoc signed).
set -euo pipefail
cd "$(dirname "$0")/.."

# The identity the system depends on; a wrong value here makes the input method invisible or
# unlaunchable, so the build refuses rather than producing a bundle that cannot be installed.
BUNDLE_ID="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' Bundle/Info.plist)"
CONNECTION="$(/usr/libexec/PlistBuddy -c 'Print :InputMethodConnectionName' Bundle/Info.plist)"
MACH_NAME="$(/usr/libexec/PlistBuddy -c 'Print :com.apple.security.temporary-exception.mach-register.global-name:0' Bundle/Typing.entitlements)"
case "$BUNDLE_ID" in *inputmethod*) ;; *) echo "error: CFBundleIdentifier '$BUNDLE_ID' must contain 'inputmethod' (macOS only treats such bundles as input methods)" >&2; exit 1;; esac
# InputMethodKit registers "<bundle id>_Connection" itself; the sandbox allows only the names in the
# entitlement, so both must be exactly that (the sandbox log otherwise shows "deny mach-register").
[ "$CONNECTION" = "${BUNDLE_ID}_Connection" ] || { echo "error: InputMethodConnectionName must be '${BUNDLE_ID}_Connection', not '$CONNECTION'" >&2; exit 1; }
[ "$CONNECTION" = "$MACH_NAME" ] || { echo "error: the entitlement's mach-register name '$MACH_NAME' must equal InputMethodConnectionName '$CONNECTION'" >&2; exit 1; }

swift build -c release --product Typing
BIN="$(swift build -c release --show-bin-path)"

mkdir -p build/tools
if [ ! -f Resources/AppIcon.icns ] || [ Tools/make_icons.swift -nt Resources/AppIcon.icns ]; then
    swiftc -O Tools/make_icons.swift -o build/tools/make_icons
    build/tools/make_icons Resources
    iconutil -c icns Resources/AppIcon.iconset -o Resources/AppIcon.icns
    rm -rf Resources/AppIcon.iconset
fi

APP=build/Typing.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp Bundle/Info.plist "$APP/Contents/Info.plist"
cp -R Bundle/*.lproj "$APP/Contents/Resources/"
cp "$BIN/Typing" "$APP/Contents/MacOS/Typing"
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
codesign --force --sign - --entitlements Bundle/Typing.entitlements "$APP"
# Never leave the build copy registered with Launch Services: with two apps for the bundle id the
# system cannot launch the installed one (LaunchInputMethod -50) and nothing can be typed.
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -u "$PWD/$APP" >/dev/null 2>&1 || true
echo "built $APP"
