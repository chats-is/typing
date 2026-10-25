#!/bin/bash
# Builds build/app.noindex/Typing.app (sandboxed, ad-hoc signed).
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

# Apple silicon only.
swift build -c release --arch arm64 --product Typing
BIN="$(swift build -c release --arch arm64 --show-bin-path)"

mkdir -p build/tools
if [ ! -f Resources/AppIcon.icns ] || [ Tools/make_icons.swift -nt Resources/AppIcon.icns ]; then
    swiftc -O Tools/make_icons.swift -o build/tools/make_icons
    build/tools/make_icons Resources
    iconutil -c icns Resources/AppIcon.iconset -o Resources/AppIcon.icns
    rm -rf Resources/AppIcon.iconset
fi

# In a folder macOS does not index: an app anywhere else is soon registered with Launch Services
# again, and with two apps for the bundle id the system may launch the build copy when the
# installed one restarts, or none at all (the input source then drops out of the input menu). Unregistering after each build is not enough.
source Scripts/app_path.sh
LSREGISTER=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister
# A copy at build/Typing.app would stay registered as a second app: unregister and remove it.
if [ -d build/Typing.app ]; then
    "$LSREGISTER" -u "$PWD/build/Typing.app" >/dev/null 2>&1 || true
    rm -rf build/Typing.app
fi
mkdir -p "$(dirname "$APP")"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp Bundle/Info.plist "$APP/Contents/Info.plist"
cp -R Bundle/*.lproj "$APP/Contents/Resources/"
cp "$BIN/Typing" "$APP/Contents/MacOS/Typing"
# swift build stamps the deployment target as the SDK version too, and AppKit picks the window style
# from that SDK version: stamped 15.0, macOS 26 and later draw the app in the old style. Record the SDK
# it was really built with; the minimum system stays LSMinimumSystemVersion, so each system still
# draws its own style.
MIN_OS="$(/usr/libexec/PlistBuddy -c 'Print :LSMinimumSystemVersion' Bundle/Info.plist)"
SDK_VERSION="$(xcrun --sdk macosx --show-sdk-version)"
vtool -set-build-version macos "$MIN_OS" "$SDK_VERSION" -replace -output "$APP/Contents/MacOS/Typing" "$APP/Contents/MacOS/Typing"
cp Resources/icon.pdf Resources/menu.pdf Resources/palette.png Resources/palette@2x.png Resources/AppIcon.icns "$APP/Contents/Resources/"
# The engine: the lexicon (what the keys can be) and the language model (what fits the context).
if [ -d Resources/LanguageModel.mlmodelc ] && [ -f Resources/lexicon.sqlite ]; then
    cp -R Resources/LanguageModel.mlmodelc Resources/vocab.json Resources/lexicon.sqlite "$APP/Contents/Resources/"
else
    echo "error: Resources/LanguageModel.mlmodelc or Resources/lexicon.sqlite is missing (Scripts/build_model.sh, Tools/build_lexicon.py)" >&2
    exit 1
fi
if [ -f Resources/bigram.bin ]; then cp Resources/bigram.bin "$APP/Contents/Resources/"; fi
# The shipped model's fingerprint: learned tensors are kept with it, and tensors learned on any other
# model (a retrained one may have the same shape) are dropped and relearned.
cat Resources/LanguageModel.mlmodelc/weights/weight.bin $(ls Resources/positions.bin Resources/weights.json 2>/dev/null) | shasum -a 256 | cut -c1-64 > "$APP/Contents/Resources/LanguageModel.sha256"
# Learning on this machine: the model's header and full position table (the weights themselves are
# read out of the Core ML model), and the general check sets.
for f in Resources/weights.json Resources/positions.bin Resources/general_lattices.tsv Resources/general_words.tsv; do [ -f "$f" ] && cp "$f" "$APP/Contents/Resources/"; done
# What learning records are redacted by (the same rules as the training text); the app stops without it.
cp Resources/redaction.json "$APP/Contents/Resources/"
codesign --force --sign - --entitlements Bundle/Typing.entitlements "$APP"
# Never leave the build copy registered with Launch Services: with two apps for the bundle id the
# system cannot launch the installed one (LaunchInputMethod -50) and nothing can be typed.
"$LSREGISTER" -u "$PWD/$APP" >/dev/null 2>&1 || true
echo "built $APP"
