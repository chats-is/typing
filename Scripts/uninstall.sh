#!/bin/bash
# Removes the input method: Scripts/uninstall.sh [--data]
# Takes it out of the enabled input sources, stops it, deletes /Library/Input Methods/Typing.app
# (asks for the administrator password once) and forgets the installer receipt. The learning data
# and settings in ~/Library/Containers/is.chats.inputmethod.Typing stay unless --data is given.
set -euo pipefail
cd "$(dirname "$0")/.."
APP="/Library/Input Methods/Typing.app"
BUNDLE_ID="is.chats.inputmethod.Typing"

# 1. Disable the input source so Keyboard settings does not keep a dead entry.
mkdir -p build/tools
cat > build/tools/disable.swift <<'SWIFT'
import Carbon
let filter = [kTISPropertyBundleID as String: "is.chats.inputmethod.Typing"] as CFDictionary
for source in TISCreateInputSourceList(filter, true)?.takeRetainedValue() as? [TISInputSource] ?? [] {
    _ = TISDisableInputSource(source)
}
SWIFT
swiftc -O build/tools/disable.swift -o build/tools/disable 2>/dev/null && build/tools/disable || true

# 2. Stop the process.
pkill -x Typing 2>/dev/null || true

# 3. Remove the app (both install locations).
if [ -d "$APP" ]; then sudo rm -rf "$APP"; fi
rm -rf "$HOME/Library/Input Methods/Typing.app"

# 4. Forget the installer receipt.
sudo pkgutil --forget "$BUNDLE_ID.pkg" >/dev/null 2>&1 || true

# 5. Learning data and settings: only when asked.
if [ "${1:-}" = "--data" ]; then
    rm -rf "$HOME/Library/Containers/$BUNDLE_ID"
    echo "removed the input method and its learning data"
else
    echo "removed the input method; learning data and settings kept in ~/Library/Containers/$BUNDLE_ID (Scripts/uninstall.sh --data removes them)"
fi
