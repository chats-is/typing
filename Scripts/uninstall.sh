#!/bin/bash
# Removes the input method: Scripts/uninstall.sh
# Cancels use, then deletes
# /Library/Input Methods/Typing.app (asks for the administrator password once),
# forgets the installer receipt, and deletes the learning data inside
# ~/Library/Containers/is.chats.inputmethod.Typing.
# The container directory and its system metadata file are left in place.
# Turning the input source off can fail. The files are still deleted.
set -euo pipefail
cd "$(dirname "$0")/.."
APP="/Library/Input Methods/Typing.app"
BUNDLE_ID="is.chats.inputmethod.Typing"

# Password first. A prompt between canceling use and deleting lets the input method start again.
sudo -v

# 1. Cancel use while the bundle is still installed. Failure does not stop deletion.
mkdir -p build/tools
if swiftc -O Tools/disable.swift -o build/tools/disable; then
    build/tools/disable || true
fi

# 2. Stop the process, then remove the app (both install locations).
pkill -x Typing 2>/dev/null || true
if [ -d "$APP" ]; then sudo rm -rf "$APP"; fi
rm -rf "$HOME/Library/Input Methods/Typing.app"

# 3. Unregister after the files are gone. Doing it while the bundle is still on disk leaves a "not found" row.
LSREGISTER=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister
for app in "$APP" "$HOME/Library/Input Methods/Typing.app"; do
    "$LSREGISTER" -u "$app" || true
done

# 4. Forget the installer receipt.
sudo pkgutil --forget "$BUNDLE_ID.pkg" >/dev/null 2>&1 || true

# 5. Learning data and settings the program wrote.
# The container directory and the system metadata file inside it cannot be removed.
container="$HOME/Library/Containers/$BUNDLE_ID"
if [ -d "$container" ]; then
    find "$container" -mindepth 1 -maxdepth 1 ! -name '.com.apple.containermanagerd.metadata.plist' -exec rm -rf {} +
fi
echo "removed the input method and its learning data"
