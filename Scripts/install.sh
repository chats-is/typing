#!/bin/bash
# Builds the input method, installs it into ~/Library/Input Methods and enables it.
set -euo pipefail
cd "$(dirname "$0")/.."

Scripts/build.sh

DEST="$HOME/Library/Input Methods/Typing.app"
mkdir -p "$HOME/Library/Input Methods"
# Stop the running copy so the system relaunches the new build.
pkill -x Typing 2>/dev/null || true
rm -rf "$DEST"
cp -R build/Typing.app "$DEST"

# Launch Services must resolve the bundle id to the installed copy, not to build/Typing.app (or a
# deleted staging copy): the system only lists input methods that live in an Input Methods folder,
# and it looks the bundle up by id. Unregister the build copy, register the installed one.
LSREGISTER=/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister
"$LSREGISTER" -u "$PWD/build/Typing.app" >/dev/null 2>&1 || true
"$LSREGISTER" -f -R "$DEST" >/dev/null 2>&1 || true

swiftc -O Tools/register.swift -o build/tools/register
# On macOS 27 the first install of a bundle id ends here with the Settings pane open: the person
# turns the input method on under 登录项与扩展 › 扩展 › 输入方式, then adds it under 键盘 › 输入法.
build/tools/register "$DEST" || true
echo "installed $DEST"
