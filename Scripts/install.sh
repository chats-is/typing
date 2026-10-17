#!/bin/bash
# Builds the input method, installs it into /Library/Input Methods (the same place the .pkg uses;
# asks for the administrator password once) and registers it with the system.
set -euo pipefail
cd "$(dirname "$0")/.."

Scripts/build.sh

DEST="/Library/Input Methods/Typing.app"
# Put the new build in place with two renames, then stop the running copy once. The system may
# relaunch the input method the moment it dies; copying over the old files after the kill would
# give that relaunch half-replaced files (it keeps the lexicon open, and typing breaks).
sudo rm -rf "$DEST.new" "$DEST.old"
sudo cp -R build/Typing.app "$DEST.new"
[ -d "$DEST" ] && sudo mv "$DEST" "$DEST.old"
sudo mv "$DEST.new" "$DEST"
pkill -x Typing 2>/dev/null || true
sudo rm -rf "$DEST.old"
# A copy from the old per-user install would show up as a second "智能输入".
rm -rf "$HOME/Library/Input Methods/Typing.app"
# Launch Services must know exactly one app for the bundle id: the system launches the input method
# by bundle id, and with build/Typing.app also registered the launch fails (LaunchInputMethod -50) and
# nothing can be typed. Unregister the build copy.
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -u "$PWD/build/Typing.app" >/dev/null 2>&1 || true

swiftc -O Tools/register.swift -o build/tools/register
if build/tools/register "$DEST"; then
    echo "installed and enabled: $DEST"
else
    echo "installed: $DEST"
    echo "系统还没列出它：到 系统设置 › 键盘 › 输入法 添加“智能输入”，或注销后重新登录。"
fi
