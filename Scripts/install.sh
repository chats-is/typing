#!/bin/bash
# Builds the input method, installs it into /Library/Input Methods (the same place the .pkg uses;
# asks for the administrator password once) and registers it with the system.
set -euo pipefail
cd "$(dirname "$0")/.."

Scripts/build.sh

DEST="/Library/Input Methods/Typing.app"
# Stop the running copy so the system relaunches the new build.
pkill -x Typing 2>/dev/null || true
sudo rm -rf "$DEST"
sudo cp -R build/Typing.app "$DEST"
# A copy from the old per-user install would show up as a second "智能输入".
rm -rf "$HOME/Library/Input Methods/Typing.app"

swiftc -O Tools/register.swift -o build/tools/register
if build/tools/register "$DEST"; then
    echo "installed and enabled: $DEST"
else
    echo "installed: $DEST"
    echo "系统还没列出它：到 系统设置 › 键盘 › 输入法 添加“智能输入”，或注销后重新登录。"
fi
