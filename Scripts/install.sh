#!/bin/bash
# Builds the input method, installs it into ~/Library/Input Methods and enables it.
set -euo pipefail
cd "$(dirname "$0")/.."

Scripts/build.sh

DEST="$HOME/Library/Input Methods/Zhiyin.app"
mkdir -p "$HOME/Library/Input Methods"
# Stop the running copy so the system relaunches the new build.
pkill -x Zhiyin 2>/dev/null || true
rm -rf "$DEST"
cp -R build/Zhiyin.app "$DEST"

swiftc -O Tools/register.swift -o build/tools/register
build/tools/register "$DEST"
echo "installed $DEST"
