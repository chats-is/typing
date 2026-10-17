#!/bin/bash
# Builds a disk image for drag-and-drop installation: Scripts/release/make_dmg.sh <version> [out dir]
# The image holds Typing.app, a link to /Library/Input Methods to drop it on, and a short note
# (the .pkg is published beside it; putting both in one image doubles the download).
set -euo pipefail
cd "$(dirname "$0")/../.."
VERSION="${1:?usage: make_dmg.sh <version> [out dir]}"
OUT="${2:-dist}"
[ -d build/Typing.app ] || Scripts/build.sh
STAGE="$(mktemp -d)/Typing"
mkdir -p "$STAGE" "$OUT"
cp -R build/Typing.app "$STAGE/"
ln -s "/Library/Input Methods" "$STAGE/Input Methods"
cat > "$STAGE/安装说明.txt" <<'NOTE'
智能输入 安装

把 Typing.app 拖到旁边的 Input Methods 文件夹里（系统会要求管理员密码），
然后到 系统设置 › 键盘 › 输入法 添加“智能输入”，或注销后重新登录。
想一步装好并自动启用，用发布页上的 .pkg 安装包。

本版本为本机临时签名，首次打开若被系统拦下，到 系统设置 › 隐私与安全性 里选择“仍要打开”。
卸载：删除 /Library/Input Methods/Typing.app。
NOTE
rm -f "$OUT/Typing-$VERSION.dmg"
hdiutil create -volname "Typing $VERSION" -srcfolder "$STAGE" -ov -format UDZO -quiet "$OUT/Typing-$VERSION.dmg"
rm -rf "$(dirname "$STAGE")"
(cd "$OUT" && shasum -a 256 "Typing-$VERSION.dmg" > "Typing-$VERSION.dmg.sha256")
ls -la "$OUT/Typing-$VERSION.dmg"
