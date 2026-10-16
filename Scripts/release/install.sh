#!/bin/bash
# 智能拼音 安装：把 Typing.app 复制到 ~/Library/Input Methods，去掉下载隔离属性，并注册启用。
# 双击 安装.command 即可；也可在终端运行。
set -euo pipefail
cd "$(dirname "$0")"
DEST="$HOME/Library/Input Methods/Typing.app"
[ -d Typing.app ] || { echo "没有找到 Typing.app，请在解压后的文件夹里运行。"; exit 1; }
mkdir -p "$HOME/Library/Input Methods"
pkill -x Typing 2>/dev/null || true
rm -rf "$DEST"
cp -R Typing.app "$DEST"
xattr -dr com.apple.quarantine "$DEST" 2>/dev/null || true
if [ -x ./register ]; then
    xattr -d com.apple.quarantine ./register 2>/dev/null || true
    ./register "$DEST" || true
fi
echo "已安装到 $DEST。"
echo "如果输入法菜单里还没有“智能拼音”，到 系统设置 › 键盘 › 输入法 添加，或注销后重新登录。"
