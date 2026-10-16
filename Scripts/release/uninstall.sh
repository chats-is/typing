#!/bin/bash
# 智能拼音 卸载：移除输入法本体；学习数据在 ~/Library/Containers/com.john.inputmethod.Typing，按提示选择是否一并删除。
set -euo pipefail
pkill -x Typing 2>/dev/null || true
rm -rf "$HOME/Library/Input Methods/Typing.app"
echo "已移除输入法。请在 系统设置 › 键盘 › 输入法 里把“智能拼音”从列表删掉。"
read -r -p "同时删除本机学习数据和设置？[y/N] " answer
if [ "${answer:-N}" = "y" ] || [ "${answer:-N}" = "Y" ]; then
    rm -rf "$HOME/Library/Containers/com.john.inputmethod.Typing"
    echo "学习数据已删除。"
fi
