#!/bin/bash
# Drives the input controller with a fake text client (no Accessibility permission needed):
# keys, candidates, paging, symbols, quotes, language switching. Needs a release build.
set -euo pipefail
cd "$(dirname "$0")/.."
swift build -c release --product Zhiyin >/dev/null 2>&1 || swift build -c release --product Zhiyin
BIN="$(swift build -c release --show-bin-path)"
OUT=build/tools/symtest
mkdir -p build/tools "$OUT.home"
# The Engine module's objects: one Engine.o on the swiftbuild layout, a folder of objects on plain SwiftPM.
if [ -f "$BIN/Engine.o" ]; then OBJECTS=("$BIN/Engine.o"); else OBJECTS=("$BIN"/Engine.build/*.o); fi
swiftc -O -swift-version 5 -I "$BIN" -I "$BIN/Modules" -F "$BIN/PackageFrameworks" -framework Cocoa -framework InputMethodKit -framework SwiftUI \
    -framework CoreML -framework Metal -framework MetalPerformanceShadersGraph -lsqlite3 "${OBJECTS[@]}" \
    App/ZhiyinInputController.swift App/CandidatePanel.swift App/Learning.swift App/SettingsWindow.swift App/Shortcuts.swift Tools/symtest/main.swift -o "$OUT"
HOME="$PWD/$OUT.home" "$OUT"
