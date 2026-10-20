#!/bin/bash
# Drives the input controller with a fake text client (no Accessibility permission needed):
# keys, candidates, paging, symbols, quotes, language switching. Needs a release build.
set -euo pipefail
cd "$(dirname "$0")/.."
swift build -c release --product Typing >/dev/null 2>&1 || swift build -c release --product Typing
BIN="$(swift build -c release --show-bin-path)"
OUT=build/tools/symtest
# A home of its own for each run: the learning store and the defaults the test writes. (HOME alone does
# not move Application Support; CFFIXED_USER_HOME does.)
rm -rf "$OUT.home"
mkdir -p build/tools "$OUT.home"
# The Engine module's objects: one Engine.o on the swiftbuild layout, a folder of objects on plain SwiftPM.
if [ -f "$BIN/Engine.o" ]; then OBJECTS=("$BIN/Engine.o"); else OBJECTS=("$BIN"/Engine.build/*.o); fi
swiftc -O -swift-version 5 -I "$BIN" -I "$BIN/Modules" -F "$BIN/PackageFrameworks" -framework Cocoa -framework InputMethodKit -framework SwiftUI \
    -framework CoreML -framework Metal -framework MetalPerformanceShadersGraph -lsqlite3 "${OBJECTS[@]}" \
    App/TypingInputController.swift App/CandidatePanel.swift App/Learning.swift App/SettingsWindow.swift App/Shortcuts.swift Tools/symtest/main.swift -o "$OUT"
HOME="$PWD/$OUT.home" CFFIXED_USER_HOME="$PWD/$OUT.home" "$OUT"
