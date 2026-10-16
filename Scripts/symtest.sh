#!/bin/bash
# Drives the input controller with a fake text client (no Accessibility permission needed):
# keys, candidates, paging, symbols, quotes, language switching. Needs a release build (swift build -c release);
# the product path below is the one SwiftPM uses on this toolchain (.build/release points at it).
set -euo pipefail
cd "$(dirname "$0")/.."
P=.build/out/Products/Release
OUT=build/tools/symtest
mkdir -p build/tools "$OUT.home"
swiftc -O -swift-version 5 -I "$P" -F "$P/PackageFrameworks" -framework Cocoa -framework InputMethodKit -framework SwiftUI \
    -framework CoreML -framework Metal -framework MetalPerformanceShadersGraph -lsqlite3 "$P/Engine.o" \
    App/TypingInputController.swift App/CandidatePanel.swift App/Learning.swift App/SettingsWindow.swift App/Shortcuts.swift Tools/symtest/main.swift -o "$OUT"
HOME="$PWD/$OUT.home" "$OUT"
