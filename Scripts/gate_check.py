#!/usr/bin/env python3
"""Reads what Scripts/gate.sh printed and says whether it passes.

    Scripts/gate_check.py Scripts/gate_baseline.tsv gate.log

- Every set in the baseline must come out no lower than its number, less one sentence (the same
  command can differ by one near tie between runs). Sets not in the baseline are only shown.
- The learning session must be kept, the stress run must end without a fault, the key tests must
  all pass.
- Speed is judged against fixed limits, not against an earlier run (the same code measures
  2.4 to 7.3 ms a key depending on what else the Mac is doing): 1–6 letters at most 16 ms a key on
  average (one frame), no buffer length slower than 50 ms in its slowest 5 %, the model's reorder
  within 100 ms in its slowest 5 %. With CI set (a shared cloud machine) speed is shown, not judged.
Exits 1 when anything fails.
"""
import os
import re
import statistics
import sys


def main():
    baseline = {}
    with open(sys.argv[1], encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                name, value = line.split("\t")
                baseline[name] = float(value)
    text = open(sys.argv[2], encoding="utf-8").read()
    failures, judged = [], 0
    for name, sentences, exact in re.findall(r"^(\S+)\s+sentences: (\d+)\s+exact: ([\d.]+)%", text, re.M):
        if name not in baseline:
            continue
        judged += 1
        allowed = baseline[name] - 100 / int(sentences) - 1e-9
        if float(exact) < allowed:
            failures.append(f"{name}: {exact}% < {baseline[name]:.2f}%")
    missing = [n for n in baseline if not re.search(rf"^{re.escape(n)}\s+sentences:", text, re.M)]
    failures += [f"{n}: not run" for n in missing]
    if not re.search(r"^session: KEPT", text, re.M):
        failures.append("learning session not kept")
    if not re.search(r"without a fault", text):
        failures.append("stress run had a fault")
    if not re.search(r"^ALL OK", text, re.M):
        failures.append("key tests failed")

    speed = []
    per_key = re.search(r"1-6 letters: mean ([\d.]+)", text)
    if per_key and float(per_key.group(1)) > 16:
        speed.append(f"1–6 letters {per_key.group(1)} ms a key on average > 16")
    for group, p95 in re.findall(r"(\d+-\d+) letters: mean [\d.]+ p95 ([\d.]+)", text):
        if float(p95) > 50:
            speed.append(f"{group} letters slowest 5 % {p95} ms > 50")
    blocked = [float(v) for v in re.findall(r"main thread blocked per key ms: mean [\d.]+  p95 ([\d.]+)", text)]
    reorder = [float(v) for v in re.findall(r"reorder landed after ms: mean [\d.]+  p95 ([\d.]+)", text)]
    if blocked and statistics.median(blocked) > 50:
        speed.append(f"main thread slowest 5 % {statistics.median(blocked)} ms (median of {len(blocked)} runs) > 50")
    if reorder and statistics.median(reorder) > 100:
        speed.append(f"reorder slowest 5 % {statistics.median(reorder)} ms (median of {len(reorder)} runs) > 100")
    if os.environ.get("CI"):
        for s in speed:
            print(f"(speed, not judged on CI) {s}")
    else:
        failures += speed

    print(f"== verdict: {judged} sets against {sys.argv[1]}")
    if failures:
        for f in failures:
            print(f"FAIL {f}")
        print("GATE FAILED")
        sys.exit(1)
    print("GATE PASSED")


if __name__ == "__main__":
    main()
