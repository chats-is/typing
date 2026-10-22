#!/usr/bin/env python3
"""Whether two models really differ on a set, sentence by sentence.

    Tools/compare_failures.py A.out B.out

A.out and B.out are `enginecli --eval SET --show-failures` outputs of the two models on the same
set. Prints how many sentences B gets wrong that A got right ("newly wrong") and the other way,
with the two-sided exact McNemar p: below 0.05 the difference is unlikely to be chance (two
training runs of the same recipe differ by a few sentences either way).
"""
import math
import sys


def failures(path):
    with open(path, encoding="utf-8") as f:
        return {tuple(line.split("\t")[:2]) for line in f if line.count("\t") >= 3}


def main():
    a, b = failures(sys.argv[1]), failures(sys.argv[2])
    lost, gained = len(b - a), len(a - b)
    n, k = lost + gained, min(lost, gained)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    print(f"newly wrong {lost}, newly right {gained}, p={p:.2f}")


if __name__ == "__main__":
    main()
