#!/usr/bin/env python3
"""How English words are written inside Chinese text, and how often: Data/lexicon/english_casing.tsv.

    Data/venv-train/bin/python Tools/make_english_casing.py

One line per word: "lower<TAB>Usual<TAB>everyday_per_million<TAB>forms", the usual casing being the
most frequent one (iPhone, OK, macOS) and forms each casing seen with its share ("Hello:0.62 hello:0.31"),
so the lexicon can offer more than one way of writing a word.
Words that start a sentence are capitalised for that reason alone, so they are not counted.
The rate is per million characters of the same everyday mixture Training/count_words.py uses,
so English words and Chinese words sit on one scale in the lexicon.
"""
import collections
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Training"))
from count_words import SAMPLE, WEIGHTS  # noqa: E402

DATA = os.path.join(ROOT, "Data", "prepared-clean")  # cased token streams
OUT = os.path.join(ROOT, "Data", "lexicon", "english_casing.tsv")
MIXED_CASE_SHARE = 0.25


def main():
    with open(os.path.join(DATA, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    ids = {t: i for i, t in enumerate(vocab) if len(t) == 1}
    letters = {i: t for t, i in ids.items() if t.isascii() and t.isalpha()}
    letter_ids = np.array(sorted(letters))
    ends = {ids[c] for c in ".!?\n。！？" if c in ids}
    forms = collections.defaultdict(collections.Counter)
    rates = collections.Counter()
    weight_sum = sum(WEIGHTS.values())
    for name, weight in WEIGHTS.items():
        path = os.path.join(DATA, f"{name}.bin")
        if not os.path.exists(path):
            continue
        stream = np.memmap(path, dtype="<u2", mode="r")
        total = min(len(stream), SAMPLE.get(name, len(stream)))
        stream = stream[:total]
        is_letter = np.isin(stream, letter_ids)
        edges = np.flatnonzero(np.diff(np.concatenate(([0], is_letter.astype(np.int8), [0]))))
        counted = collections.Counter()
        for a, b in zip(edges[::2], edges[1::2]):
            if 1 <= b - a <= 20 and a > 1 and not any(int(t) in ends for t in stream[max(0, a - 2):a]):
                word = "".join(letters[t] for t in stream[a:b])
                forms[word.lower()][word] += 1
                counted[word.lower()] += 1
        for word, count in counted.items():
            rates[word] += count * 1e6 / total * weight / weight_sum
        print(f"{name}: {total / 1e6:.0f}M characters, {len(counted):,} words", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    written = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for lower, rate in rates.most_common():
            variants = forms[lower]
            total = sum(variants.values())
            if total < 3:
                continue
            # The commonest casing wins. When plain lowercase is commonest only because it is
            # quicker to type, a deliberate mixed-case spelling with a real share wins instead (macOS, WiFi).
            usual = variants.most_common(1)[0][0]
            if usual == lower:
                for variant, count in variants.most_common():
                    if variant != lower and not variant.isupper() and not variant.istitle() and count >= MIXED_CASE_SHARE * total:
                        usual = variant
                        break
            shares = " ".join(f"{variant}:{count / total:.3f}" for variant, count in variants.most_common())
            f.write(f"{lower}\t{usual}\t{rate:.4f}\t{shares}\n")
            written += 1
    print(f"wrote {OUT}: {written:,} words")


if __name__ == "__main__":
    main()
