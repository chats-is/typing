#!/usr/bin/env python3
"""Write <out>/words.bin: lexicon words as one-word lines, drawn by how often people use them, so
the language model learns which characters form words and how common each word is (what it needs
to read an isolated word typed without any text before it).

    Data/venv-train/bin/python Training/make_words.py --out Data/prepared-clean

Words come from Data/lexicons/all_words.tsv (word, pinyin, weight, count). Only words of 2-4
characters with a positive weight are used; the draw is 85% by weight^0.8 (frequent words recur,
without the top few drowning the rest) and 15% uniform over the 200k most frequent, so the long
tail of names and terms still appears. 40M tokens, like the words source of lm3.
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENS = 40_000_000
TOP_UNIFORM = 200_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--bos", action="store_true", help="start every line with <bos>, as documents do, so an isolated word is read as a document of its own")
    args = parser.parse_args()
    with open(os.path.join(args.out, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    ids = {t: i for i, t in enumerate(vocab) if len(t) == 1}
    words, weights = [], []
    seen = set()
    with open(os.path.join(ROOT, "Data", "lexicons", "all_words.tsv"), encoding="utf-8") as f:
        for line in f:
            word, _pinyin, weight, _count = line.rstrip("\n").split("\t")
            weight = float(weight)
            if not (2 <= len(word) <= 4) or weight <= 0 or word in seen or any(c not in ids for c in word):
                continue
            seen.add(word)
            words.append(word)
            weights.append(weight)
    order = np.argsort(weights)[::-1]
    words = [words[i] for i in order]
    weights = np.array(weights)[order]
    by_frequency = weights ** 0.8
    by_frequency /= by_frequency.sum()
    uniform = np.zeros(len(words))
    uniform[: min(TOP_UNIFORM, len(words))] = 1 / min(TOP_UNIFORM, len(words))
    probabilities = 0.85 * by_frequency + 0.15 * uniform
    rng = np.random.default_rng(4)
    newline = ids["\n"]
    bos = [1] if args.bos else []
    written = 0
    with open(os.path.join(args.out, "words.bin"), "wb") as out:
        while written < TOKENS:
            chosen = rng.choice(len(words), size=200_000, p=probabilities)
            tokens = []
            for i in chosen:
                tokens.extend(bos)
                tokens.extend(ids[c] for c in words[i])
                tokens.append(newline)
            np.array(tokens, dtype="<u2").tofile(out)
            written += len(tokens)
    print(f"words.bin: {written / 1e6:.0f}M tokens from {len(words)} words; top: {' '.join(words[:12])}")


if __name__ == "__main__":
    main()
