#!/usr/bin/env python3
"""Write <out>/english.bin: English words as the corpora write them, so the model learns not only
the spelling but the casing of each word (iPhone, macOS, NASA, Python) and how common it is.

    Data/venv-train/bin/python Training/make_english.py --out Data/lm5

Casings are counted in the cased token streams of the training data (web, mixed, techdocs,
forum, dialogue, tatoeba): a word is written in its most frequent casing when that casing has at
least half of its occurrences, else in lower case. Words only in the OpenSubtitles 50k list are
lower case. Lines are runs of 1-8 words drawn by frequency, so a word at the start of a line
(what an isolated word typed without context looks like to the model) is seen often.
"""
import argparse
import collections
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENS = 20_000_000
SAMPLE = {"web": 300_000_000, "mixed": 200_000_000, "techdocs": None, "forum": 100_000_000, "dialogue": 100_000_000, "tatoeba": None}
WORD = re.compile(r"[A-Za-z]+")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--bos", action="store_true", help="start every line with <bos>, as documents do, so an isolated word is read as a document of its own")
    parser.add_argument("--data", default=os.path.join(ROOT, "Data", "lm3"), help="where the cased token streams are")
    args = parser.parse_args()
    with open(os.path.join(args.out, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    ids = {t: i for i, t in enumerate(vocab) if len(t) == 1}
    letters = {i: t for t, i in ids.items() if t.isascii() and t.isalpha()}
    forms = collections.defaultdict(collections.Counter)
    for name, limit in SAMPLE.items():
        path = os.path.join(args.data, f"{name}.bin")
        if not os.path.exists(path):
            continue
        stream = np.memmap(path, dtype="<u2", mode="r")
        if limit is not None and len(stream) > limit:
            start = (len(stream) - limit) // 2
            stream = stream[start : start + limit]
        # Runs of letter tokens are English words; everything else separates them.
        is_letter = np.isin(stream, list(letters))
        edges = np.flatnonzero(np.diff(np.concatenate(([0], is_letter.astype(np.int8), [0]))))
        # A word that starts a sentence is capitalised for that reason alone, so it is not counted.
        ends = {ids[c] for c in ".!?\n。！？" if c in ids}
        for a, b in zip(edges[::2], edges[1::2]):
            if 2 <= b - a <= 20 and not any(int(t) in ends for t in stream[max(0, a - 2):a]) and a > 1:
                word = "".join(letters[t] for t in stream[a:b])
                forms[word.lower()][word] += 1
        print(f"{name}: {len(forms)} words so far", flush=True)
    words, counts = [], []
    for lower, variants in forms.items():
        total = sum(variants.values())
        if total < 3:
            continue
        best, share = variants.most_common(1)[0]
        words.append(best if share * 2 >= total else lower)
        counts.append(total)
    with open(os.path.join(ROOT, "Data", "lexicons", "english", "en_50k.txt"), encoding="utf-8") as f:
        for line in f:
            word, count = line.split()
            if word.isalpha() and word.isascii() and word.lower() not in forms:
                words.append(word.lower())
                counts.append(float(count) / 50)  # subtitles are far larger than the sample above
    probabilities = np.array(counts, dtype=float)
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(5)
    newline = ids["\n"]
    space = ids[" "]
    bos = [1] if args.bos else []
    written = 0
    with open(os.path.join(args.out, "english.bin"), "wb") as out:
        while written < TOKENS:
            chosen = rng.choice(len(words), size=100_000, p=probabilities)
            lengths = rng.integers(1, 9, size=len(chosen))
            tokens = []
            i = 0
            while i < len(chosen):
                line = [words[j] for j in chosen[i : i + lengths[i]]]
                i += lengths[i]
                tokens.extend(bos)
                for k, word in enumerate(line):
                    if k:
                        tokens.append(space)
                    tokens.extend(ids[c] for c in word)
                tokens.append(newline)
            np.array(tokens, dtype="<u2").tofile(out)
            written += len(tokens)
    cased = [w for w in words if w != w.lower()]
    print(f"english.bin: {written / 1e6:.0f}M tokens from {len(words)} words, {len(cased)} cased, e.g. {' '.join(sorted(cased, key=lambda w: -counts[words.index(w)])[:15])}")


if __name__ == "__main__":
    main()
