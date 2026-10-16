#!/usr/bin/env python3
"""Write Data/eval_abbr.tsv: the evaluation sentences with one word typed in abbreviated form.

In each sentence one word of 2-4 characters is typed either as initials only ("srf" for 输入法)
or with its first syllable in full and the rest as initials ("shurf"). Run with the project venv.
"""
import os
import random

import build_bigram

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def word_spans(sentence):
    """(start, end) of each lexicon word in the most probable segmentation."""
    n = len(sentence)
    best = [-1e18] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    for i in range(n):
        for j in range(i + 1, min(n, i + build_bigram.MAX_WORD_CHARS) + 1):
            entry = build_bigram.WORDS.get(sentence[i:j])
            if entry is not None and best[i] + entry[1] > best[j]:
                best[j] = best[i] + entry[1]
                back[j] = i
    spans = []
    j = n
    while j > 0:
        spans.append((back[j], j))
        j = back[j]
    return spans[::-1]


def main():
    build_bigram.load_words()
    rng = random.Random(11)
    written = 0
    with open(os.path.join(ROOT, "Data", "eval.tsv"), encoding="utf-8") as source, \
            open(os.path.join(ROOT, "Data", "eval_abbr.tsv"), "w", encoding="utf-8") as out:
        for line in source:
            pinyin, sentence = line.rstrip("\n").split("\t")
            syllables = pinyin.split(" ")
            words = [(i, j) for i, j in word_spans(sentence) if 2 <= j - i <= 4]
            if not words:
                continue
            start, end = rng.choice(words)
            keep_first = rng.random() < 0.5
            for k in range(start + (1 if keep_first else 0), end):
                syllables[k] = syllables[k][0]
            out.write("".join(syllables) + "\t" + sentence + "\n")
            written += 1
    print(f"wrote Data/eval_abbr.tsv: {written} sentences")


if __name__ == "__main__":
    main()
