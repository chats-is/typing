#!/usr/bin/env python3
"""Write Data/eval_typo.tsv: the evaluation sentences with one random keystroke slip each.

The slip is a swap of two neighbouring letters, a hit on an adjacent key, a dropped letter
or a doubled letter, anywhere in the pinyin (syllable boundaries included).
"""
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]


def neighbours(letter):
    for row in ROWS:
        i = row.find(letter)
        if i >= 0:
            return [row[j] for j in (i - 1, i + 1) if 0 <= j < len(row)]
    return []


def slip(text, rng):
    while True:
        i = rng.randrange(len(text))
        kind = rng.choice(["swap", "adjacent", "drop", "double"])
        if kind == "swap" and i + 1 < len(text) and text[i] != text[i + 1]:
            return text[:i] + text[i + 1] + text[i] + text[i + 2:]
        if kind == "adjacent" and neighbours(text[i]):
            return text[:i] + rng.choice(neighbours(text[i])) + text[i + 1:]
        if kind == "drop":
            return text[:i] + text[i + 1:]
        if kind == "double":
            return text[:i] + text[i] + text[i:]


def main():
    rng = random.Random(7)
    with open(os.path.join(ROOT, "Data", "eval.tsv"), encoding="utf-8") as source, \
            open(os.path.join(ROOT, "Data", "eval_typo.tsv"), "w", encoding="utf-8") as out:
        for line in source:
            pinyin, sentence = line.rstrip("\n").split("\t")
            out.write(slip(pinyin.replace(" ", ""), rng) + "\t" + sentence + "\n")


if __name__ == "__main__":
    main()
