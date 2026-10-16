#!/usr/bin/env python3
"""Write Data/eval_abbr_mix.tsv and Data/eval_typo_mix.tsv from the genre test sets.

    Data/venv-train/bin/python Tools/make_input_eval.py

The sentences (and the text before them) come from Data/eval_{dialogue,mixed,news,forum}.tsv,
150 from each, so both sets measure the same kinds of text users type.

Abbreviations, one pattern per sentence:
  word   one word of 2-4 characters typed as initials ("srf" for 输入法)
  first  one word with its first syllable in full and the rest as initials ("shurf")
  two    two words typed as initials
Slips, one per sentence: two neighbouring letters swapped, a neighbouring key hit instead,
a letter dropped or a letter doubled, anywhere in the pinyin.
"""
import os
import random

import build_bigram
from make_abbreviation_eval import word_spans
from make_typo_eval import slip

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENRES = ("dialogue", "mixed", "news", "forum")
PER_GENRE = 150


def abbreviate(syllables, sentence, rng):
    """The sentence's pinyin with one or two words shortened, and the pattern used; None if no word fits."""
    words = [(i, j) for i, j in word_spans(sentence) if 2 <= j - i <= 4]
    if not words:
        return None
    pattern = rng.choice(["word", "first", "two"] if len(words) >= 2 else ["word", "first"])
    chosen = rng.sample(words, 2) if pattern == "two" else [rng.choice(words)]
    typed = list(syllables)
    for start, end in chosen:
        for k in range(start + (1 if pattern == "first" else 0), end):
            typed[k] = typed[k][0]
    return "".join(typed), pattern


def main():
    build_bigram.load_words()
    rng = random.Random(23)
    abbr_lines, typo_lines = [], []
    for genre in GENRES:
        with open(os.path.join(ROOT, "Data", f"eval_{genre}.tsv"), encoding="utf-8") as f:
            rows = [line.rstrip("\n").split("\t") for line in f]
        rows = [row for row in rows if len(row) == 3][-PER_GENRE:]
        for pinyin, sentence, context in rows:
            syllables = pinyin.split(" ")
            abbreviated = abbreviate(syllables, sentence, rng)
            if abbreviated:
                abbr_lines.append(f"{abbreviated[0]}\t{sentence}\t{context}\t{abbreviated[1]}")
            typo_lines.append(f"{slip(''.join(syllables), rng)}\t{sentence}\t{context}")
    for name, lines in (("eval_abbr_mix", abbr_lines), ("eval_typo_mix", typo_lines)):
        with open(os.path.join(ROOT, "Data", f"{name}.tsv"), "w", encoding="utf-8") as out:
            out.write("\n".join(lines) + "\n")
        print(f"wrote Data/{name}.tsv: {len(lines)} sentences")


if __name__ == "__main__":
    main()
