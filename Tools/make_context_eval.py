#!/usr/bin/env python3
"""Write Data/eval_context.tsv: held-out sentences together with the text that came before them.

Each line is "pin yin<TAB>sentence<TAB>preceding text" (up to 60 characters, line breaks as
spaces). It measures what the input method is for: picking the right homophone given what was
already written. Documents come from the cleaned held-out text of Data/prepared-clean (blank-line
separated, never trained on), of the sources listed in SOURCES.

    Data/venv-train/bin/python Tools/make_context_eval.py
"""
import os
import random
import sqlite3

from pypinyin import lazy_pinyin

import build_bigram

ROOT = build_bigram.ROOT
OUT = os.path.join(ROOT, "Data", "eval_context.tsv")
SENTENCES = 1500
# The sources of the held-out text (web, techdocs and tatoeba are left out: pages, manuals, English).
SOURCES = ("dialogue", "forum", "mixed", "news", "novels", "peoples_daily", "touchpal")
CONTEXT_CHARS = 60


def main():
    holdout = []
    for path in [os.path.join(ROOT, "Data", "prepared-clean", f"holdout_{name}.txt") for name in SOURCES]:
        with open(path, encoding="utf-8") as f:
            holdout += [doc for doc in f.read().split("\n\n") if doc.strip()]
    db = sqlite3.connect(build_bigram.LEXICON)
    syllables = {row[0] for row in db.execute("SELECT s FROM syllables")}
    db.close()

    rng = random.Random(3)
    rng.shuffle(holdout)
    written = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for doc in holdout:
            matches = [m for m in build_bigram.HAN_RUN.finditer(doc) if 5 <= len(m.group()) <= 18 and m.start() >= 20]
            if not matches:
                continue
            match = rng.choice(matches)
            sentence = match.group()
            pinyin = lazy_pinyin(sentence)
            if len(pinyin) != len(sentence) or any(s not in syllables for s in pinyin):
                continue
            context = " ".join(doc[max(0, match.start() - CONTEXT_CHARS):match.start()].split())
            f.write(" ".join(pinyin) + "\t" + sentence + "\t" + context.replace("\t", " ") + "\n")
            written += 1
            if written == SENTENCES:
                break
    print(f"wrote {OUT}: {written} sentences")


if __name__ == "__main__":
    main()
