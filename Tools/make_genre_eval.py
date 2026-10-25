#!/usr/bin/env python3
"""Write Data/eval_<source>.tsv for every source: held-out sentences with the text before them.

    Data/venv-train/bin/python Tools/make_genre_eval.py [--data Data/prepared-clean]

Each line is "pin yin<TAB>sentence<TAB>preceding text" (up to 60 characters; line breaks written
as a literal \\n). The sentences come from the holdout_<source>.txt files of prepare_data.py,
which the model never trains on.
"""
import argparse
import glob
import os
import random
import re

from pypinyin import lazy_pinyin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAN_RUN = re.compile(r"[一-鿿]+")
SENTENCES = 500
PER_DOCUMENT = 3
CONTEXT_CHARS = 60
MIN_CONTEXT = 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(ROOT, "Data", "prepared-clean"))
    args = parser.parse_args()
    with open(os.path.join(ROOT, "Data", "lexicon", "readings.tsv"), encoding="utf-8") as f:
        syllables = {line.split("\t")[0] for line in f}

    for path in sorted(glob.glob(os.path.join(args.data, "holdout_*.txt"))):
        name = os.path.basename(path)[len("holdout_"):-len(".txt")]
        # Sources of isolated short units have no preceding text; TouchPal has its own test set.
        if name in ("touchpal", "peoples_daily"):
            continue
        with open(path, encoding="utf-8") as f:
            documents = [doc for doc in f.read().split("\n\n") if doc.strip()]
        rng = random.Random(5)
        rng.shuffle(documents)
        # Long documents (forum threads, novels) give several sentences, short ones give one.
        per_document = max(1, min(40, -(-SENTENCES // len(documents)))) if documents else 1
        written = 0
        with open(os.path.join(ROOT, "Data", f"eval_{name}.tsv"), "w", encoding="utf-8") as out:
            for doc in documents:
                matches = [m for m in HAN_RUN.finditer(doc) if 5 <= len(m.group()) <= 18 and m.start() >= MIN_CONTEXT]
                rng.shuffle(matches)
                taken = 0
                for match in matches:
                    sentence = match.group()
                    pinyin = lazy_pinyin(sentence)
                    if len(pinyin) != len(sentence) or any(s not in syllables for s in pinyin):
                        continue
                    context = doc[max(0, match.start() - CONTEXT_CHARS):match.start()]
                    context = context.replace("\t", " ").replace("\\", "/").replace("\n", "\\n")
                    out.write(" ".join(pinyin) + "\t" + sentence + "\t" + context + "\n")
                    written += 1
                    taken += 1
                    if taken == max(per_document, PER_DOCUMENT if len(documents) < SENTENCES else 1) or written == SENTENCES:
                        break
                if written == SENTENCES:
                    break
        print(f"eval_{name}.tsv: {written} sentences from {len(documents)} held-out documents")


if __name__ == "__main__":
    main()
