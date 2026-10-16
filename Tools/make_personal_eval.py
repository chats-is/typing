#!/usr/bin/env python3
"""Material for the level-4 (learning) verification: one long text by one "person", split in time.

    Data/venv-train/bin/python Tools/make_personal_eval.py --source forum --rank 0 --out Data/personal/forum0

Takes the --rank-th largest held-out document of --source (a forum thread, a novel), writes:
  <out>.train.txt   the first --split of it (what the person has typed and confirmed so far)
  <out>.eval.tsv    sentences from the rest, "pin yin<TAB>sentence<TAB>preceding text" (the
                    person's later input, with its context), like Tools/make_genre_eval.py
The language model never trained on held-out documents, so "before" is the shipped model and
"after" is the model tuned on the person's earlier text.
"""
import argparse
import os
import random
import re
import sqlite3

from pypinyin import lazy_pinyin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAN_RUN = re.compile(r"[一-鿿]+")
CONTEXT_CHARS = 60


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="forum")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--split", type=float, default=0.6)
    parser.add_argument("--sentences", type=int, default=300)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with open(os.path.join(ROOT, "Data", "holdout", f"holdout_{args.source}.txt"), encoding="utf-8") as f:
        documents = [d for d in f.read().split("\n\n") if d.strip()]
    documents.sort(key=len, reverse=True)
    doc = documents[args.rank]
    cut = int(len(doc) * args.split)
    while cut < len(doc) and doc[cut] != "\n":
        cut += 1
    train, later = doc[:cut], doc[cut:]
    db = sqlite3.connect(os.path.join(ROOT, "Data", "lexicon", "lexicon.sqlite"))
    syllables = {s for (s,) in db.execute("SELECT s FROM syllables")}
    db.close()
    rng = random.Random(9)
    matches = [m for m in HAN_RUN.finditer(later) if 5 <= len(m.group()) <= 18 and m.start() >= 4]
    rng.shuffle(matches)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    written = 0
    with open(args.out + ".eval.tsv", "w", encoding="utf-8") as out:
        for m in sorted(matches[: args.sentences * 2], key=lambda m: m.start()):
            sentence = m.group()
            pinyin = lazy_pinyin(sentence)
            if len(pinyin) != len(sentence) or any(s not in syllables for s in pinyin):
                continue
            context = later[max(0, m.start() - CONTEXT_CHARS):m.start()].replace("\t", " ").replace("\\", "/").replace("\n", "\\n")
            out.write(" ".join(pinyin) + "\t" + sentence + "\t" + context + "\n")
            written += 1
            if written == args.sentences:
                break
    with open(args.out + ".train.txt", "w", encoding="utf-8") as f:
        f.write(train)
    print(f"{args.source} #{args.rank}: {len(doc):,} chars; train {len(train):,} chars; {written} later sentences -> {args.out}.*")


if __name__ == "__main__":
    main()
