#!/usr/bin/env python3
"""The primary test: typing one word with the text before it, from real dialogue.

    Data/venv-train/bin/python Tools/make_word_eval.py [--lexicon Data/lexicon/lexicon.sqlite] [--count 3000]

Held-out dialogue (never trained on) is cut into lexicon words (the most probable cutting under the
lexicon's rates); every word of 2-4 characters with at least 4 characters before it becomes a case:
"keys<TAB>word<TAB>preceding text (60 characters)". Three files, same words, three ways of typing:
  Data/eval_word_full.tsv      full pinyin            hennan  -> 很难
  Data/eval_word_mixed.tsv     last syllable's initial henn   -> 很难
  Data/eval_word_initials.tsv  initials only          hn      -> 很难
"""
import argparse
import collections
import os
import random
import re
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAN = re.compile(r"[一-鿿]+")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexicon", default=os.path.join(ROOT, "Data", "lexicon", "lexicon.sqlite"))
    parser.add_argument("--source", default=os.path.join(ROOT, "Data", "lm2", "holdout_dialogue.txt"))
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=21)
    args = parser.parse_args()
    db = sqlite3.connect(args.lexicon)
    key, logp = {}, {}
    for word, k, lp in db.execute("SELECT word, key, logp FROM lexicon WHERE n BETWEEN 1 AND 4"):
        if HAN.fullmatch(word) and (word not in logp or lp > logp[word]):
            key[word], logp[word] = k, lp
    docs = [d for d in open(args.source, encoding="utf-8").read().split("\n\n") if d.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(docs)

    def cut(run):
        n = len(run)
        best = [-1e18] * (n + 1)
        back = [0] * (n + 1)
        best[0] = 0.0
        for i in range(n):
            if best[i] < -1e17:
                continue
            for j in range(i + 1, min(n, i + 4) + 1):
                w = run[i:j]
                if w in logp:
                    score = best[i] + logp[w]
                    if score > best[j]:
                        best[j], back[j] = score, i
            if best[i + 1] < -1e17:
                best[i + 1], back[i + 1] = best[i] - 30, i
        spans, j = [], n
        while j > 0:
            spans.append((back[j], j))
            j = back[j]
        return reversed(spans)

    rows = []
    for doc in docs:
        for m in HAN.finditer(doc):
            run = m.group()
            if len(run) < 2 or m.start() < 4:
                continue
            for i, j in cut(run):
                w = run[i:j]
                if len(w) < 2 or w not in key:
                    continue
                syllables = key[w].split("'")
                context = doc[max(0, m.start() + i - 60):m.start() + i].replace("\t", " ").replace("\\", "/").replace("\n", "\\n")
                if not HAN.search(context):
                    continue
                rows.append((syllables, w, context))
        if len(rows) > 40000:
            break
    rng.shuffle(rows)
    sample = rows[: args.count]
    files = {
        "full": lambda s: "".join(s),
        "mixed": lambda s: "".join(s[:-1]) + s[-1][0],
        "initials": lambda s: "".join(x[0] for x in s),
    }
    for name, make in files.items():
        with open(os.path.join(ROOT, "Data", f"eval_word_{name}.tsv"), "w", encoding="utf-8") as f:
            for syllables, w, context in sample:
                f.write(f"{make(syllables)}\t{w}\t{context}\n")
    print(f"{len(rows):,} word occurrences; wrote {len(sample)} cases × 3 files; lengths {dict(collections.Counter(len(w) for _, w, _ in sample))}")


if __name__ == "__main__":
    main()
