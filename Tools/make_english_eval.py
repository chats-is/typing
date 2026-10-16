#!/usr/bin/env python3
"""Write Data/eval_english_mix.tsv: Chinese text with an English word in it, typed in one go.

    Data/venv-train/bin/python Tools/make_english_eval.py [--data Data/lm2]
    Data/venv-train/bin/python Tools/make_english_eval.py --data Data/lm3 --cased

The sentences come from the held-out text of prepare_data.py (news, mixed, forum, dialogue),
which the model never trains on. Two patterns, in the fourth column:
  sentence  Chinese, the English word, more Chinese, typed as one buffer ("zhegebugxiuhaole")
  word      the English word on its own, with all the text before it as context ("bug")
The English word must be in the English word list the model trained on, and no word is used
more than a few times, so a handful of frequent acronyms does not decide the score.
Held-out text from Data/lm2 is lowercase. With --cased (on the held-out text of a directory written
by prepare_data.py --cased-from) the expected English keeps its case ("iPhone", "CEO") while the
typed letters stay lowercase, and the file is Data/eval_english_cased.tsv. The same spans are chosen.
"""
import argparse
import os
import random
import re

from pypinyin import lazy_pinyin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = ("news", "mixed", "forum", "dialogue")
SPAN = re.compile(r"([一-鿿]{2,10})([A-Za-z]{2,12})([一-鿿]{1,8})")
PER_PATTERN = 300
PER_WORD = 3
CONTEXT_CHARS = 60
MIN_CONTEXT = 4
# Placeholders and slang initials that are not English words even when the word list has them.
NOT_WORDS = {"xx", "xxx", "yy", "pp", "sb", "bb", "jb", "ooxx", "lz", "jy", "tm", "nb", "ss", "ff", "zf"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(ROOT, "Data", "holdout"))
    parser.add_argument("--cased", action="store_true")
    args = parser.parse_args()
    with open(os.path.join(ROOT, "Resources", "readings.tsv"), encoding="utf-8") as f:
        syllables = {line.split("\t")[0] for line in f}
    with open(os.path.join(ROOT, "Data", "lexicons", "english", "en_50k.txt"), encoding="utf-8") as f:
        english = {line.split(" ")[0].lower() for line in f}

    def pinyin(text):
        result = lazy_pinyin(text)
        return result if len(result) == len(text) and all(s in syllables for s in result) else None

    found = []
    for source in SOURCES:
        with open(os.path.join(args.data, f"holdout_{source}.txt"), encoding="utf-8") as f:
            documents = [doc for doc in f.read().split("\n\n") if doc.strip()]
        for doc in documents:
            for match in SPAN.finditer(doc):
                before, written, after = match.group(1), match.group(2), match.group(3)
                word = written.lower()
                if word in NOT_WORDS or word not in english or match.start() < MIN_CONTEXT:
                    continue
                if pinyin(before) is None or pinyin(after) is None:
                    continue
                context = doc[max(0, match.start() - CONTEXT_CHARS):match.start()]
                found.append((source, before, written if args.cased else word, after, context))

    rng = random.Random(31)
    rng.shuffle(found)
    uses = {}
    lines = {"sentence": [], "word": []}
    for source, before, written, after, context in found:
        word = written.lower()
        if uses.get(word, 0) >= PER_WORD:
            continue
        pattern = "sentence" if len(lines["sentence"]) <= len(lines["word"]) else "word"
        if len(lines[pattern]) >= PER_PATTERN:
            continue
        uses[word] = uses.get(word, 0) + 1
        if pattern == "sentence":
            typed, expected = " ".join(pinyin(before) + [word] + pinyin(after)), before + written + after
        else:
            typed, expected, context = word, written, context + before
        context = context[-CONTEXT_CHARS:].replace("\t", " ").replace("\\", "/").replace("\n", "\\n")
        lines[pattern].append(f"{typed}\t{expected}\t{context}\t{pattern}")
    name = "eval_english_cased.tsv" if args.cased else "eval_english_mix.tsv"
    with open(os.path.join(ROOT, "Data", name), "w", encoding="utf-8") as out:
        out.write("\n".join(lines["sentence"] + lines["word"]) + "\n")
    print(f"wrote Data/{name}: {len(lines['sentence'])} sentences, {len(lines['word'])} words,"
          f" {len(uses)} distinct English words, from {len(found)} spans")


if __name__ == "__main__":
    main()
