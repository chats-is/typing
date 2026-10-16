#!/usr/bin/env python3
"""Write Resources/english_common.txt: the English words the input method recognises as such.

    Data/venv-train/bin/python Tools/make_english_words.py

When the typed letters are both pinyin and one of these words ("women", "man", "he"), the
English word is offered right after the best Chinese reading. Pinyin that word-frequency lists
pick up from subtitles and names ("wo", "ni", "shi", "dang") must not count as English, so the
words come from ECDICT (Data/corpus3/ecdict/ecdict.csv): those tagged with an English exam level
(zk, gk, cet4, cet6, ky, toefl, ielts, gre), plus their inflected forms ("women" from "woman").
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plain(word):
    return len(word) >= 2 and word.isascii() and word.isalpha() and word.islower()


def main():
    words = set()
    with open(os.path.join(ROOT, "Data", "corpus3", "ecdict", "ecdict.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["tag"].strip() or not plain(row["word"]):
                continue
            words.add(row["word"])
            # "p:went/d:gone/i:going/3:goes", "s:women"; "0:" and "1:" name the lemma, not a form.
            for item in row["exchange"].split("/"):
                kind, _, form = item.partition(":")
                if kind not in ("0", "1") and plain(form):
                    words.add(form)
    with open(os.path.join(ROOT, "Resources", "english_common.txt"), "w", encoding="utf-8") as out:
        out.write("\n".join(sorted(words)) + "\n")
    print(f"wrote Resources/english_common.txt: {len(words)} words")


if __name__ == "__main__":
    main()
