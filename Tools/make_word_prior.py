#!/usr/bin/env python3
"""Write Resources/word_prior.bin: how common each word of 2-4 characters is, for the decoder.

    Data/venv-train/bin/python Tools/make_word_prior.py

The language model knows common words well but rarer ones (词频, 内卷) only faintly, and prefers
character runs that are no word at all (此品, 词拼). The decoder adds a bonus to readings that end
in a known word. The words and their log-probabilities come from the bundled lexicon
(Resources/lexicon.sqlite, built from rime-ice); a word whose characters are not all in the
model's vocabulary is left out.

Format (little-endian): UInt32 count, then count entries of UInt64 key and Float32 log-probability.
A key packs the word's length (top byte) and its token ids, 14 bits each, first character highest:
  key = length << 56 | t0 << 42 | t1 << 28 | t2 << 14 | t3   (unused slots 0, tokens right-aligned)
"""
import json
import os
import sqlite3
import struct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOOR = -20.0  # rarer words get no bonus anyway


def pack(tokens):
    key = len(tokens) << 56
    for i, token in enumerate(reversed(tokens)):
        key |= token << (14 * i)
    return key


def main():
    with open(os.path.join(ROOT, "Resources", "vocab.json"), encoding="utf-8") as f:
        vocab = {token: index for index, token in enumerate(json.load(f))}
    assert len(vocab) < 1 << 14
    best = {}
    db = sqlite3.connect(os.path.join(ROOT, "Resources", "lexicon.sqlite"))
    for word, logp in db.execute("SELECT word, MAX(logp) FROM lexicon WHERE n BETWEEN 2 AND 4 GROUP BY word"):
        if logp <= FLOOR:
            continue
        tokens = [vocab.get(c) for c in word]
        if len(tokens) < 2 or len(tokens) > 4 or None in tokens:
            continue
        best[pack(tokens)] = logp
    with open(os.path.join(ROOT, "Resources", "word_prior.bin"), "wb") as out:
        out.write(struct.pack("<I", len(best)))
        for key, logp in best.items():
            out.write(struct.pack("<Qf", key, logp))
    print(f"wrote Resources/word_prior.bin: {len(best)} words")


if __name__ == "__main__":
    main()
