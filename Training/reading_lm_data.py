#!/usr/bin/env python3
"""Training data for the reading-conditioned model (reading_lm.py): everyday text with the
reading of every character.

    Data/venv-train/bin/python Training/reading_lm_data.py --out Data/rlm

Han runs of the everyday token streams are segmented into lexicon words (unigram Viterbi under the
lexicon's frequencies), so each character gets the reading its word has in the lexicon (长 is
zhang in 长大, chang in 长城); characters outside any word get pypinyin's reading. Writes, per
source, two parallel uint16 streams: <name>.bin (language-model token ids, the same vocabulary as
Data/lm9) and <name>.syl (syllable ids: 0 none - not a Han character or unknown, 1.. the lexicon's
syllables as listed in syllables.json).
"""
import argparse
import json
import multiprocessing
import os
import re
import sqlite3
import sys
import time

import numpy as np
from pypinyin import lazy_pinyin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data", "lm9")
LEXICON = os.path.join(ROOT, "Data", "lexicon", "lexicon.sqlite")
SAMPLE = {"dialogue": 120_000_000, "forum": 80_000_000, "touchpal": None, "mixed": 50_000_000, "web": 50_000_000, "news": 20_000_000, "novels": None}
HAN_RUN = re.compile(r"[一-鿿]+")
MAX_WORD = 4

WORDS = {}  # word -> (logp, [syllable ids])
CHAR_READING = {}  # character -> syllable id of its usual reading
SYLLABLE_ID = {}


def segment(run):
    n = len(run)
    best = [-1e18] * (n + 1)
    back = [None] * (n + 1)
    best[0] = 0.0
    get = WORDS.get
    for i in range(n):
        base = best[i]
        if base < -1e17:
            continue
        for j in range(i + 1, min(n, i + MAX_WORD) + 1):
            entry = get(run[i:j])
            if entry is not None:
                score = base + entry[0]
                if score > best[j]:
                    best[j] = score
                    back[j] = (i, entry[1])
        if best[i + 1] < -1e17:  # a character no word covers: its usual reading, a step of its own
            best[i + 1] = base - 30.0
            back[i + 1] = (i, [CHAR_READING.get(run[i], 0)])
    out = []
    j = n
    while j > 0:
        i, syllables = back[j]
        out = list(syllables) + out
        j = i
    return out


def readings(text):
    """Syllable id for every character of `text` (0 where not Han)."""
    out = [0] * len(text)
    for match in HAN_RUN.finditer(text):
        ids = segment(match.group())
        out[match.start() : match.end()] = ids
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    started = time.time()
    with open(os.path.join(DATA, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    chars = np.array([t if len(t) == 1 else "\n" for t in vocab])
    db = sqlite3.connect(LEXICON)
    syllables = [s for (s,) in db.execute("SELECT s FROM syllables ORDER BY s")]
    for index, s in enumerate(syllables):
        SYLLABLE_ID[s] = index + 1
    for word, key, logp in db.execute("SELECT word, key, MAX(logp) FROM lexicon GROUP BY word"):
        if len(word) <= MAX_WORD and all("一" <= c <= "鿿" for c in word):
            ids = [SYLLABLE_ID.get(s, 0) for s in key.split("'")]
            if len(ids) == len(word):
                WORDS[word] = (logp, ids)
    for word, key in db.execute("SELECT word, key FROM lexicon WHERE n = 1 ORDER BY logp"):
        CHAR_READING[word] = SYLLABLE_ID.get(key, 0)  # ascending logp: the best reading is written last
    db.close()
    for token in vocab:
        if len(token) == 1 and "一" <= token <= "鿿" and token not in CHAR_READING:
            CHAR_READING[token] = SYLLABLE_ID.get(lazy_pinyin(token)[0], 0)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "syllables.json"), "w", encoding="utf-8") as f:
        json.dump(syllables, f)
    with open(os.path.join(args.out, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"{len(WORDS):,} words, {len(syllables)} syllables ({time.time() - started:.0f}s)", flush=True)

    pool = multiprocessing.get_context("fork").Pool(args.workers)
    for name, limit in SAMPLE.items():
        path = os.path.join(DATA, f"{name}.bin")
        if not os.path.exists(path):
            continue
        stream = np.memmap(path, dtype="<u2", mode="r")
        total = min(len(stream), limit or len(stream))
        with open(os.path.join(args.out, f"{name}.bin"), "wb") as tokens_out, open(os.path.join(args.out, f"{name}.syl"), "wb") as syl_out:
            for start in range(0, total, 20_000_000):
                ids = np.array(stream[start : min(total, start + 20_000_000)])
                text = "".join(chars[ids])
                step = max(1, len(text) // 64)
                pieces = []
                b = 0
                while b < len(text):
                    e = min(len(text), b + step)
                    while e < len(text) and HAN_RUN.match(text[e]) and HAN_RUN.match(text[e - 1]):
                        e += 1
                    pieces.append(text[b:e])
                    b = e
                syl = np.concatenate([np.array(part, dtype="<u2") for part in pool.map(readings, pieces)])
                assert len(syl) == len(ids)
                tokens_out.write(ids.astype("<u2").tobytes())
                syl_out.write(syl.tobytes())
        print(f"{name}: {total / 1e6:.0f}M tokens ({time.time() - started:.0f}s)", flush=True)
    pool.close()


if __name__ == "__main__":
    main()
