#!/usr/bin/env python3
"""Count how often every candidate word is typed as a unit in everyday text, by probabilistic
segmentation.

    Data/venv-train/bin/python Tools/count_words_viterbi.py [--start Resources/lexicon.sqlite] [--iterations 3] [--out Data/lexicon/word_counts_vit3.tsv]

Why: Training/count_words.py --segment 6 cut the text with a longest-match dictionary of words that
six or more lexicons list. Everyday words outside that dictionary (眼熟, 没啦, 有啊, 高了, 也成,
好啦 ...) were never counted at all and fell to the floor rate, so rare dictionary words with the
same pinyin (鼹鼠, 梅拉, 高乐, 叶城) outranked them. Here every candidate word (listed by at least
two lexicons, one to four characters) competes: each Han run is cut into the most probable
sequence of candidate words under the current rates (unigram Viterbi), the segments are counted,
the rates are re-estimated, and the pass is repeated. Nothing is hand-set: a word's rate is how
often the most probable cutting of real text uses it.

Output has the layout of word_counts_seg.tsv: word, everyday per million (source-weighted), then
per-source per million. Characters never covered by a candidate word count as single characters.
"""
import argparse
import json
import math
import multiprocessing
import os
import re
import sqlite3
import sys
import time
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Training"))
from count_words import SAMPLE, WEIGHTS  # noqa: E402

DATA = os.path.join(ROOT, "Data", "prepared-clean")
TABLE = os.path.join(ROOT, "Data", "lexicons", "all_words.tsv")
# The lexicon whose rates start the first pass: the shipped one, so an update starts from the last version.
START = os.path.join(ROOT, "Resources", "lexicon.sqlite")
MAX_WORD = 4
MIN_SOURCES = 2  # one- and two-character words
MIN_SOURCES_LONG = 3  # three- and four-character words: more lexicons must agree before a longer unit may absorb shorter ones
CHUNK = 20_000_000
HAN_RUN = re.compile(r"[一-鿿]+")
UNSEEN_PER_MILLION = 0.01  # the rate a candidate keeps when a pass never used it, so it can still win later

LOGP = {}  # word -> log probability used for cutting (set before the pool is forked / refreshed per pass)
PAIRS = set()  # two-character candidate words that may be credited from adjacent single characters


def segment(run):
    n = len(run)
    best = [-1e18] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    get = LOGP.get
    for i in range(n):
        base = best[i]
        for j in range(i + 1, min(n, i + MAX_WORD) + 1):
            lp = get(run[i:j])
            if lp is not None:
                score = base + lp
                if score > best[j]:
                    best[j] = score
                    back[j] = i
        if best[i + 1] < -1e17:  # a character no candidate covers
            best[i + 1] = base - 20.0
            back[i + 1] = i
    out = []
    j = n
    while j > 0:
        i = back[j]
        out.append(run[i:j])
        j = i
    out.reverse()  # pieces in text order (pairs depend on it)
    return out


def count_text(text):
    """Segment counts, plus for every two adjacent single-character segments whose concatenation is a
    candidate word, one "pair" occurrence of that word (keyed "\x01" + word): 很难, 都是, 有啊 are cut
    into frequent single characters by the most probable segmentation and would otherwise never be
    counted, although they are typed as one unit."""
    counts = Counter()
    pairs = PAIRS
    for run in HAN_RUN.findall(text):
        pieces = segment(run)
        counts.update(pieces)
        for a, b in zip(pieces, pieces[1:]):
            if len(a) == 1 and len(b) == 1:
                ab = a + b
                if ab in pairs:
                    counts["\x01" + ab] += 1
    return counts


def set_logp(table):
    global LOGP, PAIRS
    LOGP = table
    PAIRS = {w for w in table if len(w) == 2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)  # the shipped lexicon used 3 passes
    parser.add_argument("--out", default=os.path.join(ROOT, "Data", "lexicon", "word_counts_vit3.tsv"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--start", default=START, help="lexicon.sqlite whose rates start the first pass")
    args = parser.parse_args()
    started = time.time()

    with open(os.path.join(DATA, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    chars = np.array([t if len(t) == 1 else "\n" for t in vocab])

    # Candidates: words at least MIN_SOURCES lexicons list, plus every Han character of the vocabulary.
    candidates = set()
    with open(TABLE, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and 1 <= len(parts[0]) <= MAX_WORD and int(parts[3]) >= (MIN_SOURCES if len(parts[0]) <= 2 else MIN_SOURCES_LONG) and HAN_RUN.fullmatch(parts[0]):
                candidates.add(parts[0])
    for token in vocab:
        if len(token) == 1 and "一" <= token <= "鿿":
            candidates.add(token)
    # Starting rates: the given lexicon's, so the first pass already cuts sensibly.
    logp = {}
    if os.path.exists(args.start):
        db = sqlite3.connect(f"file:{args.start}?mode=ro", uri=True)
        for word, lp in db.execute("SELECT word, MAX(logp) FROM lexicon GROUP BY word"):
            if word in candidates:
                logp[word] = lp
        db.close()
    # Candidates the lexicon never rated start at the unseen rate, not at the lexicon's floor.
    unseen0 = math.log(UNSEEN_PER_MILLION / 1e6)
    for word in candidates:
        logp.setdefault(word, unseen0)
    floor = unseen0
    print(f"{len(candidates):,} candidate words, {len(logp) - sum(1 for w in candidates if logp[w] == floor):,} with a starting rate ({time.time() - started:.0f}s)", flush=True)

    names = [n for n in WEIGHTS if os.path.exists(os.path.join(DATA, f"{n}.bin"))]
    weight_sum = sum(WEIGHTS[n] for n in names)
    set_logp(logp)
    pool = multiprocessing.get_context("fork").Pool(args.workers)
    try:
        for iteration in range(1, args.iterations + 1):
            counts = {}
            total_chars = {}
            for name in names:
                stream = np.memmap(os.path.join(DATA, f"{name}.bin"), dtype="<u2", mode="r")
                total = min(len(stream), SAMPLE.get(name, len(stream)))
                count = Counter()
                for start in range(0, total, CHUNK):
                    text = "".join(chars[np.asarray(stream[start : min(total, start + CHUNK)])])
                    n = len(text)
                    step = max(1, n // (args.workers * 4))
                    pieces = []
                    b = 0
                    while b < n:
                        e = min(n, b + step)
                        while e < n and HAN_RUN.match(text[e]) and HAN_RUN.match(text[e - 1]):  # do not cut a run
                            e += 1
                        pieces.append(text[b:e])
                        b = e
                    for part in pool.imap_unordered(count_text, pieces):
                        count.update(part)
                counts[name] = count
                total_chars[name] = total
                print(f"pass {iteration} {name}: {total / 1e6:.0f}M characters, {len(count):,} distinct segments ({time.time() - started:.0f}s)", flush=True)
            # Fold the pair occurrences into the word's own count.
            for n in names:
                for key in [k for k in counts[n] if k.startswith("\x01")]:
                    counts[n][key[1:]] += counts[n].pop(key)
            everyday = Counter()
            for n in names:
                for w, c in counts[n].items():
                    everyday[w] += c * 1e6 / total_chars[n] * WEIGHTS[n] / weight_sum
            # Rates for the next pass: per-million → log probability per word token.
            unseen = math.log(UNSEEN_PER_MILLION / 1e6)
            new_logp = {w: unseen for w in candidates}
            for w, rate in everyday.items():
                if w in candidates:
                    new_logp[w] = math.log(max(rate, UNSEEN_PER_MILLION) / 1e6)
            changed = sum(1 for w in candidates if abs(new_logp[w] - logp[w]) > 0.5)
            print(f"pass {iteration}: {len(everyday):,} words seen, {changed:,} candidates moved by more than 0.5 nats ({time.time() - started:.0f}s)", flush=True)
            logp = new_logp
            # Workers were forked with the old table: refresh it in every worker.
            pool.close()
            pool.join()
            set_logp(logp)
            pool = multiprocessing.get_context("fork").Pool(args.workers)
            with open(args.out, "w", encoding="utf-8") as f:
                f.write("word\teveryday_per_million\t" + "\t".join(names) + "\n")
                for w, rate in everyday.most_common():
                    if not w.strip():
                        continue
                    f.write(f"{w}\t{rate:.3f}\t" + "\t".join(f"{counts[n].get(w, 0) * 1e6 / total_chars[n]:.3f}" for n in names) + "\n")
            print(f"wrote {args.out} after pass {iteration} ({time.time() - started:.0f}s)", flush=True)
    finally:
        pool.close()


if __name__ == "__main__":
    main()
