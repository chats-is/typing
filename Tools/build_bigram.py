#!/usr/bin/env python3
"""Count word bigrams over the everyday corpora and write Data/lexicon/bigram.bin.

    Data/venv-train/bin/python Tools/build_bigram.py

Han runs of the token streams (the same everyday mixture Training/count_words.py uses) are
segmented into lexicon words (unigram Viterbi under the lexicon's own frequencies) and the word
pairs counted. Word ids are the lexicon's wid, so the file belongs to one lexicon.sqlite build.

bigram.bin layout "BGM2" (little endian): uint32 rows (= largest w1 + 2), uint32 count, then
rows + 1 uint32 row starts, count uint32 second words (sorted within each row), count float16
log2 P(w2 | w1). w1 == 0 means sentence start. The counts of every pair are cached in
bigram_counts.npz so other --min-count values can be written without recounting
(--from-counts).
"""
import argparse
import json
import multiprocessing
import os
import re
import sqlite3
import struct
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Training"))
from count_words import SAMPLE, WEIGHTS  # noqa: E402

DATA = os.path.join(ROOT, "Data", "lm9")
LEXICON = os.path.join(ROOT, "Data", "lexicon", "lexicon.sqlite")
OUT = os.path.join(ROOT, "Data", "lexicon", "bigram.bin")
MAX_WORD_CHARS = 4
MIN_COUNT = 3
COUNTS_CACHE = os.path.join(ROOT, "Data", "lexicon", "bigram_counts.npz")
CHUNK = 20_000_000
HAN_RUN = re.compile(r"[一-鿿]+")

WORDS = {}
TEXT = None


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
        for j in range(i + 1, min(n, i + MAX_WORD_CHARS) + 1):
            entry = get(run[i:j])
            if entry is not None:
                score = base + entry[1]
                if score > best[j]:
                    best[j] = score
                    back[j] = (i, entry[0])
    if back[n] is None:
        return None
    ids = []
    j = n
    while j > 0:
        i, wid = back[j]
        ids.append(wid)
        j = i
    ids.reverse()
    return ids


def count_span(text):
    keys = []
    for run in HAN_RUN.findall(text):
        ids = segment(run)
        if ids is None:
            continue
        previous = 0
        for wid in ids:
            keys.append((previous << 32) | wid)
            previous = wid
        keys.append(previous << 32)  # run end, only for the P(w2 | w1) denominator
    return np.unique(np.array(keys, dtype=np.uint64), return_counts=True)


def merge(parts):
    keys = np.concatenate([p[0] for p in parts])
    counts = np.concatenate([p[1] for p in parts])
    unique, inverse = np.unique(keys, return_inverse=True)
    return unique, np.bincount(inverse, weights=counts, minlength=len(unique)).astype(np.int64)


def write_bgm2(keys, probabilities, out):
    """Compressed rows: one start offset per first word, second words and log2 probabilities per pair."""
    firsts = (keys >> np.uint64(32)).astype(np.int64)
    seconds = (keys & np.uint64(0xFFFFFFFF)).astype("<u4")
    rows = int(firsts.max()) + 2 if len(firsts) else 1
    starts = np.zeros(rows + 1, dtype="<u4")
    np.add.at(starts, firsts + 1, 1)
    starts = np.cumsum(starts).astype("<u4")
    values = np.log2(probabilities.astype(np.float64)).astype("<f2")
    with open(out, "wb") as f:
        f.write(b"BGM2")
        f.write(struct.pack("<II", rows, len(keys)))
        f.write(starts.tobytes())
        f.write(seconds.tobytes())
        f.write(values.tobytes())


def write(keys, counts, min_count, out):
    firsts = keys >> np.uint64(32)
    totals = np.bincount(firsts.astype(np.int64), weights=counts)  # over every pair, before pruning
    keep = (counts >= min_count) & ((keys & np.uint64(0xFFFFFFFF)) != 0)
    keys, counts, firsts = keys[keep], counts[keep], firsts[keep]
    probabilities = counts / totals[firsts.astype(np.int64)]
    write_bgm2(keys, probabilities, out)
    print(f"wrote {out}: {len(keys):,} bigrams (count >= {min_count}), {os.path.getsize(out) / 1e6:.0f} MB")


def main():
    global TEXT
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-count", type=int, default=MIN_COUNT)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--from-counts", action="store_true", help="reuse the cached counts instead of recounting")
    args = parser.parse_args()
    if args.from_counts:
        cached = np.load(COUNTS_CACHE)
        write(cached["keys"], cached["counts"], args.min_count, args.out)
        return
    started = time.time()
    db = sqlite3.connect(LEXICON)
    for word, wid, logp in db.execute("SELECT word, wid, MAX(logp) FROM lexicon GROUP BY word"):
        if len(word) <= MAX_WORD_CHARS and all("一" <= c <= "鿿" for c in word):
            WORDS[word] = (wid, logp)
    db.close()
    print(f"{len(WORDS):,} words ({time.time() - started:.0f}s)", flush=True)
    with open(os.path.join(DATA, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    chars = np.array([t if len(t) == 1 else "\n" for t in vocab])

    merged = None
    pending = []
    # One pool for the whole run, forked while the process is still small: the workers share the
    # word table and get their text as plain arguments.
    pool = multiprocessing.get_context("fork").Pool(8)
    for name in WEIGHTS:
        path = os.path.join(DATA, f"{name}.bin")
        if not os.path.exists(path):
            continue
        stream = np.memmap(path, dtype="<u2", mode="r")
        total = min(len(stream), SAMPLE.get(name, len(stream)))
        for start in range(0, total, CHUNK):
            TEXT = "".join(chars[np.asarray(stream[start : min(total, start + CHUNK)])])
            n = len(TEXT)
            step = max(1, n // 32)
            bounds = []
            b = 0
            while b < n:
                e = min(n, b + step)
                while e < n and HAN_RUN.match(TEXT[e]):  # do not cut a run
                    e += 1
                bounds.append((b, e))
                b = e
            pending += pool.map(count_span, [TEXT[b:e] for b, e in bounds])
            merged = merge(pending + ([merged] if merged else []))
            pending = []
        print(f"{name}: {total / 1e6:.0f}M characters, {len(merged[0]):,} distinct bigrams ({time.time() - started:.0f}s)", flush=True)
    pool.close()
    keys, counts = merged
    np.savez_compressed(COUNTS_CACHE, keys=keys.astype("<u8"), counts=counts.astype(np.uint32))
    print(f"cached {len(keys):,} pair counts in {COUNTS_CACHE} ({time.time() - started:.0f}s)")
    write(keys, counts, args.min_count, args.out)


if __name__ == "__main__":
    main()
