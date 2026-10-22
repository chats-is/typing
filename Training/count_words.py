#!/usr/bin/env python3
"""Count how often the lexicon's words occur in everyday text, to rank homophones the way people type.

    Data/venv-train/bin/python Training/count_words.py --data Data/prepared-clean --out Data/lexicon/word_counts.tsv

Every word of Data/lexicons/all_words.tsv (listed by at least --sources lexicons, or a single
character) is counted as a substring of each corpus stream (token ids from prepare_data.py), per
source. The result has one line per word: the word, then per-source occurrences per million
characters, then a weighted "everyday" rate (--weights: how much each source stands for what
people type: chat and real input first, news last). Only used to build the lexicon and the tests.
"""
import argparse
import json
import os
import sys
import time

import multiprocessing
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_LEN = 4
# What people type into an input method is mostly chat; real input records are the best sample
# but small, so they get a large weight per character; news and encyclopedia the smallest.
WEIGHTS = {"touchpal": 0.25, "dialogue": 0.30, "forum": 0.15, "mixed": 0.10, "web": 0.10, "news": 0.05, "novels": 0.05}
SAMPLE = {"news": 100_000_000, "mixed": 150_000_000, "web": 150_000_000}  # tokens read from the big ones


def codes(ids, n):
    """One uint64 per position: the n token ids starting there (16 bits each)."""
    out = np.zeros(len(ids) - n + 1, dtype=np.uint64)
    for k in range(n):
        out |= ids[k : len(ids) - n + 1 + k].astype(np.uint64) << np.uint64(16 * (n - 1 - k))
    return out


SEG_TEXT = None  # the chunk being segmented, shared with forked workers
SEG_DICT = None


def segment_chunk(bounds):
    """Longest-match segmentation of SEG_TEXT[start:end] with SEG_DICT; counts of the segments."""
    start, end = bounds
    text, words = SEG_TEXT, SEG_DICT
    counts = Counter()
    i = start
    while i < end:
        for length in range(min(8, end - i), 1, -1):
            piece = text[i : i + length]
            if piece in words:
                counts[piece] += 1
                i += length
                break
        else:
            counts[text[i]] += 1
            i += 1
    return counts


def segmented(args):
    """Like main(), but a word is counted only where the longest-match segmentation puts it: 才是
    inside 他才是 is 才 and 是, not the "word" 才是, as nobody types it as one."""
    global SEG_TEXT, SEG_DICT
    with open(os.path.join(args.data, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    chars = np.array([t if len(t) == 1 else "\n" for t in vocab])
    words = set()
    with open(args.table, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and 2 <= len(parts[0]) <= 8 and int(parts[3]) >= args.segment:
                words.add(parts[0])
    SEG_DICT = words
    print(f"{len(words):,} words in the segmenting dictionary", flush=True)
    started = time.time()
    counts = {}
    total_chars = {}
    for name in WEIGHTS:
        path = os.path.join(args.data, f"{name}.bin")
        if not os.path.exists(path):
            continue
        stream = np.memmap(path, dtype="<u2", mode="r")
        total = min(len(stream), SAMPLE.get(name, len(stream)))
        count = Counter()
        for start in range(0, total, args.chunk):
            SEG_TEXT = "".join(chars[np.asarray(stream[start : min(total, start + args.chunk)])])
            n = len(SEG_TEXT)
            step = max(1, n // (args.workers * 4))
            bounds = [(b, min(n, b + step)) for b in range(0, n, step)]
            with multiprocessing.get_context("fork").Pool(args.workers) as pool:
                for part in pool.imap_unordered(segment_chunk, bounds):
                    count.update(part)
        counts[name] = count
        total_chars[name] = total
        print(f"{name}: {total / 1e6:.0f}M characters, {len(count):,} distinct segments ({time.time() - started:.0f}s)", flush=True)
    names = list(counts)
    weight_sum = sum(WEIGHTS[n] for n in names)
    everyday = Counter()
    for n in names:
        for w, c in counts[n].items():
            everyday[w] += c * 1e6 / total_chars[n] * WEIGHTS[n] / weight_sum
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("word\teveryday_per_million\t" + "\t".join(names) + "\n")
        for w, rate in everyday.most_common():
            if not w.strip() or "\n" in w or "\t" in w:
                continue
            f.write(f"{w}\t{rate:.3f}\t" + "\t".join(f"{counts[n].get(w, 0) * 1e6 / total_chars[n]:.3f}" for n in names) + "\n")
    print(f"wrote {args.out} ({time.time() - started:.0f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(ROOT, "Data", "prepared-clean"))
    parser.add_argument("--table", default=os.path.join(ROOT, "Data", "lexicons", "all_words.tsv"))
    parser.add_argument("--out", default=os.path.join(ROOT, "Data", "lexicon", "word_counts.tsv"))
    parser.add_argument("--sources", type=int, default=2, help="a word is counted if this many lexicons list it")
    parser.add_argument("--chunk", type=int, default=40_000_000)
    parser.add_argument("--segment", type=int, default=0, help="count words as segments of a longest-match segmentation instead of substrings; the segmenting dictionary is the words this many lexicons list")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.segment:
        return segmented(args)

    with open(os.path.join(args.data, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    ids = {t: i for i, t in enumerate(vocab) if len(t) == 1}

    started = time.time()
    words = []
    seen = set()
    with open(args.table, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            word, sources = parts[0], int(parts[3])
            if word in seen or not (1 <= len(word) <= MAX_LEN):
                continue
            if len(word) > 1 and sources < args.sources:
                continue
            if not all(c in ids for c in word):
                continue
            seen.add(word)
            words.append(word)
    print(f"{len(words):,} words to count ({time.time() - started:.0f}s)", flush=True)

    # Per length: sorted codes and the word index they belong to.
    by_len = {}
    for n in range(1, MAX_LEN + 1):
        group = [(w, i) for i, w in enumerate(words) if len(w) == n]
        code = np.array([sum(ids[c] << (16 * (n - 1 - k)) for k, c in enumerate(w)) for w, _ in group], dtype=np.uint64)
        order = np.argsort(code)
        by_len[n] = (code[order], np.array([i for _, i in group], dtype=np.int64)[order])

    counts = {}
    chars = {}
    for name in WEIGHTS:
        path = os.path.join(args.data, f"{name}.bin")
        if not os.path.exists(path):
            print(f"no {name}.bin, skipped", flush=True)
            continue
        stream = np.memmap(path, dtype="<u2", mode="r")
        total = min(len(stream), SAMPLE.get(name, len(stream)))
        count = np.zeros(len(words), dtype=np.int64)
        for start in range(0, total, args.chunk):
            chunk = np.array(stream[start : min(total, start + args.chunk + MAX_LEN - 1)])
            for n, (code, index) in by_len.items():
                if len(chunk) < n:
                    continue
                c = codes(chunk, n)
                pos = np.searchsorted(code, c)
                pos[pos == len(code)] = 0
                hit = code[pos] == c
                found, k = np.unique(pos[hit], return_counts=True)
                count[index[found]] += k
        counts[name] = count
        chars[name] = total
        print(f"{name}: {total / 1e6:.0f}M characters, {int(count.sum()):,} word occurrences ({time.time() - started:.0f}s)", flush=True)

    names = list(counts)
    weight_sum = sum(WEIGHTS[n] for n in names)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("word\teveryday_per_million\t" + "\t".join(names) + "\n")
        per_million = {n: counts[n] * 1e6 / chars[n] for n in names}
        everyday = sum(per_million[n] * WEIGHTS[n] for n in names) / weight_sum
        order = np.argsort(-everyday)
        for i in order:
            f.write(f"{words[i]}\t{everyday[i]:.3f}\t" + "\t".join(f"{per_million[n][i]:.3f}" for n in names) + "\n")
    print(f"wrote {args.out} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
