#!/usr/bin/env python3
"""Removes repeated text from a prepared directory (Training/prepare_data.py output), in place.

    Data/venv-train/bin/python Training/dedup.py --data Data/prepared-clean

- A document identical to an earlier one is dropped.
- A long line (MIN_LINE tokens or more) seen before is dropped, wherever it was seen: the
  boilerplate of web pages, reposted paragraphs, the earlier posts a forum reply quotes. Short
  lines stay however often they recur ("哈哈哈", "好的" are what people type).
- "Earlier" follows ORDER, the better sources first, so a paragraph found in the news and in a
  web copy stays in the news. The held-out documents come before everything: text that is in a
  test set is taken out of training.
- In the sources of documents (CAPPED), a short line (SHORT_LINE to MIN_LINE tokens) is kept at
  most SHORT_CAP times: the 7,408 "我也要打赏" of the forum, bylines such as "(小风)" in the news,
  "下载附件 保存到相册" on web pages. What people type stays often enough to be learned ("支持楼主").
  Chat sources keep every line: there, how often a phrase recurs is how often people type it.
A document left without text is dropped. Running it again changes nothing. Writes dedup.log.
"""
import argparse
import hashlib
import json
import os

import numpy as np

import prepare_data as P

MIN_LINE = 20
ORDER = ["peoples_daily", "news", "novels", "dialogue", "touchpal", "forum", "mixed", "web", "techdocs", "tatoeba"]
CHUNK_DOCS = 200_000
SHORT_LINE = 4
SHORT_CAP = 20
CAPPED = {"news", "forum", "mixed", "web", "techdocs"}


def digest(tokens):
    return int.from_bytes(hashlib.blake2b(tokens.tobytes(), digest_size=8).digest(), "little")


def doc_bounds(data, bos):
    starts = np.flatnonzero(data == bos)
    return starts, np.append(starts[1:], len(data))


def short_lines(doc, newline):
    """(start, end) of every line of a document with SHORT_LINE to MIN_LINE - 1 tokens."""
    cuts = np.flatnonzero(doc == newline)
    line_starts = np.concatenate(([1], cuts[:-1] + 1)) if len(cuts) else np.array([1])
    line_ends = cuts if len(cuts) else np.array([len(doc)])
    length = line_ends - line_starts
    keep = (length >= SHORT_LINE) & (length < MIN_LINE)
    return line_starts[keep], line_ends[keep]


def long_lines(data, start, end, newline):
    """(start, end) of every line of a document with MIN_LINE tokens or more, newline excluded."""
    doc = data[start:end]
    cuts = np.flatnonzero(doc == newline)
    line_starts = np.concatenate(([1], cuts[:-1] + 1)) if len(cuts) else np.array([1])
    line_ends = cuts if len(cuts) else np.array([len(doc)])
    keep = line_ends - line_starts >= MIN_LINE
    return line_starts[keep] + start, line_ends[keep] + start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    with open(os.path.join(args.data, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    P.KEEP_CASE = "A" in vocab
    encode = P.encoder(P.token_table(vocab))
    newline = vocab.index("\n")
    names = [n for n in ORDER if os.path.exists(os.path.join(args.data, f"{n}.bin"))]
    log = open(os.path.join(args.data, "dedup.log"), "w", encoding="utf-8")

    def say(text):
        print(text, flush=True)
        log.write(text + "\n")

    seen_docs, seen_lines = set(), set()
    # The held-out documents first.
    for name in names:
        path = os.path.join(args.data, f"holdout_{name}.txt")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for text in f.read().split("\n\n"):
                if not text.strip():
                    continue
                ids = encode(P.codepoints(text))
                seen_docs.add(digest(ids))
                for s, e in zip(*long_lines(ids, 0, len(ids), newline)):
                    seen_lines.add(digest(ids[s:e]))
    say(f"held out: {len(seen_docs)} documents, {len(seen_lines)} long lines")

    for name in names:
        path = os.path.join(args.data, f"{name}.bin")
        data = np.memmap(path, dtype="<u2", mode="r")
        starts, ends = doc_bounds(data, P.BOS)
        docs_dropped = lines_dropped = short_dropped = 0
        short_counts = {}
        with open(path + ".dedup", "wb") as out:
            for c in range(0, len(starts), CHUNK_DOCS):
                pieces = []
                for start, end in zip(starts[c:c + CHUNK_DOCS], ends[c:c + CHUNK_DOCS]):
                    doc = np.asarray(data[start:end])
                    key = digest(doc)
                    if key in seen_docs:
                        docs_dropped += 1
                        continue
                    seen_docs.add(key)
                    keep = np.ones(len(doc), dtype=bool)
                    for s, e in zip(*long_lines(data, start, end, newline)):
                        line = digest(np.asarray(data[s:e]))
                        if line in seen_lines:
                            keep[s - start:e - start + 1] = False  # the line and its newline
                            lines_dropped += 1
                        else:
                            seen_lines.add(line)
                    if name in CAPPED:
                        for s, e in zip(*short_lines(doc, newline)):
                            if not keep[s]:
                                continue
                            line = hash(doc[s:e].tobytes())
                            n = short_counts.get(line, 0) + 1
                            short_counts[line] = n
                            if n > SHORT_CAP:
                                keep[s:e + 1] = False
                                short_dropped += 1
                    kept = doc[keep]
                    if len(kept) <= 2 or not (kept[1:] != newline).any():
                        docs_dropped += 1  # nothing but its start and line breaks left
                        continue
                    if kept[-1] != newline:
                        kept = np.append(kept, np.uint16(newline))
                    pieces.append(kept)
                if pieces:
                    out.write(np.concatenate(pieces).astype("<u2").tobytes())
        after = os.path.getsize(path + ".dedup") // 2
        say(f"{name}: {len(starts)} documents, {docs_dropped} dropped; {lines_dropped} repeated long lines dropped;"
            f" {short_dropped} short lines over {SHORT_CAP} times dropped;"
            f" {len(data) / 1e6:.0f}M → {after / 1e6:.0f}M tokens ({100 * (1 - after / len(data)):.1f}% less)")
        del data
        os.replace(path + ".dedup", path)
    log.close()


if __name__ == "__main__":
    main()
