#!/usr/bin/env python3
"""Turn the collected corpora into token files for language model training.

    Data/venv-train/bin/python Training/prepare_data.py [--out Data/lm2]

For every source in sources.py this writes <out>/<source>.bin (flat little-endian uint16 token
ids) and <out>/holdout_<source>.txt (documents kept out of training, separated by a blank line,
for evaluation). It also writes <out>/vocab.json (a JSON list, index = token id), <out>/val.bin
(the held-out documents of all sources, tokenised) and <out>/sources.json (name -> sampling weight).

Tokens are single characters: Chinese by character, everything else lowercased character by
character. Documents written mostly in traditional characters are dropped.

--cased-from Data/lm2 keeps the case of English letters instead ("iPhone", "CEO"): it reuses that
directory's vocabulary, so a model trained on it can be trained further, and adds the 26
capital letters at the end. Other capitals (Greek, Cyrillic) still read as their small letters.
The documents and the held-out split are the same as without it.

--add web,techdocs only writes those sources (and appends their held-out documents to val.bin),
keeping everything else in --out; the vocabulary in --out is reused.
"""
import argparse
import json
import os
import re

import numpy as np

from sources import ROOT, SOURCES

SPECIAL = ["<pad>", "<bos>", "<unk>"]
PAD, BOS, UNK = 0, 1, 2
VOCAB_SAMPLE_DOCS = 150_000
MIN_COUNT = 30
MAX_VOCAB = 9000
HOLDOUT_EVERY = 300
MAX_HOLDOUT_DOCS = 3000
# A document with more than this share of Chinese characters outside the simplified set is dropped.
MAX_FOREIGN_SHARE = 0.1
WORDS_TOKENS = 40_000_000
ENGLISH_TOKENS = 10_000_000
SPACES = re.compile(r"[ \t　\xa0]+")
NEWLINES = re.compile(r"\s*\n\s*")


KEEP_CASE = False


def normalize(text):
    return NEWLINES.sub("\n", SPACES.sub(" ", text if KEEP_CASE else text.lower())).strip()


def codepoints(text):
    return np.frombuffer(text.encode("utf-32-le", "ignore"), dtype="<u4")


def simplified_characters():
    known = np.zeros(0x110000, dtype=bool)
    with open(os.path.join(ROOT, "Resources", "readings.tsv"), encoding="utf-8") as f:
        for line in f:
            known[ord(line.split("\t")[1])] = True
    return known


def is_simplified(points, known):
    han = points[(points >= 0x4E00) & (points <= 0x9FFF)]
    return len(han) == 0 or (~known[han]).mean() <= MAX_FOREIGN_SHARE


def documents(name, known):
    """Normalised documents of a source as code point arrays, traditional-script ones removed."""
    for doc in SOURCES[name][0]():
        points = codepoints(normalize(doc))
        if len(points) >= 5 and is_simplified(points, known):
            yield points


def build_vocab(names, known):
    counts = np.zeros(0x110000, dtype=np.int64)
    for name in names:
        batch, seen = [], 0
        for points in documents(name, known):
            batch.append(points)
            seen += 1
            if len(batch) == 2000 or seen == VOCAB_SAMPLE_DOCS:
                counts += np.bincount(np.concatenate(batch), minlength=0x110000)
                batch = []
            if seen == VOCAB_SAMPLE_DOCS:
                break
        if batch:
            counts += np.bincount(np.concatenate(batch), minlength=0x110000)
    # Printable ASCII is always kept so English and digits can be typed whatever the corpus holds,
    # and so is every character of the simplified set that occurs at all.
    forced = [ord("\n")] + [c for c in range(0x20, 0x7F) if not chr(c).isupper()]
    counts[forced] = 0
    typable = [int(c) for c in np.nonzero(known & (counts > 0))[0]]
    counts[typable] = 0
    room = MAX_VOCAB - len(SPECIAL) - len(forced) - len(typable)
    frequent = [int(c) for c in np.argsort(-counts)[:max(room, 0)] if counts[c] >= MIN_COUNT]
    return SPECIAL + [chr(c) for c in forced + typable + frequent]


def lexicon_words():
    """Every word of the merged lexicons (Tools/merge_lexicons.py) and a sampling distribution
    that is half by frequency and half uniform, so common words recur and rare names still appear."""
    words, weights = [], []
    seen = set()
    with open(os.path.join(ROOT, "Data", "lexicons", "all_words.tsv"), encoding="utf-8") as f:
        for line in f:
            word, _pinyin, weight, _count = line.rstrip("\n").split("\t")
            if len(word) >= 2 and word not in seen:
                seen.add(word)
                words.append(word)
                weights.append(float(weight))
    weights = np.array(weights)
    by_frequency = weights / weights.sum() if weights.sum() > 0 else np.full(len(words), 1 / len(words))
    return words, 0.5 * by_frequency + 0.5 / len(words)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(ROOT, "Data", "prepared"))
    parser.add_argument("--only-words", action="store_true", help="only rewrite words.bin with the existing vocabulary")
    parser.add_argument("--cased-from", default=None, help="keep English capitals, extending this directory's vocabulary")
    parser.add_argument("--add", default=None, help="comma-separated sources to write into an existing --out")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    global KEEP_CASE
    KEEP_CASE = args.cased_from is not None

    known = simplified_characters()
    names = [name for name, (reader, _) in SOURCES.items() if reader is not None]
    if args.only_words or args.add:
        with open(os.path.join(args.out, "vocab.json"), encoding="utf-8") as f:
            vocab = json.load(f)
        names = args.add.split(",") if args.add else []
    elif args.cased_from:
        with open(os.path.join(args.cased_from, "vocab.json"), encoding="utf-8") as f:
            vocab = json.load(f)
        vocab += [chr(c) for c in range(ord("A"), ord("Z") + 1) if chr(c) not in vocab]
        with open(os.path.join(args.out, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)
        print(f"vocab: {len(vocab)} tokens (extended from {args.cased_from})", flush=True)
    else:
        vocab = build_vocab(names, known)
        with open(os.path.join(args.out, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)
        print(f"vocab: {len(vocab)} tokens", flush=True)

    table = np.full(0x110000, UNK, dtype=np.uint16)
    for token_id, token in enumerate(vocab[len(SPECIAL):], len(SPECIAL)):
        table[ord(token)] = token_id
    if KEEP_CASE:
        # Only English letters keep their case; every other capital reads as its small letter.
        for point in range(0x80, 0x110000):
            if 0xD800 <= point <= 0xDFFF:
                continue
            small = chr(point).lower()
            if len(small) == 1 and small != chr(point):
                table[point] = table[ord(small)]
    newline = table[ord("\n")]

    def encode(points):
        return np.concatenate(([BOS], table[points], [newline])).astype("<u2")

    with open(os.path.join(args.out, "val.bin"), "ab" if args.only_words or args.add else "wb") as val:
        for name in names:
            total = kept = dropped = held = 0
            with open(os.path.join(args.out, f"{name}.bin"), "wb") as train, \
                    open(os.path.join(args.out, f"holdout_{name}.txt"), "w", encoding="utf-8") as holdout:
                for index, points in enumerate(documents(name, known), 1):
                    ids = encode(points)
                    if index % HOLDOUT_EVERY == 0 and held < MAX_HOLDOUT_DOCS:
                        held += 1
                        val.write(ids.tobytes())
                        holdout.write(points.astype("<u4").tobytes().decode("utf-32-le") + "\n\n")
                        continue
                    train.write(ids.tobytes())
                    total += len(ids)
                    kept += 1
                    if kept % 1_000_000 == 0:
                        print(f"  {name}: {kept} documents, {total / 1e6:.0f}M tokens", flush=True)
            print(f"{name}: {kept} documents, {total / 1e6:.0f}M tokens, {held} held out", flush=True)

    if args.add:
        with open(os.path.join(args.out, "sources.json"), "w", encoding="utf-8") as f:
            json.dump({name: weight for name, (_, weight) in SOURCES.items()}, f, indent=1)
        return

    # Lexicon entries as one-word lines, drawn by frequency: teaches which characters form words.
    words, probabilities = lexicon_words()
    rng = np.random.default_rng(0)
    written = 0
    with open(os.path.join(args.out, "words.bin"), "wb") as f:
        while written < WORDS_TOKENS:
            chosen = rng.choice(len(words), size=200_000, p=probabilities)
            ids = table[codepoints("\n".join(words[i] for i in chosen) + "\n")].astype("<u2")
            f.write(ids.tobytes())
            written += len(ids)
    print(f"words: {written / 1e6:.0f}M tokens from {len(words)} lexicon words", flush=True)

    # Common English words by frequency (the 50k list of hermitdave/FrequencyWords, OpenSubtitles),
    # so the model knows English spelling; dedicated English text comes with English input support.
    english, counts = [], []
    with open(os.path.join(ROOT, "Data", "lexicons", "english", "en_50k.txt"), encoding="utf-8") as f:
        for line in f:
            word, count = line.split()
            if word.isalpha() and word.isascii():
                english.append(word.lower())
                counts.append(float(count))
    probabilities = np.array(counts) / sum(counts)
    written = 0
    with open(os.path.join(args.out, "english.bin"), "wb") as f:
        while written < ENGLISH_TOKENS:
            chosen = rng.choice(len(english), size=100_000, p=probabilities)
            ids = table[codepoints(" ".join(english[i] for i in chosen) + "\n")].astype("<u2")
            f.write(ids.tobytes())
            written += len(ids)
    print(f"english: {written / 1e6:.0f}M tokens from {len(english)} words", flush=True)

    with open(os.path.join(args.out, "sources.json"), "w", encoding="utf-8") as f:
        json.dump({name: weight for name, (_, weight) in SOURCES.items()}, f, indent=1)


if __name__ == "__main__":
    main()
