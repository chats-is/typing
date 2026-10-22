#!/usr/bin/env python3
"""Turn the collected corpora into token files for language model training.

    Data/venv-train/bin/python Training/prepare_data.py [--out Data/prepared]

For every source in sources.py this writes <out>/<source>.bin (flat little-endian uint16 token
ids) and <out>/holdout_<source>.txt (documents kept out of training, separated by a blank line,
for evaluation). It also writes <out>/vocab.json (a JSON list, index = token id), <out>/val.bin
(the held-out documents of all sources, tokenised) and <out>/sources.json (name -> sampling weight).

Tokens are single characters: Chinese by character, everything else lowercased character by
character. Documents written mostly in traditional characters are dropped. Every document is cleaned
(Training/cleaning.py: nothing identifying, nothing one does not type), the held-out ones too, so
that validation and the test sets made from them measure the text the model is trained for; which
documents are held out does not change.

--cased-from Data/lm9 keeps the case of English letters instead ("iPhone", "CEO"): it reuses that
directory's vocabulary, so a model trained on it can be trained further, and adds the 26
capital letters at the end. Other capitals (Greek, Cyrillic) still read as their small letters.
The documents and the held-out split are the same as without it.

--add web,techdocs only writes those sources (and appends their held-out documents to val.bin),
keeping everything else in --out; the vocabulary in --out is reused.

--clean-holdouts cleans the holdout_<source>.txt files already in --out (written by an earlier
version that kept them as they were) and writes val.bin again from them, with the vocabulary in
--out; the training files are left alone.
"""
import argparse
import json
import os
import re

import numpy as np

from cleaning import clean
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
    with open(os.path.join(ROOT, "Data", "lexicon", "readings.tsv"), encoding="utf-8") as f:
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


def token_table(vocab):
    """Code point -> token id for a vocabulary."""
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
    return table


def encoder(table):
    newline = table[ord("\n")]
    return lambda points: np.concatenate(([BOS], table[points], [newline])).astype("<u2")


def clean_holdouts(out):
    """Cleans the held-out documents an earlier version kept as they were, and val.bin with them."""
    global KEEP_CASE
    with open(os.path.join(out, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    KEEP_CASE = "A" in vocab
    encode = encoder(token_table(vocab))
    names = [name for name in SOURCES if os.path.exists(os.path.join(out, f"holdout_{name}.txt"))]
    with open(os.path.join(out, "val.bin"), "wb") as val:
        for name in names:
            path = os.path.join(out, f"holdout_{name}.txt")
            with open(path, encoding="utf-8") as f:
                documents = [doc for doc in f.read().split("\n\n") if doc.strip()]
            kept, changed = [], 0
            for doc in documents:
                cleaned = clean(doc)
                changed += cleaned != doc
                if cleaned is not None and len(cleaned) >= 5:
                    kept.append(cleaned)
                    val.write(encode(codepoints(cleaned)).tobytes())
            with open(path, "w", encoding="utf-8") as f:
                f.write("".join(doc + "\n\n" for doc in kept))
            print(f"holdout_{name}.txt: {len(kept)} of {len(documents)} documents kept, {changed} changed by cleaning", flush=True)


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
    parser.add_argument("--clean-holdouts", action="store_true", help="clean the held-out files in --out and rewrite val.bin")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    global KEEP_CASE
    KEEP_CASE = args.cased_from is not None
    if args.clean_holdouts:
        clean_holdouts(args.out)
        return

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

    table = token_table(vocab)
    encode = encoder(table)

    with open(os.path.join(args.out, "val.bin"), "ab" if args.only_words or args.add else "wb") as val:
        for name in names:
            total = kept = dropped = held = 0
            before_chars = after_chars = 0
            with open(os.path.join(args.out, f"{name}.bin"), "wb") as train, \
                    open(os.path.join(args.out, f"holdout_{name}.txt"), "w", encoding="utf-8") as holdout:
                for index, points in enumerate(documents(name, known), 1):
                    if index % HOLDOUT_EVERY == 0 and held < MAX_HOLDOUT_DOCS:
                        held += 1
                        cleaned = clean(points.astype("<u4").tobytes().decode("utf-32-le"))
                        if cleaned is not None and len(cleaned) >= 5:
                            val.write(encode(codepoints(cleaned)).tobytes())
                            holdout.write(cleaned + "\n\n")
                        continue
                    text = points.astype("<u4").tobytes().decode("utf-32-le")
                    cleaned = clean(text)
                    before_chars += len(text)
                    if cleaned is None or len(cleaned) < 5:
                        dropped += 1
                        continue
                    after_chars += len(cleaned)
                    ids = encode(codepoints(cleaned))
                    train.write(ids.tobytes())
                    total += len(ids)
                    kept += 1
                    if kept % 1_000_000 == 0:
                        print(f"  {name}: {kept} documents, {total / 1e6:.0f}M tokens", flush=True)
            print(f"{name}: {kept} documents, {total / 1e6:.0f}M tokens, {held} held out; cleaning dropped {dropped} documents"
                  f" and {100 * (1 - after_chars / max(1, before_chars)):.2f}% of the characters", flush=True)

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
