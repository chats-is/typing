#!/usr/bin/env python3
"""Turn text into keys-to-text training examples, so one model learns the whole input method.

    Data/venv-train/bin/python Training/make_ime_data.py --samples 30          # show examples
    Data/venv-train/bin/python Training/make_ime_data.py --out Data/ime1 --rows 3000000

The model reads the text before the cursor, then each typed key chunk followed by the text it
stands for:

    今天去商场 ▶ wo我 xin新 mai买 le了 yi一 ge个 iI pP hh oo nn ee

(the interleaved layout), or, with --layout keys-first, all the keys first and then the text in
one piece, so the text reads like plain text and every character is chosen knowing all the keys:

    今天去商场 ▶ wxmlygiphone ⇒ 我新买了一个 iPhone ■

so it learns which characters a pinyin syllable can be (no reading table), which letter runs are
English and how they are written ("iPhone"), and how people really type: slips of the finger
("mia" for 买), initials ("w" for 我), an unfinished last syllable ("zhongg"). The decoder then
only has to keep the key chunks equal to what was typed.

Pinyin comes from pypinyin, here only, when the examples are made; the input method never sees it.

--out writes a training directory for train.py: vocab.json (the lm3 vocabulary plus a token per
key, "⌨a" ... "⌨z" and "⌨'", and "<type>", which ends the text before the cursor), ime.rows and
val.rows (rows of ROW tokens, each holding examples one after another: <bos>, text before the
cursor, <type>, key and text tokens; the rest of a row is <pad>), and sources.json, which also mixes in plain text from Data/lm3 so the
model keeps reading text well.
"""
import argparse
import json
import multiprocessing
import os
import random
import re

import numpy as np
from pypinyin import lazy_pinyin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "Data", "lm3")
SOURCES = {"dialogue": 0.30, "mixed": 0.20, "forum": 0.12, "news": 0.10, "web": 0.18, "novels": 0.04, "techdocs": 0.03, "touchpal": 0.03}

CONTEXT_CHARS = 20  # --context
MAX_UNITS = 10
ROW = 65  # block_size + 1: inputs and shifted targets (--block)
KEYS = "abcdefghijklmnopqrstuvwxyz'"
TYPE = "<type>"
OUT = "<out>"  # keys-first layout: the keys end, the text starts
END = "<end>"  # keys-first layout: the text ends
LAYOUT = "interleaved"  # --layout
PLAIN = {"dialogue": 0.06, "web": 0.06, "news": 0.03, "forum": 0.03, "mixed": 0.02}
INITIALS = ("zh", "ch", "sh")
# Neighbouring keys on a QWERTY keyboard, for slips of the finger.
ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
NEIGHBOURS = {}
for r, row in enumerate(ROWS):
    for c, key in enumerate(row):
        near = set()
        for dr in (-1, 0, 1):
            if 0 <= r + dr < len(ROWS):
                other = ROWS[r + dr]
                for dc in (-1, 0, 1):
                    if 0 <= c + dc < len(other) and (dr, dc) != (0, 0):
                        near.add(other[c + dc])
        NEIGHBOURS[key] = sorted(near)

HAN = re.compile(r"[一-鿿]")
# A stretch to type: Chinese characters and English letters, with spaces between English words.
TYPED = re.compile(r"[一-鿿A-Za-z](?:[一-鿿A-Za-z]| (?=[A-Za-z]))*")


def initial(syllable):
    for i in INITIALS:
        if syllable.startswith(i):
            return i
    return syllable[0]


def slip(keys, rng):
    """One slip of the finger in a syllable: a neighbouring key, a key missed, doubled or swapped."""
    if not keys:
        return keys
    i = rng.randrange(len(keys))
    kind = rng.random()
    if kind < 0.4:
        return keys[:i] + rng.choice(NEIGHBOURS.get(keys[i], [keys[i]])) + keys[i + 1:]
    if kind < 0.6 and len(keys) > 1:
        return keys[:i] + keys[i + 1:]
    if kind < 0.8:
        return keys[:i + 1] + keys[i] + keys[i + 1:]
    if len(keys) > 1:
        i = min(i, len(keys) - 2)
        return keys[:i] + keys[i + 1] + keys[i] + keys[i + 2:]
    return keys


def units(text, rng):
    """(keys, output) per character, typed the way someone might, or None if a character has no pinyin."""
    pairs = []
    pending_space = ""
    han_runs = {m.start(): m.group() for m in re.finditer(r"[一-鿿]+", text)}
    readings = {}
    for start, run in han_runs.items():
        for offset, syllable in enumerate(lazy_pinyin(run)):
            if not syllable.isascii() or not syllable.isalpha():
                return None
            readings[start + offset] = syllable
    for position, character in enumerate(text):
        if character == " ":
            pending_space = " "
            continue
        if position in readings:
            pairs.append([readings[position], pending_space + character, True])
        else:
            pairs.append([character.lower(), pending_space + character, False])
        pending_space = ""

    syllables = [p for p in pairs if p[2]]
    style = rng.random()
    if style < 0.12:
        for p in syllables:
            p[0] = initial(p[0])
    elif style < 0.25:
        for p in syllables:
            if rng.random() < 0.4:
                p[0] = initial(p[0])
    if syllables and rng.random() < 0.25:
        p = rng.choice(syllables)
        p[0] = slip(p[0], rng) or p[0]
    # Typing still going on: the last syllable is only partly there.
    if pairs and pairs[-1][2] and len(pairs[-1][0]) > 1 and rng.random() < 0.15:
        pairs[-1][0] = pairs[-1][0][:rng.randrange(1, len(pairs[-1][0]))]
    # Now and then a separator before a syllable ("xi'an").
    for p in syllables[1:]:
        if rng.random() < 0.03:
            p[0] = "'" + p[0]
    return [(keys, output) for keys, output, _ in pairs]


def examples(source, count, rng, vocab):
    tokens = np.memmap(os.path.join(DATA, f"{source}.bin"), dtype="<u2", mode="r")
    made = 0
    while made < count:
        start = rng.randrange(0, len(tokens) - 400)
        text = "".join(vocab[t] if len(vocab[t]) == 1 else "\n" for t in tokens[start:start + 400])
        matches = [m for m in TYPED.finditer(text) if m.start() > CONTEXT_CHARS and HAN.search(m.group())]
        if not matches:
            continue
        match = rng.choice(matches)
        typed = match.group()
        length = rng.randint(1, MAX_UNITS)
        cut = rng.randrange(0, max(1, len(typed) - length + 1))
        piece = typed[cut:cut + length].strip()
        if not piece:
            continue
        end = match.start() + cut + typed[cut:].find(piece)
        context = text[max(0, end - rng.randint(0, CONTEXT_CHARS)):end]
        # Often only the current line; a model with a wide window sees earlier lines more often.
        keep_line = 0.7 if CONTEXT_CHARS <= 20 else 0.3
        context = context[context.rfind("\n") + 1:] if rng.random() < keep_line else context
        pairs = units(piece, rng)
        if pairs:
            made += 1
            yield context, pairs


def encode(context, pairs, ids):
    """The tokens of one example, or None if a character is unknown or it does not fit a row."""
    text = "".join(output for _, output in pairs)
    if any(character not in ids for character in text):
        return None
    body = [ids[TYPE]]
    if LAYOUT == "keys-first":
        body += [ids["⌨" + key] for keys, _ in pairs for key in keys]
        body += [ids[OUT]] + [ids[character] for character in text] + [ids[END]]
    else:
        for keys, output in pairs:
            body += [ids["⌨" + key] for key in keys] + [ids[character] for character in output]
    room = ROW - 1 - len(body)
    if room < 0:
        return None
    context = [ids.get(c, 2) for c in context][-room:] if room else []
    return [1] + context + body


def extra_tokens():
    return ["⌨" + key for key in KEYS] + [TYPE] + ([OUT, END] if LAYOUT == "keys-first" else [])


def work(job):
    seed, count, context_chars, row_length, max_units, layout = job
    global CONTEXT_CHARS, ROW, MAX_UNITS, LAYOUT
    CONTEXT_CHARS, ROW, MAX_UNITS, LAYOUT = context_chars, row_length, max_units, layout
    with open(os.path.join(DATA, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    full = vocab + extra_tokens()
    ids = {token: index for index, token in enumerate(full)}
    rng = random.Random(seed)
    names = list(SOURCES)
    # Examples are packed into rows one after another, each starting with <bos>; the rest of a
    # row is <pad>, which the loss skips.
    rows = []
    row = []
    while len(rows) < count:
        source = rng.choices(names, weights=list(SOURCES.values()))[0]
        for context, pairs in examples(source, 64, rng, vocab):
            tokens = encode(context, pairs, ids)
            if not tokens:
                continue
            if len(row) + len(tokens) > ROW:
                rows.append(row + [0] * (ROW - len(row)))
                row = []
            row += tokens
    return np.array(rows[:count], dtype="<u2")


def write(out, total, workers, seed):
    """Rows of ROW tokens with up to CONTEXT_CHARS of text before the cursor."""
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(DATA, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    with open(os.path.join(out, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab + extra_tokens(), f, ensure_ascii=False)
    per_job = 20_000
    settings = (CONTEXT_CHARS, ROW, MAX_UNITS, LAYOUT)
    jobs = [(seed + i, per_job, *settings) for i in range(-(-total // per_job))]
    with multiprocessing.Pool(workers) as pool, open(os.path.join(out, "ime.rows"), "wb") as f:
        for done, rows in enumerate(pool.imap_unordered(work, jobs), 1):
            rows.tofile(f)
            if done % 10 == 0:
                print(f"{done * per_job:,} rows", flush=True)
    work((7, 20_000, *settings)).tofile(os.path.join(out, "val.rows"))
    # Plain text keeps the model reading text well; the rows teach it to type.
    for name in PLAIN:
        link = os.path.join(out, f"{name}.bin")
        if not os.path.exists(link):
            os.symlink(os.path.join(DATA, f"{name}.bin"), link)
    with open(os.path.join(out, "sources.json"), "w", encoding="utf-8") as f:
        json.dump({"ime": 1 - sum(PLAIN.values()), **PLAIN}, f, indent=1)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=None)
    parser.add_argument("--rows", type=int, default=3_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--first-seed", type=int, default=1000, help="seed of the first job; give each data set its own")
    parser.add_argument("--block", type=int, default=64, help="the model's window; rows are one token longer")
    parser.add_argument("--context", type=int, default=20, help="most characters of text before the cursor")
    parser.add_argument("--units", type=int, default=10, help="most characters or letters in one example")
    parser.add_argument("--layout", choices=["interleaved", "keys-first"], default="interleaved")
    args = parser.parse_args()
    global CONTEXT_CHARS, ROW, MAX_UNITS, LAYOUT
    CONTEXT_CHARS, ROW, MAX_UNITS, LAYOUT = args.context, args.block + 1, args.units, args.layout
    if args.out:
        write(args.out, args.rows, args.workers, args.first_seed)
        return
    with open(os.path.join(DATA, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    rng = random.Random(args.seed)
    names = list(SOURCES)
    for _ in range(args.samples):
        source = rng.choices(names, weights=list(SOURCES.values()))[0]
        for context, pairs in examples(source, 1, rng, vocab):
            if LAYOUT == "keys-first":
                print(f"[{source}] {context.replace(chr(10), '⏎')} ▶ " + "".join(k for k, _ in pairs) + " ⇒ " + "".join(o for _, o in pairs) + " ■")
            else:
                print(f"[{source}] {context.replace(chr(10), '⏎')} ▶ " + " ".join(f"{k}{o}" for k, o in pairs))


if __name__ == "__main__":
    main()
