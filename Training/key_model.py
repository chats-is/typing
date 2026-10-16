#!/usr/bin/env python3
"""The reading part of the input method, the other way round: given a character, how is it typed?

    Data/venv-train/bin/python Training/key_model.py --out Data/keys --steps 20000

A tiny transformer reads a character token and then its keys, predicting each next key and the
terminator (<end> after a complete reading, <cut> after a syllable still being typed). So for a
span of keys and a character it gives log P(keys | character), one key at a time. Wrong readings
need several unlikely keys in a row, so they come out very unlikely, as they should; a slip of
the finger costs one unlikely key, as in the training pairs (reading_model.py makes them).

The decoder scores a span against the characters the language model proposes, plus the
characters whose likeliest readings start with the span, which the checkpoint lists ("index":
the readings the model itself writes for every character, found once after training).
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_ime_data import DATA, SOURCES, initial, slip  # noqa: E402
import reading_model  # noqa: E402
from reading_model import ENGLISH, KEY_TOKENS, MAX_KEYS, pairs  # noqa: E402
from pypinyin import Style, lazy_pinyin, pinyin  # noqa: E402

END, CUT = 2, 3


class KeyModel(nn.Module):
    """Positions: 0 the character, 1.. the keys; the next key is predicted at every position."""

    def __init__(self, vocab_size, d_model=128, n_layer=2, n_head=4, block=MAX_KEYS + 2, n_keys=len(KEY_TOKENS), prev=0):
        super().__init__()
        self.prev = prev  # keys of the previous character read first (0: none), so the model knows the typist's style
        self.block = prev + block
        block = self.block
        self.characters = nn.Embedding(vocab_size, d_model)
        self.keys = nn.Embedding(n_keys, d_model)
        self.positions = nn.Embedding(block, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_head, 4 * d_model, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layer, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d_model)
        self.next_key = nn.Linear(d_model, n_keys)
        self.register_buffer("causal", torch.full((block, block), float("-inf")).triu(1), persistent=False)

    def forward(self, characters, keys, prev=None):
        """characters: (batch,); keys: (batch, n) key tokens, <pad> after the span; prev: (batch, self.prev)
        the previous character's keys, <pad> after them (all <pad> at the start of a sentence).
        Returns next-key logits at the character and every key (the character predicts the first key)."""
        parts = [self.characters(characters)[:, None], self.keys(keys)]
        if self.prev:
            parts.insert(0, self.keys(prev if prev is not None else torch.zeros((len(keys), self.prev), dtype=torch.long, device=keys.device)))
        x = torch.cat(parts, 1)
        n = x.shape[1]
        x = x + self.positions.weight[:n]
        x = self.ln(self.encoder(x, mask=self.causal[:n, :n]))
        return self.next_key(x)[:, self.prev :]

    @torch.no_grad()
    def span_logprobs(self, characters, keys, final, prev=None):
        """log P(keys | character) for rows of (character, keys): keys is (batch, n) of key tokens
        with 0 after the span; `final` rows may also end in <cut>; prev as in forward()."""
        logits = self(characters, keys, prev)
        logp = torch.log_softmax(logits.float(), -1)
        length = (keys > 0).sum(1)
        positions = torch.arange(keys.shape[1] + 1, device=keys.device)
        targets = torch.cat([keys, torch.zeros((len(keys), 1), dtype=torch.long, device=keys.device)], 1)
        stepwise = logp.gather(2, targets[..., None])[..., 0]
        keys_lp = (stepwise * (positions[None] < length[:, None])).sum(1)
        at_end = logp[torch.arange(len(keys), device=keys.device), length]
        end = at_end[:, END]
        if final is not None:
            end = torch.where(final, torch.logaddexp(at_end[:, END], at_end[:, CUT]), end)
        return keys_lp + end


def slip_variants(text):
    """The strings one slip of the finger away from `text`, as the decoder's slip index makes them:
    a neighbouring key, a key missed, doubled or swapped."""
    from make_ime_data import NEIGHBOURS
    variants = set()
    for i in range(len(text)):
        for near in NEIGHBOURS.get(text[i], []):
            variants.add(text[:i] + near + text[i + 1:])
        if len(text) > 1:
            variants.add(text[:i] + text[i + 1:])
        variants.add(text[:i + 1] + text[i] + text[i + 1:])
        if i + 1 < len(text) and text[i] != text[i + 1]:
            variants.add(text[:i] + text[i + 1] + text[i] + text[i + 2:])
    return variants


def index_readings(model, vocab, device, width=4, keep=3, margin=None, dedupe_slips=False):
    """For every character the readings the model writes most readily (beam of `width`, best `keep`,
    and when `margin` is given only those within that many nats of its best), so the decoder can
    find rare characters by their keys."""
    model.eval()
    index = {}
    characters = [i for i, t in enumerate(vocab) if len(t) == 1 and ("一" <= t <= "鿿" or t.isascii() and t.isalpha())]
    for start in range(0, len(characters), 512):
        chunk = characters[start : start + 512]
        n = len(chunk)
        beams = torch.zeros((n, width, MAX_KEYS + 1), dtype=torch.long, device=device)
        scores = torch.zeros((n, width), device=device)
        scores[:, 1:] = float("-inf")
        done = [[] for _ in chunk]
        chars = torch.tensor(chunk, device=device).repeat_interleave(width)
        with torch.no_grad():
            for step in range(MAX_KEYS + 1):
                logits = model(chars, beams.reshape(n * width, -1))[:, step].float().reshape(n, width, -1)
                logp = torch.log_softmax(logits, -1)
                logp[..., :2] = float("-inf")  # no <pad>, no <bos>
                logp[..., CUT] = float("-inf")
                if step == 0:
                    logp[..., END] = float("-inf")
                total = scores[..., None] + logp
                for b in range(n):
                    finished = total[b, :, END]
                    for w in range(width):
                        if finished[w] > float("-inf"):
                            done[b].append((finished[w].item(), beams[b, w, :step].tolist()))
                total[..., END] = float("-inf")
                flat = total.reshape(n, -1)
                best, which = flat.topk(width)
                scores = best
                beams = torch.stack([beams[b, which[b] // logp.shape[-1]] for b in range(n)])
                beams[:, :, step] = which % logp.shape[-1]
        for b, c in enumerate(chunk):
            ranked = sorted(done[b], key=lambda d: -d[0])
            kept = []
            for score, keys in ranked:
                if len(kept) == keep or (margin is not None and ranked and score < ranked[0][0] - margin):
                    break
                reading = "".join(KEY_TOKENS[k] for k in keys)
                if not reading or ENGLISH in reading:
                    continue
                # A reading one slip away from a better one of the same character adds nothing: the
                # decoder's slip index reaches it from the better one.
                if dedupe_slips and any(reading in slip_variants(better) for better in kept):
                    continue
                kept.append(reading)
                index.setdefault(reading, []).append((c, score))
    return index


ABBR_ENTER, ABBR_STAY = 0.08, 0.7  # typing by initials comes in runs: whole words and phrases (--abbr-enter/--abbr-stay)
ABBR_CHAR = 0.15  # share of character_pairs typed by the initial (--abbr-char)
# Letters mostly reach the text without the pinyin decoder: people switch to English with Shift,
# nearly always for capitals. Only the rest is typed as pinyin keys and must be read by it.
ENGLISH_UPPER, ENGLISH_LOWER = 0.8, 0.3  # --english-upper/--english-lower
UNIT = re.compile(r"[一-鿿]+|[A-Za-z]|.", re.S)


def typed(syllable, rng, abbr):
    """Keys for one character in a run, (keys, still typing): its initial mostly while abbreviating,
    otherwise the syllable, sometimes unfinished, slipped or preceded by an apostrophe."""
    keys, partial = syllable, False
    if abbr and rng.random() < 0.9:
        keys = initial(keys)
        if len(keys) == 2 and rng.random() < 0.5:
            keys = keys[0]  # zh, ch, sh abbreviated to one letter, as people mostly do
    elif not abbr and rng.random() < 0.05 and len(keys) > 1:
        keys = keys[: rng.randrange(1, len(keys))]
        partial = True
    if rng.random() < 0.08:
        keys = slip(keys, rng) or keys
    if rng.random() < 0.03:
        keys = "'" + keys
    return keys[:MAX_KEYS], partial


def context_pairs(count, rng, vocab):
    """(previous keys, keys, character, still typing) sampled from the corpora, typed in runs with a
    style that persists: either syllable by syllable or by initials."""
    names = list(SOURCES)
    weights = list(SOURCES.values())
    streams = {name: np.memmap(os.path.join(DATA, f"{name}.bin"), dtype="<u2", mode="r") for name in names}
    out = []
    while len(out) < count:
        tokens = streams[rng.choices(names, weights=weights)[0]]
        start = rng.randrange(0, len(tokens) - 200)
        text = "".join(vocab[t] if len(vocab[t]) == 1 else "\n" for t in tokens[start : start + 200])
        prev, abbr = "", False
        for unit in UNIT.finditer(text):
            unit = unit.group()
            if "一" <= unit[0] <= "鿿":
                syllables = lazy_pinyin(unit)
                if len(syllables) != len(unit):
                    prev = ""
                    continue
                for character, syllable in zip(unit, syllables):
                    if not (syllable.isascii() and syllable.isalpha()):
                        prev = ""
                        continue
                    abbr = rng.random() < (ABBR_STAY if abbr else ABBR_ENTER)
                    keys, partial = typed(syllable, rng, abbr)
                    out.append((prev if rng.random() > 0.05 else "", keys, character, partial))
                    prev = keys
            elif unit.isascii() and unit.isalpha():
                keys = ENGLISH if rng.random() < (ENGLISH_UPPER if unit.isupper() else ENGLISH_LOWER) else unit.lower()
                out.append((prev, keys, unit, False))
                prev, abbr = keys, False
    rng.shuffle(out)
    return out[:count]


def character_pairs(count, rng, vocab):
    """(previous keys, keys, character, still typing) with every character of the vocabulary drawn
    equally often and read by each of its pinyin readings (pypinyin, heteronyms included): rare
    characters are seldom seen in the corpora, and a character the model cannot read cannot be
    typed. The previous keys are those of a random syllable, or none."""
    characters = [t for t in vocab if len(t) == 1 and "一" <= t <= "鿿"]
    readings = {}
    for c in characters:
        options = [r for r in pinyin(c, style=Style.NORMAL, heteronym=True)[0] if r.isascii() and r.isalpha()]
        if options:
            readings[c] = options
    # Readings people actually type that pypinyin lacks (嗯 as "en"), from the old readings table
    # when it is around; the table is only read here, to make training pairs.
    table = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "engine_old", "readings.tsv")
    if os.path.exists(table):
        with open(table, encoding="utf-8") as f:
            for line in f:
                reading, c = line.rstrip("\n").split("\t")[:2]
                if c in readings and reading not in readings[c] and reading.isascii() and reading.isalpha():
                    readings[c].append(reading)
                elif c not in readings and len(c) == 1 and c in characters:
                    readings[c] = [reading]
    characters = list(readings)
    syllables = sorted({r for options in readings.values() for r in options})
    out = []
    while len(out) < count:
        c = rng.choice(characters)
        options = readings[c]
        # The first reading pypinyin lists is the usual one; the others still come up.
        syllable = options[0] if len(options) == 1 or rng.random() < 0.6 else rng.choice(options[1:])
        abbr = rng.random() < ABBR_CHAR
        keys, partial = typed(syllable, rng, abbr)
        previous = "" if rng.random() < 0.4 else typed(rng.choice(syllables), rng, rng.random() < ABBR_CHAR)[0]
        out.append((previous, keys, c, partial))
    return out


def main():
    global ABBR_ENTER, ABBR_STAY, ABBR_CHAR, ENGLISH_UPPER, ENGLISH_LOWER
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(os.path.dirname(DATA), "keys"))
    parser.add_argument("--vocab", default=os.path.join(DATA, "vocab.json"))
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--pairs", type=int, default=4_000_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--initial", type=float, default=reading_model.INITIAL_RATE, help="share of training pairs typed by the initial alone")
    parser.add_argument("--context", action="store_true", help="read the previous character's keys too (trained on runs)")
    parser.add_argument("--index-only", action="store_true", help="only rebuild the readings index of --out")
    parser.add_argument("--keep", type=int, default=3, help="readings per character kept in the index")
    parser.add_argument("--margin", type=float, default=None, help="index only readings within this many nats of the character's best")
    parser.add_argument("--dedupe-slips", action="store_true", help="leave out index readings one slip away from a better reading of the same character")
    parser.add_argument("--uniform", type=float, default=0.0, help="share of training pairs drawn equally over all characters (character_pairs)")
    parser.add_argument("--abbr-enter", type=float, default=ABBR_ENTER, help="chance a run of initials starts at a character (context_pairs)")
    parser.add_argument("--abbr-stay", type=float, default=ABBR_STAY, help="chance a run of initials continues at the next character")
    parser.add_argument("--abbr-char", type=float, default=ABBR_CHAR, help="share of character_pairs typed by the initial")
    parser.add_argument("--english-upper", type=float, default=ENGLISH_UPPER, help="share of capital letters typed in English mode, not as pinyin keys")
    parser.add_argument("--english-lower", type=float, default=ENGLISH_LOWER, help="share of small letters typed in English mode")
    args = parser.parse_args()
    reading_model.INITIAL_RATE = args.initial
    ABBR_ENTER, ABBR_STAY, ABBR_CHAR = args.abbr_enter, args.abbr_stay, args.abbr_char
    ENGLISH_UPPER, ENGLISH_LOWER = args.english_upper, args.english_lower
    with open(args.vocab, encoding="utf-8") as f:
        vocab = json.load(f)
    ids = {token: index for index, token in enumerate(vocab)}
    key_ids = {k: i for i, k in enumerate(KEY_TOKENS)}
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    path = os.path.join(args.out, "model.pt")
    prev = MAX_KEYS if args.context else 0

    def save(model, index=None):
        os.makedirs(args.out, exist_ok=True)
        torch.save({"kind": "keys", "config": {"vocab_size": len(vocab), "d_model": args.dim, "n_layer": args.layers, "n_keys": len(KEY_TOKENS), "block": MAX_KEYS + 2, "prev": prev},
                    "model": model.state_dict(), "key_tokens": KEY_TOKENS, "index": index}, path)
        with open(os.path.join(args.out, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)

    if args.index_only:
        checkpoint = torch.load(path, map_location="cpu")
        model = KeyModel(**checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["model"])
        index = index_readings(model, vocab, device, width=max(4, args.keep), keep=args.keep, margin=args.margin, dedupe_slips=args.dedupe_slips)
        checkpoint["index"] = index
        torch.save(checkpoint, path)
        print(f"{len(index)} readings in the index")
        return

    rng = random.Random(1)
    started = time.time()
    uniform = int(args.pairs * args.uniform)
    data = context_pairs(args.pairs - uniform, rng, vocab) if args.context else [("",) + p for p in pairs(args.pairs - uniform, rng, vocab)]
    if uniform:
        data += character_pairs(uniform, rng, vocab)
        rng.shuffle(data)
    held_out, data = data[:20000], data[20000:]
    print(f"{len(data):,} pairs in {(time.time() - started) / 60:.1f} min", flush=True)
    model = KeyModel(len(vocab), args.dim, args.layers, prev=prev).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters", flush=True)

    def encode(batch):
        characters = torch.tensor([ids.get(c, 2) for _, _, c, _ in batch])
        before = torch.zeros((len(batch), MAX_KEYS), dtype=torch.long)
        keys = torch.zeros((len(batch), MAX_KEYS + 1), dtype=torch.long)
        targets = torch.zeros((len(batch), MAX_KEYS + 2), dtype=torch.long)
        for r, (previous, span, _, partial) in enumerate(batch):
            row = [key_ids[k] for k in span]
            keys[r, : len(row)] = torch.tensor(row)
            targets[r, : len(row) + 1] = torch.tensor(row + [CUT if partial else END])
            if previous:
                previous = [key_ids[k] for k in previous[-MAX_KEYS:]]
                before[r, : len(previous)] = torch.tensor(previous)
        return characters.to(device), keys.to(device), targets.to(device), before.to(device)

    def loss_of(batch):
        characters, keys, targets, before = encode(batch)
        logits = model(characters, keys, before if prev else None)
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=0)

    model.train()
    for step in range(1, args.steps + 1):
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps)) if step > 500 else args.lr * step / 500
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss = loss_of([data[rng.randrange(len(data))] for _ in range(args.batch)])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 1000 == 0:
            model.eval()
            with torch.no_grad():
                held = sum(loss_of(held_out[i : i + 1000]).item() for i in range(0, 5000, 1000)) / 5
            model.train()
            print(f"step {step}/{args.steps}  train {loss.item():.3f}  held-out {held:.3f}  {(time.time() - started) / 60:.1f} min", flush=True)
            save(model)
    index = index_readings(model, vocab, device, width=max(4, args.keep), keep=args.keep, margin=args.margin, dedupe_slips=args.dedupe_slips)
    save(model, index)
    print(f"wrote {args.out}: {len(index)} readings in the index")


if __name__ == "__main__":
    main()
