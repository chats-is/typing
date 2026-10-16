#!/usr/bin/env python3
"""The reading part of the input method: a small model that reads one character's typed keys.

    Data/venv-train/bin/python Training/reading_model.py --out Data/reading --steps 20000

It is trained on (keys, character) pairs made the way make_ime_data.py makes them: the pinyin
of each character (pypinyin, used here only), its initial, an unfinished syllable, a slip of the
finger, a separating apostrophe, and English letters in their case. Characters are sampled as
often as they occur in the corpora, so the model also knows their overall frequency.

For a span of keys it gives two things the decoder multiplies with the language model:
  - log P(keys): how likely someone types this span for one character (sum of next-key
    log-probabilities and the terminator: <end> for a complete reading, <cut> for a syllable
    still being typed, which only the last span may be), which settles how keys are cut;
  - log P(character | keys, terminator) over the whole vocabulary, read at the terminator.
The checkpoint also carries log P(character), so that
  log P(keys | character) = log P(character | keys) + log P(keys) - log P(character).
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
from make_ime_data import DATA, INITIALS, KEYS, SOURCES, initial, slip  # noqa: E402
from pypinyin import lazy_pinyin  # noqa: E402

MAX_KEYS = 7
INITIAL_RATE = 0.12  # how often a character is typed by its initial alone (key_model.py may change it)
ENGLISH = "\u21e7"  # a letter typed after switching to English (Shift), which the pinyin decoder never sees
KEY_TOKENS = ["<pad>", "<bos>", "<end>", "<cut>"] + list(KEYS) + [ENGLISH]  # <cut>: the syllable is still being typed
HAN = re.compile(r"[一-鿿]")
LETTER = re.compile(r"[A-Za-z]")


class ReadingModel(nn.Module):
    """A tiny transformer over a key span with a next-key head and a character head."""

    def __init__(self, vocab_size, d_model=128, n_layer=2, n_head=4, block=MAX_KEYS + 2, n_keys=len(KEY_TOKENS)):
        super().__init__()
        self.block = block
        self.keys = nn.Embedding(n_keys, d_model)
        self.positions = nn.Embedding(block, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_head, 4 * d_model, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layer)
        self.ln = nn.LayerNorm(d_model)
        self.next_key = nn.Linear(d_model, n_keys)
        self.character = nn.Linear(d_model, vocab_size)
        self.register_buffer("causal", torch.full((block, block), float("-inf")).triu(1), persistent=False)

    def forward(self, ids):
        """ids: (batch, block) of key tokens starting with <bos>. Returns next-key logits and
        character logits at every position."""
        x = self.keys(ids) + self.positions.weight[: ids.shape[1]]
        n = ids.shape[1]
        x = self.ln(self.encoder(x, mask=self.causal[:n, :n]))
        return self.next_key(x), self.character(x)


def keys_for(character, syllable, rng):
    """How someone might type one character whose pinyin is `syllable` (None for a letter):
    (keys, still typing)."""
    if syllable is None:
        return character.lower(), False
    keys = syllable
    partial = False
    r = rng.random()
    if r < INITIAL_RATE:
        keys = initial(keys)
    elif r < INITIAL_RATE + 0.05 and len(keys) > 1:
        keys = keys[: rng.randrange(1, len(keys))]  # typing still going on
        partial = True
    if rng.random() < 0.08:
        keys = slip(keys, rng) or keys
    if rng.random() < 0.03:
        keys = "'" + keys
    return keys[:MAX_KEYS], partial


def pairs(count, rng, vocab):
    """(keys, character) pairs sampled from the corpora."""
    names = list(SOURCES)
    weights = list(SOURCES.values())
    streams = {name: np.memmap(os.path.join(DATA, f"{name}.bin"), dtype="<u2", mode="r") for name in names}
    out = []
    while len(out) < count:
        tokens = streams[rng.choices(names, weights=weights)[0]]
        start = rng.randrange(0, len(tokens) - 200)
        text = "".join(vocab[t] if len(vocab[t]) == 1 else "\n" for t in tokens[start : start + 200])
        for run in re.finditer(r"[一-鿿]+", text):
            syllables = lazy_pinyin(run.group())
            if len(syllables) != len(run.group()):
                continue
            for character, syllable in zip(run.group(), syllables):
                if syllable.isascii() and syllable.isalpha():
                    keys, partial = keys_for(character, syllable, rng)
                    out.append((keys, character, partial))
        for match in LETTER.finditer(text):
            out.append((match.group().lower(), match.group(), False))
    rng.shuffle(out)
    return out[:count]


def encode(batch, ids, device):
    """Rows <bos> keys terminator, with next-token targets and the character target at the terminator."""
    x = torch.zeros((len(batch), MAX_KEYS + 2), dtype=torch.long)
    next_keys = torch.zeros_like(x)
    characters = torch.zeros(len(batch), dtype=torch.long)
    last = torch.zeros(len(batch), dtype=torch.long)
    key_ids = {k: i for i, k in enumerate(KEY_TOKENS)}
    for r, (keys, character, partial) in enumerate(batch):
        row = [1] + [key_ids[k] for k in keys] + [3 if partial else 2]
        x[r, : len(row)] = torch.tensor(row)
        next_keys[r, : len(row) - 1] = torch.tensor(row[1:])
        characters[r] = ids.get(character, 2)
        last[r] = len(row) - 1
    return x.to(device), next_keys.to(device), characters.to(device), last.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(os.path.dirname(DATA), "reading"))
    parser.add_argument("--vocab", default=os.path.join(DATA, "vocab.json"))
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--pairs", type=int, default=4_000_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--samples", type=int, default=0, help="only print this many example pairs")
    args = parser.parse_args()
    with open(args.vocab, encoding="utf-8") as f:
        vocab = json.load(f)
    ids = {token: index for index, token in enumerate(vocab)}
    rng = random.Random(1)
    if args.samples:
        for keys, character, partial in pairs(args.samples, rng, vocab):
            print(keys, character, "…" if partial else "")
        return
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    started = time.time()
    data = pairs(args.pairs, rng, vocab)
    held_out, data = data[:20000], data[20000:]
    print(f"{len(data):,} pairs in {(time.time() - started) / 60:.1f} min", flush=True)
    counts = torch.full((len(vocab),), 0.5)
    for _, character, _ in data:
        counts[ids.get(character, 2)] += 1
    unigram = torch.log(counts / counts.sum())

    model = ReadingModel(len(vocab), args.dim, args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters", flush=True)

    def losses(batch):
        x, next_keys, characters, last = encode(batch, ids, device)
        key_logits, char_logits = model(x)
        key_loss = F.cross_entropy(key_logits.flatten(0, 1), next_keys.flatten(), ignore_index=0)
        char_loss = F.cross_entropy(char_logits[torch.arange(len(batch), device=device), last], characters)
        return key_loss, char_loss

    def save():
        os.makedirs(args.out, exist_ok=True)
        torch.save({"config": {"vocab_size": len(vocab), "d_model": args.dim, "n_layer": args.layers, "n_keys": len(KEY_TOKENS), "block": MAX_KEYS + 2},
                    "model": model.state_dict(), "unigram": unigram, "key_tokens": KEY_TOKENS}, os.path.join(args.out, "model.pt"))
        with open(os.path.join(args.out, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)

    model.train()
    for step in range(1, args.steps + 1):
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps)) if step > 500 else args.lr * step / 500
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = [data[rng.randrange(len(data))] for _ in range(args.batch)]
        key_loss, char_loss = losses(batch)
        optimizer.zero_grad(set_to_none=True)
        (key_loss + char_loss).backward()
        optimizer.step()
        if step % 500 == 0:
            model.eval()
            with torch.no_grad():
                k, c = zip(*(losses(held_out[i : i + 1000]) for i in range(0, 5000, 1000)))
            model.train()
            print(f"step {step}/{args.steps}  train keys {key_loss.item():.3f} char {char_loss.item():.3f}"
                  f"  held-out keys {sum(k).item() / len(k):.3f} char {sum(c).item() / len(c):.3f}  {(time.time() - started) / 60:.1f} min", flush=True)
            save()
    save()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
