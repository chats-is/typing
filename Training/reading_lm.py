#!/usr/bin/env python3
"""The reading-conditioned model: given the text so far and the reading of the next character,
which character is it? (Data from reading_lm_data.py.)

    Data/venv-train/bin/python Training/reading_lm.py --data Data/rlm --out Data/rlm1 --steps 15000

This is exactly the decision the lattice search makes at every position: the lexicon says which
words the keys can be, the model says which of them the context wants. It is small (4 layers,
256 wide) because it only ever has to tell homophones apart, never to produce text; that is what
lets it score every candidate at every position within a keystroke.

In a quarter of the positions the reading is withheld (id 0), so the same model also predicts
the next character with no reading given - what suggestions need.

Checkpoint: {"kind": "reading_lm", "config", "model", "syllables", "vocab_size"}.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = {"dialogue": 0.38, "forum": 0.22, "touchpal": 0.05, "mixed": 0.10, "web": 0.12, "news": 0.08, "novels": 0.05}
MASK_READING = 0.25


class Block(nn.Module):
    def __init__(self, d, heads, block):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d)
        self.out = nn.Linear(4 * d, d)
        self.heads = heads
        self.d = d
        self.block = block
        self.register_buffer("mask", torch.full((block, block), float("-inf")).triu(1), persistent=False)

    def forward(self, x):
        # Shapes come from the configuration, not the input, so the exported graph is static.
        d, heads = self.d, self.heads
        t = x.shape[1]
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q = q.reshape(-1, t, heads, d // heads).transpose(1, 2)
        k = k.reshape(-1, t, heads, d // heads).transpose(1, 2)
        v = v.reshape(-1, t, heads, d // heads).transpose(1, 2)
        weights = (q @ k.transpose(-2, -1)) / math.sqrt(d // heads) + self.mask
        attended = (F.softmax(weights, dim=-1) @ v).transpose(1, 2).reshape(-1, t, d)
        x = x + self.proj(attended)
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class ReadingLM(nn.Module):
    def __init__(self, vocab_size, n_syllables, d_model=256, n_layer=4, n_head=4, block_size=64):
        super().__init__()
        self.block_size = block_size
        self.tokens = nn.Embedding(vocab_size, d_model)
        self.readings = nn.Embedding(n_syllables + 1, d_model)  # 0: no reading given
        self.positions = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_head, block_size) for _ in range(n_layer))
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tokens.weight
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def hidden(self, ids, next_readings):
        """Rows are always block_size long (padded with 0), so the graph is static for export."""
        x = self.tokens(ids) + self.readings(next_readings) + self.positions.weight
        for block in self.blocks:
            x = block(x)
        return self.ln(x)

    def forward(self, ids, next_readings):
        """ids: (batch, t) tokens; next_readings: (batch, t) the reading of the token that follows
        each position (0 = not given). Returns logits for the following token at every position."""
        return self.head(self.hidden(ids, next_readings))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(ROOT, "Data", "rlm"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--block", type=int, default=64)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--init", default=None)
    args = parser.parse_args()

    with open(os.path.join(args.data, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    with open(os.path.join(args.data, "syllables.json"), encoding="utf-8") as f:
        syllables = json.load(f)
    names = [n for n in SOURCES if os.path.exists(os.path.join(args.data, f"{n}.bin"))]
    weights = np.array([SOURCES[n] for n in names])
    weights /= weights.sum()
    tokens = {n: np.memmap(os.path.join(args.data, f"{n}.bin"), dtype="<u2", mode="r") for n in names}
    readings = {n: np.memmap(os.path.join(args.data, f"{n}.syl"), dtype="<u2", mode="r") for n in names}
    # The last 2M tokens of dialogue are held out.
    held = 2_000_000
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    config = {"vocab_size": len(vocab), "n_syllables": len(syllables), "d_model": args.dim, "n_layer": args.layers, "n_head": args.heads, "block_size": args.block}
    model = ReadingLM(**config).to(device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu")["model"])
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters, sources {names}, device {device}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    rng = np.random.default_rng(1)

    def batch(validation=False):
        rows, syl = [], []
        for _ in range(args.batch):
            name = "dialogue" if validation else names[rng.choice(len(names), p=weights)]
            data, syls = tokens[name], readings[name]
            if validation:
                start = rng.integers(len(data) - held, len(data) - args.block - 1)
            else:
                limit = len(data) - args.block - 1 - (held if name == "dialogue" else 0)
                start = rng.integers(0, limit)
            rows.append(data[start : start + args.block + 1])
            syl.append(syls[start : start + args.block + 1])
        rows = torch.from_numpy(np.stack(rows).astype(np.int64))
        syl = torch.from_numpy(np.stack(syl).astype(np.int64))
        x, y = rows[:, :-1], rows[:, 1:]
        next_readings = syl[:, 1:].clone()
        if not validation:
            next_readings[torch.rand(next_readings.shape) < MASK_READING] = 0
        return x.to(device), next_readings.to(device), y.to(device)

    os.makedirs(args.out, exist_ok=True)
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps)) if step > 300 else args.lr * step / 300
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, r, y = batch()
        logits = model(x, r)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"step {step}/{args.steps}  loss {loss.item():.3f}  lr {lr:.2e}  {(time.time() - started) / 60:.1f} min", flush=True)
        if step % 1000 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                total, hits, n = 0.0, 0, 0
                for _ in range(10):
                    x, r, y = batch(validation=True)
                    logits = model(x, r)
                    total += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=0).item()
                    han = r > 0
                    hits += ((logits.argmax(-1) == y) & han).sum().item()
                    n += han.sum().item()
            model.train()
            print(f"step {step}  validation loss {total / 10:.3f}  character right given reading and context {100 * hits / max(1, n):.1f}%", flush=True)
            torch.save({"kind": "reading_lm", "config": config, "model": model.state_dict(), "syllables": syllables, "vocab_size": len(vocab)}, os.path.join(args.out, "model.pt"))
            with open(os.path.join(args.out, "vocab.json"), "w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False)
    print(f"finished in {(time.time() - started) / 60:.0f} min")


if __name__ == "__main__":
    main()
