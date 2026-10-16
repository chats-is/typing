#!/usr/bin/env python3
"""Train the language model on the Mac's GPU.

    Data/venv-train/bin/python Training/train.py --data Data/lm9 --out Data/lm9/model.pt --init Data/base/lm3_model.pt --lr 5e-5 --steps 6000

--data is a directory written by prepare_data.py. Every batch mixes its sources in the
proportions given by sources.json. Stops after --steps (or --hours), saving a checkpoint every
--save-every steps. Running it again with the same --out resumes from the checkpoint.

--init model.pt starts from another run's weights instead of from scratch (with a fresh optimizer
and learning-rate schedule). --block widens its window; the new positions are spread out from the
old ones. If --data has a larger vocabulary, as after prepare_data.py
--cased-from, the new tokens start as copies of their small letters (key tokens "⌨a" of an earlier key-model experiment
as copies of their letters).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAD = 0  # filler at the end of .rows rows; never a target


def batch_from(sources, weights, batch_size, block_size, device, generator):
    """Rows of block_size + 1 tokens, each from a source picked according to `weights`: a window
    of a token stream, or one of the ready-made rows of a .rows source."""
    picks = generator.choice(len(sources), size=batch_size, p=weights)
    rows = []
    for pick in picks:
        data = sources[pick]
        if data.ndim == 2:
            rows.append(data[generator.integers(0, len(data))])
            continue
        start = generator.integers(0, len(data) - block_size - 1)
        rows.append(data[start : start + block_size + 1])
    rows = torch.from_numpy(np.stack(rows).astype(np.int64)).to(device)
    return rows[:, :-1], rows[:, 1:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(ROOT, "Data", "lm9"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--hours", type=float, default=None)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--only", default=None, help="comma-separated sources to train on instead of the mix in sources.json")
    parser.add_argument("--init", default=None, help="checkpoint whose weights a new run starts from")
    parser.add_argument("--skip-keys", action="store_true", help="no loss on the typed keys (they are given, not predicted)")
    parser.add_argument("--block", type=int, default=None, help="window size (default 64, or the --init model's)")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    with open(os.path.join(args.data, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    with open(os.path.join(args.data, "sources.json"), encoding="utf-8") as f:
        shares = json.load(f)
    if args.only:
        shares = {name: shares[name] for name in args.only.split(",")}
    def source(name):
        # <name>.rows (an earlier key-model experiment) holds rows of block_size + 1 tokens, <pad>-filled.
        rows = os.path.join(args.data, f"{name}.rows")
        if os.path.exists(rows):
            return np.memmap(rows, dtype="<u2", mode="r").reshape(-1, config.block_size + 1)
        return np.memmap(os.path.join(args.data, f"{name}.bin"), dtype="<u2", mode="r")

    step = 0
    if os.path.exists(args.out):
        checkpoint = torch.load(args.out, map_location="cpu")
        config = Config(**checkpoint["config"])
        model = GPT(config).to(device)
        model.load_state_dict(checkpoint["model"])
        step = checkpoint["step"]
        total_steps = checkpoint["total_steps"]
    elif args.init:
        checkpoint = None
        initial = torch.load(args.init, map_location="cpu")
        weights_in = initial["model"]
        old_size = initial["config"]["vocab_size"]
        config = Config(**{**initial["config"], "vocab_size": vocab_size, "block_size": args.block or initial["config"]["block_size"]})
        old_positions = weights_in["positions.weight"]
        if config.block_size != len(old_positions):
            # Stretch the learned positions over the wider window.
            weights_in["positions.weight"] = F.interpolate(old_positions.T[None], size=config.block_size, mode="linear", align_corners=True)[0].T.contiguous()
        if vocab_size > old_size:
            # Token embeddings and output weights are shared; each new token copies its small
            # letter, with a little noise so the two can drift apart.
            embedding = weights_in["tokens.weight"]
            rows = []
            for token in vocab[old_size:]:
                # A key token ("⌨a") starts as its letter.
                plain = token[1:] if token.startswith("⌨") else token.lower()
                origin = vocab.index(plain) if plain in vocab[:old_size] else None
                row = embedding[origin].clone() if origin is not None else embedding.mean(0)
                rows.append(row + 0.01 * embedding.std() * torch.randn_like(row))
            expanded = torch.cat([embedding, torch.stack(rows)])
            weights_in["tokens.weight"] = expanded
            weights_in["head.weight"] = expanded
        model = GPT(config).to(device)
        model.load_state_dict(weights_in)
        total_steps = args.steps
    else:
        checkpoint = None
        config = Config(vocab_size=vocab_size, block_size=args.block or 64, n_layer=args.layers, n_head=args.heads, d_model=args.dim)
        model = GPT(config).to(device)
        total_steps = args.steps
    train = [source(name) for name in shares]
    weights = np.array(list(shares.values()), dtype=np.float64)
    weights /= weights.sum()
    val = [source("val")]
    # Typed keys are given to the model, not predicted: with --skip-keys they are never targets.
    skipped = torch.tensor([i for i, token in enumerate(vocab) if token.startswith("⌨")] if args.skip_keys else [], dtype=torch.long, device=device)

    def targets(y):
        return y.masked_fill(torch.isin(y, skipped), PAD) if len(skipped) else y

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    parameters = sum(p.numel() for p in model.parameters())
    print(f"{parameters / 1e6:.1f}M parameters, {sum(len(d) for d in train) / 1e6:.0f}M training tokens, "
          f"step {step}/{total_steps}, device {device}", flush=True)

    warmup = min(1000, total_steps // 10)

    def learning_rate(at):
        if at < warmup:
            return args.lr * (at + 1) / warmup
        progress = (at - warmup) / max(1, total_steps - warmup)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))

    def save():
        temporary = args.out + ".tmp"
        torch.save({"config": config.to_dict(), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "total_steps": total_steps}, temporary)
        os.replace(temporary, args.out)

    @torch.no_grad()
    def validation_loss():
        model.eval()
        generator = np.random.default_rng(0)
        losses = []
        for _ in range(20):
            x, y = batch_from(val, [1.0], args.batch, config.block_size, device, generator)
            losses.append(F.cross_entropy(model(x).flatten(0, 1), targets(y).flatten(), ignore_index=PAD).item())
        model.train()
        return sum(losses) / len(losses)

    generator = np.random.default_rng(step + 1)
    started = time.time()
    deadline = started + args.hours * 3600 if args.hours else None
    running = None
    model.train()
    while step < total_steps and (deadline is None or time.time() < deadline):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)
        x, y = batch_from(train, weights, args.batch, config.block_size, device, generator)
        loss = F.cross_entropy(model(x).flatten(0, 1), targets(y).flatten(), ignore_index=PAD)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        value = loss.item()
        running = value if running is None else 0.98 * running + 0.02 * value
        if step % 100 == 0:
            elapsed = time.time() - started
            print(f"step {step}/{total_steps}  loss {running:.3f}  lr {learning_rate(step):.2e}  {elapsed / 60:.1f} min", flush=True)
        if step % args.save_every == 0:
            print(f"step {step}  validation loss {validation_loss():.3f}", flush=True)
            save()
    print(f"finished at step {step}  validation loss {validation_loss():.3f}", flush=True)
    save()


if __name__ == "__main__":
    main()
