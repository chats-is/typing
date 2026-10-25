#!/usr/bin/env python3
"""Level-4 verification: does tuning the language model on earlier typed text make
the later input of the same text come out right more often, without hurting general input?

    venv/bin/python Training/personalize_eval.py --model Data/clean3/model.pt --vocab Data/prepared-clean/vocab.json \
        --train Data/personal/forum2.train.txt --eval Data/personal/forum2.lattice.tsv \
        --control Resources/general_lattices.tsv --steps 400 --eval-every 100

--eval / --control are lattice dumps from `enginecli --no-rerank --dump-lattice`: every line holds
the typed letters, the right sentence, the preceding text and the lattice's readings as
text|lattice score|typing cost. The readings are reranked exactly as the engine does
(score − 2·cost + 2·log P(reading | context)), first with the shipped model, then with the model
after N steps of tuning on --train. Reports the sentence exact rate of both sets at every
checkpoint, plus the model's perplexity on the later sentences.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, Config  # noqa: E402

BOS, UNK = 1, 2
CONTEXT_WEIGHT = 2.0  # LexiconDecoder.contextWeight
LATTICE_WEIGHT = 1.0  # LexiconDecoder.rerankLatticeWeight


def load(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    config = Config(**checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    return model.to(device), config


def read_dump(path):
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4 or not fields[3]:
                continue
            readings = []
            for item in fields[3].split(" ;; "):
                parts = item.rsplit("|", 2)
                if len(parts) == 3:
                    readings.append((parts[0], float(parts[1]), float(parts[2])))
            lines.append((fields[1], fields[2].replace("\\n", "\n"), readings))
    return lines


class Harness:
    def __init__(self, vocab_path, block, device):
        with open(vocab_path, encoding="utf-8") as f:
            vocab = json.load(f)
        self.ids = {t: i for i, t in enumerate(vocab) if len(t) == 1}
        self.block = block  # rows are padded to the model's block size
        self.row_limit = block  # but hold at most this many real tokens (the engine's row length)
        self.device = device

    def encode(self, text):
        return [self.ids.get(c, self.ids.get(c.lower(), UNK)) for c in text]

    @torch.no_grad()
    def rerank_exact(self, model, lines, batch=64):
        """Sentence exact rate when the engine's rerank uses `model`."""
        model.eval()
        rows, meta = [], []
        for n, (expected, context, readings) in enumerate(lines):
            ctx = self.encode(context)
            for k, (text, score, cost) in enumerate(readings):
                cand = self.encode(text)
                limit = min(self.row_limit, self.block)
                head = ([BOS] + ctx)[-(limit - len(cand)):] if len(cand) < limit else [BOS]
                row = (head + cand)[:limit]
                start = len(head)
                rows.append(row)
                meta.append((n, k, start, len(cand)))
        lm = [0.0] * len(rows)
        for b in range(0, len(rows), batch):
            chunk = rows[b : b + batch]
            x = torch.zeros((len(chunk), self.block), dtype=torch.long)
            for i, row in enumerate(chunk):
                x[i, : len(row)] = torch.tensor(row)
            logp = F.log_softmax(model(x.to(self.device)).float(), -1).cpu()
            for i in range(len(chunk)):
                n, k, start, length = meta[b + i]
                total = 0.0
                for t in range(length):
                    pos = start + t
                    if pos >= self.block:
                        break
                    total += logp[i, pos - 1, chunk[i][pos]].item()
                lm[b + i] = total
        exact = 0
        best_scores = {}
        for (n, k, _, _), value in zip(meta, lm):
            text, score, cost = lines[n][2][k]
            final = LATTICE_WEIGHT * (score + cost) + CONTEXT_WEIGHT * value - (LATTICE_WEIGHT + CONTEXT_WEIGHT) * cost
            if n not in best_scores or final > best_scores[n][0]:
                best_scores[n] = (final, text)
        for n, (expected, _, _) in enumerate(lines):
            if n in best_scores and best_scores[n][1] == expected:
                exact += 1
        return 100 * exact / max(1, len(lines))

    @torch.no_grad()
    def perplexity(self, model, lines):
        model.eval()
        total, count = 0.0, 0
        for expected, context, _ in lines:
            ctx = self.encode(context)
            cand = self.encode(expected)
            head = ([BOS] + ctx)[-(self.block - len(cand)):]
            row = head + cand
            x = torch.tensor([row + [0] * (self.block - len(row))]).to(self.device)
            logp = F.log_softmax(model(x).float(), -1)[0].cpu()
            for t in range(len(cand)):
                pos = len(head) + t
                total -= logp[pos - 1, row[pos]].item()
                count += 1
        return math.exp(total / max(1, count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--save", default=None, help="write the tuned checkpoint here")
    parser.add_argument("--tune", default="all", help="all | top2 (last two blocks + final norm) | ln (layer norms only)")
    parser.add_argument("--replay", default=None, help="uint16 token stream of general text mixed into every batch")
    parser.add_argument("--replay-fraction", type=float, default=0.5)
    args = parser.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, config = load(args.model, device)
    harness = Harness(args.vocab, config.block_size, device)
    person = read_dump(args.eval)
    control = read_dump(args.control)
    with open(args.train, encoding="utf-8") as f:
        stream = torch.tensor(harness.encode(f.read()), dtype=torch.long)
    print(f"device {device}; {len(person)} personal sentences, {len(control)} control sentences, {len(stream):,} training tokens", flush=True)

    def report(step):
        started = time.time()
        a = harness.rerank_exact(model, person)
        b = harness.rerank_exact(model, control)
        p = harness.perplexity(model, person)
        print(f"step {step:4d}  personal exact {a:.2f}%  control exact {b:.2f}%  personal perplexity {p:.2f}  ({time.time() - started:.0f}s)", flush=True)

    report(0)
    # Which weights move: everything, the top of the network, or only the layer norms.
    if args.tune == "all":
        parameters = list(model.parameters())
    else:
        names = [n for n, _ in model.named_parameters()]
        blocks = sorted({int(n.split(".")[1]) for n in names if n.startswith("blocks.")})
        chosen = []
        for n, p in model.named_parameters():
            if args.tune == "ln":
                keep = n.endswith("ln.weight") or n.endswith("ln.bias") or ".ln1." in n or ".ln2." in n
            else:  # top2
                keep = n.startswith("ln.") or any(n.startswith(f"blocks.{b}.") for b in blocks[-2:])
            p.requires_grad_(keep)
            if keep:
                chosen.append(p)
        parameters = chosen
        print(f"tuning {sum(p.numel() for p in parameters) / 1e6:.2f}M of {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters", flush=True)
    replay = None
    if args.replay:
        replay = torch.from_numpy(np.array(np.memmap(args.replay, dtype="<u2", mode="r")[:20_000_000]).astype(np.int64))
        print(f"replaying {len(replay):,} general tokens, fraction {args.replay_fraction}", flush=True)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.01)
    rng = random.Random(1)
    block = config.block_size
    for step in range(1, args.steps + 1):
        model.train()
        personal_count = args.batch if replay is None else max(1, round(args.batch * (1 - args.replay_fraction)))
        windows = [stream[s : s + block + 1] for s in (rng.randrange(0, len(stream) - block - 1) for _ in range(personal_count))]
        if replay is not None:
            windows += [replay[s : s + block + 1] for s in (rng.randrange(0, len(replay) - block - 1) for _ in range(args.batch - personal_count))]
        rows = torch.stack(windows).to(device)
        x, y = rows[:, :-1], rows[:, 1:]
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            print(f"step {step:4d}  loss {loss.item():.3f}", flush=True)
            report(step)
    if args.save:
        torch.save({"config": config.__dict__, "model": model.state_dict()}, args.save)
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
