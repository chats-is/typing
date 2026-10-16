#!/usr/bin/env python3
"""Measure pinyin-to-text accuracy when the language model generates the text itself.

    Data/venv-train/bin/python Training/eval_generation.py Data/lm/model.pt [--limit 1000]

No word lexicon is used: for each syllable the model may only pick characters that have that
reading, and a beam search keeps the best partial sentences. The only data besides the model is
a character-to-pinyin table (taken here from the single-character entries of lexicon.sqlite).
The syllable boundaries come from the evaluation file, so segmentation ambiguity is not tested.
"""
import argparse
import json
import os
import sqlite3
import time

import torch
import torch.nn.functional as F

from model import GPT, Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEAM = 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--eval", default=os.path.join(ROOT, "Data", "eval.tsv"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    with open(os.path.join(ROOT, "Data", "lm", "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    ids = {token: i for i, token in enumerate(vocab)}

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = Config(**checkpoint["config"])
    model = GPT(config).to(args.device).eval()
    model.load_state_dict(checkpoint["model"])
    print(f"checkpoint at step {checkpoint['step']}/{checkpoint['total_steps']}", flush=True)

    # syllable -> ids of the characters that can be read that way
    readings = {}
    db = sqlite3.connect(os.path.join(ROOT, "Resources", "lexicon.sqlite"))
    for key, word in db.execute("SELECT key, word FROM lexicon WHERE n = 1"):
        if word in ids:
            readings.setdefault(key, set()).add(ids[word])
    db.close()
    allowed = {key: torch.tensor(sorted(chars), device=args.device) for key, chars in readings.items()}

    start = ids["\n"]
    sentences = exact = characters = correct = within = 0
    seconds = 0.0
    with open(args.eval, encoding="utf-8") as f, torch.no_grad():
        for line in list(f)[: args.limit]:
            pinyin, expected = line.rstrip("\n").split("\t")
            began = time.time()
            beams = [([start], 0.0)]
            for position, syllable in enumerate(pinyin.split(" ")):
                choices = allowed.get(syllable)
                if choices is None:
                    beams = []
                    break
                rows = torch.zeros((len(beams), config.block_size), dtype=torch.long, device=args.device)
                for r, (tokens, _) in enumerate(beams):
                    rows[r, : len(tokens)] = torch.tensor(tokens, device=args.device)
                logp = F.log_softmax(model(rows)[:, position, :], dim=-1)[:, choices]
                scores = logp + torch.tensor([score for _, score in beams], device=args.device).unsqueeze(1)
                top = torch.topk(scores.flatten(), min(BEAM, scores.numel()))
                beams = [
                    (beams[index // len(choices)][0] + [int(choices[index % len(choices)])], float(value))
                    for value, index in zip(top.values.tolist(), top.indices.tolist())
                ]
            seconds += time.time() - began
            texts = ["".join(vocab[t] for t in tokens[1:]) for tokens, _ in beams]
            best = texts[0] if texts else ""

            sentences += 1
            characters += len(expected)
            exact += best == expected
            within += expected in texts
            if len(best) == len(expected):
                correct += sum(a == b for a, b in zip(best, expected))
            if sentences % 200 == 0:
                print(f"{sentences} sentences  exact {100 * exact / sentences:.2f}%", flush=True)

    print(f"sentences: {sentences}  exact: {100 * exact / sentences:.2f}%  characters: {100 * correct / characters:.2f}%"
          f"  among best {BEAM}: {100 * within / sentences:.2f}%")
    print(f"mean {1000 * seconds / sentences:.0f} ms per sentence, {1000 * seconds / characters:.1f} ms per syllable on {args.device}")


if __name__ == "__main__":
    main()
