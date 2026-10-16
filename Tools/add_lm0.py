#!/usr/bin/env python3
"""Fill the lexicon's lm0 column: what the language model gives each word with no context.

    Data/venv-train/bin/python Tools/add_lm0.py --model Data/lm9/model.pt --lexicon Data/lexicon/lexicon.sqlite

lm0(word) = log P(word | <bos>) under the language model. The decoder ranks a sentence by the
lexicon's own frequencies plus what the context adds, log P(words | context) - sum lm0(word), so
with nothing before the cursor a lone word is ordered by the lexicon alone, as the standard asks.
"""
import argparse
import json
import os
import sqlite3
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Training"))
from model import GPT, Config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(ROOT, "Data", "lm9", "model.pt"))
    parser.add_argument("--lexicon", default=os.path.join(ROOT, "Data", "lexicon", "lexicon.sqlite"))
    parser.add_argument("--batch", type=int, default=1024)
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    config = Config(**checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device).eval()
    with open(os.path.join(os.path.dirname(args.model), "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    ids = {t: i for i, t in enumerate(vocab)}
    bos = ids["<bos>"]
    block = config.block_size

    db = sqlite3.connect(args.lexicon)
    words = [w for (w,) in db.execute("SELECT DISTINCT word FROM lexicon")]
    known = set(words)
    words += [w for (w,) in db.execute("SELECT DISTINCT word FROM english") if w not in known]
    words = [w for w in words if all(c in ids for c in w) and len(w) < block - 1]
    cache = os.path.join(os.path.dirname(args.lexicon), "lm0.tsv")
    cached = {}
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            for line in f:
                word, value = line.rstrip("\n").split("\t")[:2]
                cached[word] = float(value)
    words = [w for w in words if w not in cached]
    print(f"{len(words):,} words to score ({len(cached):,} cached)", flush=True)

    started = time.time()
    results = []
    with torch.no_grad():
        for start in range(0, len(words), args.batch):
            batch = words[start : start + args.batch]
            rows = torch.zeros((len(batch), block), dtype=torch.long)
            for r, w in enumerate(batch):
                rows[r, 0] = bos
                for t, c in enumerate(w):
                    rows[r, t + 1] = ids[c]
            rows = rows.to(device)
            logp = torch.log_softmax(model(rows, True).float(), -1)
            targets = rows[:, 1:]
            picked = logp[:, :-1].gather(2, targets[..., None])[..., 0]
            mask = targets > 0
            total = (picked * mask).sum(1).cpu().tolist()
            results += zip(total, batch)
            if (start // args.batch) % 50 == 0:
                print(f"{start:,} ({time.time() - started:.0f}s)", flush=True)
    for value, word in results:
        cached[word] = value
    with open(cache, "w", encoding="utf-8") as f:
        for word, value in cached.items():
            f.write(f"{word}\t{value:.4f}\n")
    # Only the newly scored words need writing: Tools/build_lexicon.py fills the rest from the cache.
    db.executemany("UPDATE lexicon SET lm0 = ? WHERE word = ?", results)
    try:
        db.execute("ALTER TABLE english ADD COLUMN lm0 REAL NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    db.executemany("UPDATE english SET lm0 = ? WHERE word = ?", results)
    db.commit()
    db.close()
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
