#!/usr/bin/env python3
"""Takes out what reads least like written Chinese, judged by a language model, not by word lists.

    P=Data/venv-train/bin/python
    $P Training/quality_filter.py --data Data/prepared-clean --model Data/clean3/model.pt score web mixed news forum
    $P Training/quality_filter.py --data Data/prepared-clean show web          # documents at each level, to choose the cut
    $P Training/quality_filter.py --data Data/prepared-clean filter web=3 mixed=2

score: every document of the sources (and of their held-out files) gets the model's mean loss per
token over its first WINDOWS windows of 64 tokens: gambling and SEO pages strung from keywords,
machine-made text and garbled text score far worse than any written text. Documents shorter
than MIN_TOKENS keep no score (too few tokens to judge; most are chat). Writes <source>.quality.npy
and holdout_<source>.quality.npy.
show: prints documents from the best to the worst per mille, to see where the junk begins.
filter: source=percent drops that share of the scored documents, the worst; the held-out documents
above the same loss go too, and val.bin is written again from the held-out files. The score files
are cut the same way, so they stay one score per document; scores that do not match the documents
stop the run. Appends to quality.log.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import prepare_data as P
from model import GPT, Config

WINDOWS = 4
MIN_TOKENS = 200
BATCH = 512
ROWS = 64


def documents(data):
    starts = np.flatnonzero(data == P.BOS)
    return starts, np.append(starts[1:], len(data))


def holdout_docs(directory, name, encode):
    path = os.path.join(directory, f"holdout_{name}.txt")
    if not os.path.exists(path):
        return [], []
    with open(path, encoding="utf-8") as f:
        texts = [t for t in f.read().split("\n\n") if t.strip()]
    return texts, [encode(P.codepoints(t)) for t in texts]


def score_docs(model, device, docs):
    """Mean loss per token over the first windows of each document (NaN for short ones)."""
    scores = np.full(len(docs), np.nan, dtype=np.float32)
    rows, owners = [], []

    def flush():
        if not rows:
            return
        x = torch.tensor(np.stack(rows)[:, :ROWS].astype(np.int64), device=device)
        y = torch.tensor(np.stack(rows)[:, 1:].astype(np.int64), device=device)
        with torch.no_grad():
            loss = F.cross_entropy(model(x).float().reshape(-1, model.tokens.weight.shape[0]), y.reshape(-1), ignore_index=P.PAD,
                                   reduction="none").reshape(len(rows), ROWS)
        counts = (y != P.PAD).sum(1).cpu().numpy()
        sums = loss.sum(1).cpu().numpy()
        for owner, s, c in zip(owners, sums, counts):
            totals[owner] += s
            tokens[owner] += c
        rows.clear()
        owners.clear()

    totals = np.zeros(len(docs))
    tokens = np.zeros(len(docs))
    for i, doc in enumerate(docs):
        if len(doc) < MIN_TOKENS:
            continue
        for w in range(WINDOWS):
            piece = np.asarray(doc[w * ROWS:w * ROWS + ROWS + 1])
            if len(piece) < 2:
                break
            if len(piece) < ROWS + 1:
                piece = np.concatenate((piece, np.zeros(ROWS + 1 - len(piece), dtype=piece.dtype)))
            rows.append(piece)
            owners.append(i)
            if len(rows) == BATCH:
                flush()
    flush()
    judged = tokens > 0
    scores[judged] = totals[judged] / tokens[judged]
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model")
    parser.add_argument("command", choices=["score", "show", "filter"])
    parser.add_argument("items", nargs="+")
    args = parser.parse_args()
    with open(os.path.join(args.data, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    P.KEEP_CASE = "A" in vocab
    encode = P.encoder(P.token_table(vocab))
    lut = np.array(["⏎" if t == "\n" else "" if i < 3 else t for i, t in enumerate(vocab)], dtype=object)

    if args.command == "score":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        checkpoint = torch.load(args.model, map_location="cpu")
        model = GPT(Config(**checkpoint["config"])).to(device).eval().half()
        model.load_state_dict(checkpoint["model"])
        for name in args.items:
            data = np.memmap(os.path.join(args.data, f"{name}.bin"), dtype="<u2", mode="r")
            starts, ends = documents(data)
            scores = score_docs(model, device, [data[s:e] for s, e in zip(starts, ends)])
            np.save(os.path.join(args.data, f"{name}.quality.npy"), scores)
            _, held = holdout_docs(args.data, name, encode)
            np.save(os.path.join(args.data, f"holdout_{name}.quality.npy"), score_docs(model, device, held))
            judged = scores[~np.isnan(scores)]
            print(f"{name}: {len(scores)} documents, {len(judged)} scored; loss per mille 500/900/990: "
                  f"{np.percentile(judged, 50):.2f} / {np.percentile(judged, 90):.2f} / {np.percentile(judged, 99):.2f}", flush=True)

    elif args.command == "show":
        for name in args.items:
            data = np.memmap(os.path.join(args.data, f"{name}.bin"), dtype="<u2", mode="r")
            starts, ends = documents(data)
            scores = np.load(os.path.join(args.data, f"{name}.quality.npy"))
            judged = np.flatnonzero(~np.isnan(scores))
            order = judged[np.argsort(scores[judged])]
            rng = np.random.default_rng(1)
            for permille in [500, 800, 900, 950, 970, 980, 990, 995, 999]:
                at = int(len(order) * permille / 1000)
                print(f"== {name} {permille / 10:.1f}% (loss {scores[order[min(at, len(order) - 1)]]:.2f})")
                for i in rng.choice(order[max(0, at - 50):at + 50], 3, replace=False):
                    print("   ", "".join(lut[np.asarray(data[starts[i]:min(ends[i], starts[i] + 90)])]))

    else:
        log = open(os.path.join(args.data, "quality.log"), "a", encoding="utf-8")
        for item in args.items:
            name, percent = item.split("=")
            path = os.path.join(args.data, f"{name}.bin")
            data = np.memmap(path, dtype="<u2", mode="r")
            starts, ends = documents(data)
            scores_path = os.path.join(args.data, f"{name}.quality.npy")
            scores = np.load(scores_path)
            if len(scores) != len(starts):
                raise SystemExit(f"{name}: {len(scores)} scores for {len(starts)} documents; score them again")
            judged = scores[~np.isnan(scores)]
            cut = np.percentile(judged, 100 - float(percent))
            drop = np.nan_to_num(scores, nan=-np.inf) > cut
            with open(path + ".quality", "wb") as out:
                for c in range(0, len(starts), 200_000):
                    pieces = [np.asarray(data[s:e]) for s, e, d in zip(starts[c:c + 200_000], ends[c:c + 200_000], drop[c:c + 200_000]) if not d]
                    if pieces:
                        out.write(np.concatenate(pieces).astype("<u2").tobytes())
            after = os.path.getsize(path + ".quality") // 2
            del data
            texts, _ = holdout_docs(args.data, name, encode)
            held_path = os.path.join(args.data, f"holdout_{name}.quality.npy")
            held_scores = np.load(held_path)
            if len(held_scores) != len(texts):
                os.remove(path + ".quality")
                raise SystemExit(f"{name}: {len(held_scores)} scores for {len(texts)} held-out documents; score them again")
            held_keep = ~(np.nan_to_num(held_scores, nan=-np.inf) > cut)
            kept = [t for t, k in zip(texts, held_keep) if k]
            # The scores go with the documents they belong to, so a later filter finds them in step.
            os.replace(path + ".quality", path)
            np.save(scores_path, scores[~drop])
            with open(os.path.join(args.data, f"holdout_{name}.txt"), "w", encoding="utf-8") as f:
                f.write("".join(t + "\n\n" for t in kept))
            np.save(held_path, held_scores[held_keep])
            line = (f"{name}: dropped the worst {percent}% of scored documents (loss above {cut:.2f}): {int(drop.sum())} of {len(starts)},"
                    f" {len(drop) and 100 * (1 - after / ends[-1]):.1f}% of the tokens; held out {len(texts) - len(kept)} of {len(texts)}")
            print(line, flush=True)
            log.write(line + "\n")
        log.close()
        P.clean_holdouts(args.data)  # val.bin from the held-out files as they are now


if __name__ == "__main__":
    main()
