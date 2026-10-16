#!/usr/bin/env python3
"""Verify the reading-conditioned model as the lattice's ranker, in Python.

    Data/venv-train/bin/python Training/eval_reading_lm.py --model Data/rlm1/model.pt --eval Data/eval_dialogue.tsv --limit 300

For every sentence (full pinyin given, syllable by syllable) the lexicon lists the words each span
of syllables can be; a beam search over the positions scores every (hypothesis, word) pair with
the model - log P(character | context, text so far, reading) summed over the word's characters -
batched per position, nothing else: no word frequencies, no pair statistics, no hand-set costs
except that a lone word with no context is left to the lexicon (the standard). Reports the exact
first-reading rate, the time per model call, and the calls per sentence.
"""
import argparse
import os
import sqlite3
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Training"))
from reading_lm import ReadingLM  # noqa: E402

LEXICON = os.path.join(ROOT, "Data", "lexicon", "lexicon.sqlite")
MAX_WORD = 4
BOS = 1


class Searcher:
    def __init__(self, path, device, beam=8, pairs=64, context_chars=40, lexicon_weight=0.0):
        checkpoint = torch.load(path, map_location="cpu")
        self.model = ReadingLM(**checkpoint["config"])
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device).eval()
        self.device = device
        self.block = checkpoint["config"]["block_size"]
        self.syllable_id = {s: i + 1 for i, s in enumerate(checkpoint["syllables"])}
        import json
        with open(os.path.join(os.path.dirname(path), "vocab.json"), encoding="utf-8") as f:
            vocab = json.load(f)
        self.ids = {t: i for i, t in enumerate(vocab) if len(t) == 1}
        self.beam = beam
        self.pairs = pairs
        self.context_chars = context_chars
        self.lexicon_weight = lexicon_weight
        self.db = sqlite3.connect(LEXICON)
        self.cache = {}
        self.calls = 0
        self.rows = 0
        self.model_seconds = 0.0

    def words(self, key):
        if key in self.cache:
            return self.cache[key]
        out = [(w, lp) for (w, lp) in self.db.execute("SELECT word, logp FROM lexicon WHERE key = ? ORDER BY logp DESC LIMIT 60", (key,))]
        self.cache[key] = out
        return out

    @torch.no_grad()
    def score(self, prefix_tokens, items):
        """items: list of (text_tokens_so_far, word, syllables). Returns log P(word | prefix + so far)."""
        rows, reads, spans = [], [], []
        for so_far, word, syllables in items:
            word_ids = [self.ids.get(c, 2) for c in word]
            syl_ids = [self.syllable_id.get(s, 0) for s in syllables]
            head = (prefix_tokens + so_far)[-(self.block - len(word_ids)) :]
            row = head + word_ids
            # next_readings[t] is the reading of row[t+1]; only the word's characters are scored.
            nxt = [0] * len(row)
            for k in range(len(word_ids)):
                nxt[len(head) - 1 + k] = syl_ids[k]
            rows.append(row)
            reads.append(nxt)
            spans.append((len(head) - 1, len(word_ids), word_ids))
        t = self.block
        x = torch.zeros((len(rows), t), dtype=torch.long)
        r = torch.zeros((len(rows), t), dtype=torch.long)
        for i, (row, nxt) in enumerate(zip(rows, reads)):
            x[i, : len(row)] = torch.tensor(row)
            r[i, : len(nxt)] = torch.tensor(nxt)
        started = time.time()
        logp = torch.log_softmax(self.model(x.to(self.device), r.to(self.device)).float(), -1).cpu()
        self.model_seconds += time.time() - started
        self.calls += 1
        self.rows += len(rows)
        out = []
        for i, (start, length, word_ids) in enumerate(spans):
            out.append(sum(logp[i, start + k, word_ids[k]].item() for k in range(length)))
        return out

    def decode(self, syllables, context):
        n = len(syllables)
        prefix = [BOS] + [self.ids.get(c, 2) for c in context[-self.context_chars :]]
        # Edges: (start, end, word, syllables)
        edges = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, min(n, i + MAX_WORD) + 1):
                key = "'".join(syllables[i:j])
                for word, lp in self.words(key):
                    edges[i].append((j, word, syllables[i:j], lp))
        # Hypotheses per position: (score, text, tokens)
        hyps = [[] for _ in range(n + 1)]
        hyps[0] = [(0.0, "", [])]
        for i in range(n):
            if not hyps[i]:
                continue
            best = {}
            for score, text, toks in hyps[i]:
                if text not in best or best[text][0] < score:
                    best[text] = (score, text, toks)
            hyps[i] = sorted(best.values(), key=lambda h: -h[0])[: self.beam]
            items, meta = [], []
            for h, (score, text, toks) in enumerate(hyps[i]):
                for end, word, syls, lp in edges[i]:
                    items.append((toks, word, syls))
                    meta.append((h, end, word, lp))
            if not items:
                continue
            scores = []
            for s in range(0, len(items), self.pairs):
                scores += self.score(prefix, items[s : s + self.pairs])
            for (h, end, word, lp), lm in zip(meta, scores):
                score, text, toks = hyps[i][h]
                hyps[end].append((score + lm + self.lexicon_weight * lp, text + word, toks + [self.ids.get(c, 2) for c in word]))
        final = sorted({h[1]: h for h in hyps[n]}.values(), key=lambda h: -h[0])
        return [h[1] for h in final]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--lexicon-weight", type=float, default=0.0)
    parser.add_argument("--show-failures", action="store_true")
    args = parser.parse_args()
    device = "cpu" if args.cpu or not torch.backends.mps.is_available() else "mps"
    searcher = Searcher(args.model, device, beam=args.beam, lexicon_weight=args.lexicon_weight)
    total = exact = top16 = 0
    started = time.time()
    with open(args.eval, encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                continue
            syllables = fields[0].split()
            expected = fields[1]
            context = "" if args.no_context or len(fields) < 3 else fields[2].replace("\\n", "\n")
            if len(syllables) != len(expected):
                continue
            readings = searcher.decode(syllables, context)
            total += 1
            if readings and readings[0] == expected:
                exact += 1
            elif args.show_failures:
                print(f"{expected}\t{readings[0] if readings else ''}")
            if expected in readings[:16]:
                top16 += 1
            if total == args.limit:
                break
    elapsed = time.time() - started
    print(f"sentences: {total}  exact: {100 * exact / max(1, total):.2f}%  among best 16: {100 * top16 / max(1, total):.2f}%")
    print(f"model calls per sentence: {searcher.calls / max(1, total):.1f}  rows per call: {searcher.rows / max(1, searcher.calls):.0f}  ms per call: {1000 * searcher.model_seconds / max(1, searcher.calls):.1f}  decode s per sentence: {elapsed / max(1, total):.2f}")


if __name__ == "__main__":
    main()
