#!/usr/bin/env python3
"""Evaluate a keys-to-text model (make_ime_data.py, train.py) on the input method's eval sets.

    Data/venv-train/bin/python Training/eval_ime.py --model Data/ime1/model.pt --eval Data/eval_dialogue.tsv --limit 300

Each eval line is "typed keys<TAB>expected text<TAB>text before the cursor" (spaces in the keys are
dropped; "\\n" in the context is a line break). The search knows nothing about pinyin: it only
keeps the key tokens equal to what was typed. After the keys of a chunk the model writes the text
for them, and the best complete readings are the candidates.

A keys-first model (make_ime_data.py --layout keys-first) reads all the keys, then writes the text in
one piece until <end>; the search is a plain beam over the text.

A reading longer than the model's 64-token window keeps its last chunks as keys and text; the
earlier ones move into the text before the cursor, as when they had been typed and committed.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPT, Config  # noqa: E402

# The window comes from the model; text before the cursor and typed keys share it as in training
# (make_ime_data.py --context): a 64-token model keeps 20 characters of context and 40 tokens of body.
CONTEXT_CHARS = {64: 20, 128: 64}
KEYS_FIRST_CONTEXT = 48
OUTPUTS_PER_CHUNK = 8  # --outputs


class Searcher:
    def __init__(self, path, device, beam, context_chars=None):
        checkpoint = torch.load(path, map_location="cpu")
        config = Config(**checkpoint["config"])
        self.block = config.block_size
        self.context_chars = context_chars if context_chars is not None else CONTEXT_CHARS.get(self.block, self.block // 3)
        self.body_tokens = self.block - CONTEXT_CHARS.get(self.block, self.block // 3) - 4
        self.model = GPT(config)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device).eval()
        self.device = device
        self.beam = beam
        with open(os.path.join(os.path.dirname(path), "vocab.json"), encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.ids = {token: index for index, token in enumerate(self.vocab)}
        self.key = {token[1:]: index for index, token in enumerate(self.vocab) if token.startswith("⌨")}
        self.type = self.ids["<type>"]
        self.keys_first = "<out>" in self.ids
        if self.keys_first:
            self.out, self.end = self.ids["<out>"], self.ids["<end>"]
            self.context_chars = context_chars if context_chars is not None else KEYS_FIRST_CONTEXT
        self.space = self.ids[" "]
        output = torch.ones(len(self.vocab), dtype=torch.bool)
        output[:3] = False  # <pad> <bos> <unk>
        output[self.ids["\n"]] = False
        output[list(self.key.values())] = False
        output[self.type] = False
        if self.keys_first:
            output[self.out] = False  # <end> stays: it finishes a text
        self.not_output = (~output).to(device)
        self.calls = 0
        self.outputs_per_chunk = OUTPUTS_PER_CHUNK

    def row(self, context, pairs, pending):
        """Tokens for a hypothesis: committed pairs beyond the window move into the context."""
        body_pairs = list(pairs)
        def body_of(ps):
            tokens = []
            for keys, text in ps:
                tokens += [self.key[k] for k in keys] + [self.ids.get(c, 2) for c in text]
            return tokens + [self.key[k] for k in pending]
        body = body_of(body_pairs)
        moved = ""
        while len(body) > self.body_tokens and body_pairs:
            moved += body_pairs.pop(0)[1]
            body = body_of(body_pairs)
        text = (context + moved)[-self.context_chars:] if self.context_chars else ""
        tokens = [1] + [self.ids.get(c, 2) for c in text] + [self.type] + body
        return tokens[-self.block:]

    @torch.no_grad()
    def logprobs(self, rows):
        self.calls += 1
        batch = torch.zeros((len(rows), self.block), dtype=torch.long)
        for r, tokens in enumerate(rows):
            batch[r, : len(tokens)] = torch.tensor(tokens)
        logits = self.model(batch.to(self.device)).float()
        last = torch.tensor([len(t) - 1 for t in rows], device=self.device)
        return torch.log_softmax(logits[torch.arange(len(rows), device=self.device), last], dim=-1)

    def expand(self, hyps):
        """Next-token log-probabilities for each hypothesis (score, pairs, pending)."""
        out = []
        for start in range(0, len(hyps), 64):
            chunk = hyps[start : start + 64]
            out.append(self.logprobs([self.row(self.context, h[1], h[2]) for h in chunk]))
        return torch.cat(out) if out else None

    def outputs(self, logp, k):
        masked = logp.masked_fill(self.not_output, float("-inf"))
        values, indices = masked.topk(k)
        return [(v, self.vocab[i]) for v, i in zip(values.tolist(), indices.tolist())]

    def search_keys_first(self, keys, context):
        """Texts for `keys`, best first, as (score, text): a beam over the text after all the keys."""
        longest = len(keys) + 4
        room = max(0, min(self.context_chars, self.block - 3 - len(keys) - longest))
        prefix = [1] + ([self.ids.get(c, 2) for c in context[-room:]] if room else []) + [self.type] + [self.key[k] for k in keys] + [self.out]
        hyps, finals = [(0.0, [])], []
        for _ in range(longest):
            logp = []
            for start in range(0, len(hyps), 64):
                logp.append(self.logprobs([prefix + tokens for _, tokens in hyps[start : start + 64]]))
            logp = torch.cat(logp).masked_fill(self.not_output, float("-inf"))
            values, indices = logp.topk(self.beam)
            grown = []
            for (score, tokens), vs, ts in zip(hyps, values.tolist(), indices.tolist()):
                for v, t in zip(vs, ts):
                    if t == self.end:
                        finals.append((score + v, tokens))
                    else:
                        grown.append((score + v, tokens + [t]))
            hyps = sorted(grown, key=lambda h: -h[0])[: self.beam]
            finals = sorted(finals, key=lambda h: -h[0])[: self.beam]
            # Scores only fall, so nothing left can beat a full list of finished texts.
            if not hyps or (len(finals) == self.beam and hyps[0][0] < finals[-1][0]):
                break
        seen, results = set(), []
        for score, tokens in finals:
            text = "".join(self.vocab[t] for t in tokens)
            if text not in seen:
                seen.add(text)
                results.append((score, text))
        return results

    def search(self, keys, context):
        """Complete readings of `keys`, best first, as (score, text)."""
        if self.keys_first:
            return self.search_keys_first(keys, context)
        self.context = context
        n = len(keys)
        prune = lambda hyps: sorted(hyps, key=lambda h: -h[0])[: self.beam]
        chunk = [[] for _ in range(n + 1)]  # keys typed, text not yet written; i keys consumed
        written = [[] for _ in range(n + 1)]  # the last chunk's text written
        written[0] = [(0.0, (), "")]
        finals = []
        for i in range(n + 1):
            hyps = prune(chunk[i])
            if hyps:
                logp = self.expand(hyps)
                spaced = []
                for h, lp in zip(hyps, logp):
                    score, pairs, pending = h
                    if i < n:
                        chunk[i + 1].append((score + lp[self.key[keys[i]]].item(), pairs, pending + keys[i]))
                    for value, token in self.outputs(lp, self.outputs_per_chunk):
                        if token == " ":
                            spaced.append((score + value, pairs, pending))
                        else:
                            written[i].append((score + value, pairs + ((pending, token),), ""))
                # A space before an English word: the word's first letter follows.
                spaced = prune(spaced)
                if spaced:
                    logp = self.expand([(s, p, k) for s, p, k in spaced])
                    for (score, pairs, pending), lp in zip(spaced, logp):
                        for value, token in self.outputs(lp, 3):
                            if token != " ":
                                written[i].append((score + value, pairs + ((pending, " " + token),), ""))
            hyps = prune(written[i])
            if i == n:
                finals = hyps
                break
            if hyps:
                logp = self.expand(hyps)
                for (score, pairs, _), lp in zip(hyps, logp):
                    chunk[i + 1].append((score + lp[self.key[keys[i]]].item(), pairs, keys[i]))
        seen, results = set(), []
        for score, pairs, _ in finals:
            text = "".join(t for _, t in pairs)
            if text not in seen:
                seen.add(text)
                results.append((score, text))
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Data/ime1/model.pt")
    parser.add_argument("--eval", action="append", required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--beam", type=int, default=16)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--show-failures", type=int, default=0)
    parser.add_argument("--outputs", type=int, default=OUTPUTS_PER_CHUNK, help="texts tried after each key chunk")
    parser.add_argument("--context", type=int, default=None, help="characters of text before the cursor (default: as trained)")
    parser.add_argument("--by-length", action="store_true", help="also report accuracy by sentence length")
    args = parser.parse_args()
    searcher = Searcher(args.model, args.device, args.beam, args.context)
    searcher.outputs_per_chunk = args.outputs
    for path in args.eval:
        total = exact = top9 = 0
        by_pattern = {}
        by_length = {}
        shown = 0
        started = time.time()
        calls = searcher.calls
        with open(path, encoding="utf-8") as f:
            lines = [line.rstrip("\n").split("\t") for line in f if line.strip()][: args.limit]
        for fields in lines:
            keys = fields[0].replace(" ", "")
            expected = fields[1]
            context = fields[2].replace("\\n", "\n") if len(fields) > 2 else ""
            if any(k not in searcher.key for k in keys):
                continue
            results = searcher.search(keys, context)
            texts = [t for _, t in results]
            total += 1
            hit = bool(texts) and texts[0] == expected
            exact += hit
            top9 += expected in texts[:9]
            bucket = by_length.setdefault("≤8" if len(expected) <= 8 else "9-12" if len(expected) <= 12 else "13+", [0, 0])
            bucket[0] += 1
            bucket[1] += hit
            if len(fields) > 3:
                p = by_pattern.setdefault(fields[3], [0, 0])
                p[0] += 1
                p[1] += hit
            if not hit and shown < args.show_failures:
                shown += 1
                print(f"  {keys}\t{expected}\t{' / '.join(texts[:3])}")
        elapsed = time.time() - started
        patterns = "  ".join(f"{name}: {100 * hit / count:.1f}% of {count}" for name, (count, hit) in sorted(by_pattern.items()))
        if args.by_length:
            patterns += "  by length " + "  ".join(f"{name}: {100 * hit / count:.1f}% of {count}" for name, (count, hit) in by_length.items())
        print(f"{os.path.basename(path)}  sentences: {total}  exact: {100 * exact / max(1, total):.2f}%  first page: {100 * top9 / max(1, total):.2f}%"
              f"  {patterns}  ({elapsed / max(1, total):.2f}s, {(searcher.calls - calls) / max(1, total):.0f} calls per sentence)", flush=True)


if __name__ == "__main__":
    main()
