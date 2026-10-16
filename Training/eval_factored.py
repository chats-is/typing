#!/usr/bin/env python3
"""Evaluate the language model and the reading model (reading_model.py) together.

    Data/venv-train/bin/python Training/eval_factored.py --model Data/lm3/model.pt --reading Data/reading/model.pt \
        --eval Data/eval_dialogue.tsv --limit 300

Same options and output as eval_ime.py. The search cuts the typed keys into spans of one
character each; a span's text is scored by the language model, log P(character | text so far),
plus the reading model, log P(keys | character) = log P(character | keys) + log P(keys) - log P(character).
--reading-weight scales the reading part; --reading-floor drops characters that fit the keys worse than it.
With a key model (key_model.py) as --reading, a span is scored as log P(keys | character) for the
characters the language model proposes (--candidates per hypothesis) and those in the model's readings index.
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_ime  # noqa: E402
from model import GPT, Config  # noqa: E402
from key_model import KeyModel  # noqa: E402
from reading_model import MAX_KEYS, ReadingModel  # noqa: E402

READING = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "reading", "model.pt")
READING_WEIGHT = 1.0
READING_FLOOR = None  # --reading-floor: characters whose log P(keys | character) - log P(keys) is below it are out
READING_GAP = None  # --reading-gap: characters fitting the keys this many nats worse than the best fit are out
FUZZY_MIN = 2  # --fuzzy-min: slips are only looked for in spans of at least this many keys
FUZZY_INDEX = False  # --fuzzy-index: candidates also from readings one slip away from the span (no language-model proposals needed)
FIRST_KEY = False  # --first-key: a character is tried for a span of 2+ keys only if one of its readings starts with the first key
USE_INDEX = True  # --no-index: candidates from the language model only
CANDIDATES = 100  # --candidates: with a key model (key_model.py), characters per hypothesis the language model proposes


class FactoredSearcher:
    def __init__(self, path, device, beam, context_chars=None):
        checkpoint = torch.load(path, map_location="cpu")
        config = Config(**checkpoint["config"])
        self.block = config.block_size
        self.model = GPT(config)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(device).eval()
        self.device = device
        self.beam = beam
        self.outputs_per_chunk = eval_ime.OUTPUTS_PER_CHUNK
        self.calls = 0
        self.rows_scored = 0
        self.positions = 0
        self.sentences = 0
        FactoredSearcher.last = self
        with open(os.path.join(os.path.dirname(path), "vocab.json"), encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.ids = {token: index for index, token in enumerate(self.vocab)}
        reading = torch.load(READING, map_location="cpu")
        assert reading["config"]["vocab_size"] == len(self.vocab), "the reading model must share the language model's vocabulary"
        config = dict(reading["config"])
        config.setdefault("n_keys", len(reading["key_tokens"]))
        config.setdefault("block", MAX_KEYS + 1)
        self.generative = reading.get("kind") == "keys"
        if self.generative:
            # A key model scores log P(keys | character); its index finds characters by the start of their readings.
            self.reading = KeyModel(**config)
            self.prefix_index = {}
            for text, entries in ((reading["index"] or {}) if USE_INDEX else {}).items():
                for length in range(1, len(text) + 1):
                    self.prefix_index.setdefault(text[:length], set()).update(c for c, _ in entries)
            self.fuzzy_index = {}
            if FUZZY_INDEX:
                from make_ime_data import NEIGHBOURS
                for text, entries in (reading["index"] or {}).items():
                    chars = {c for c, _ in entries}
                    variants = set()
                    for i in range(len(text)):
                        for near in NEIGHBOURS.get(text[i], []):
                            variants.add(text[:i] + near + text[i + 1:])
                        if len(text) > 1:
                            variants.add(text[:i] + text[i + 1:])
                        variants.add(text[:i + 1] + text[i] + text[i + 1:])
                        if i + 1 < len(text) and text[i] != text[i + 1]:
                            variants.add(text[:i] + text[i + 1] + text[i] + text[i + 2:])
                    for variant in variants:
                        for length in range(FUZZY_MIN, len(variant) + 1):
                            self.fuzzy_index.setdefault(variant[:length], set()).update(chars)
                print(f"fuzzy index: {len(self.fuzzy_index)} prefixes", flush=True)
        else:
            self.reading = ReadingModel(**config)
            self.unigram = reading["unigram"].to(device)
        self.reading.load_state_dict(reading["model"])
        self.reading.to(device).eval()
        self.key_ids = {token: index for index, token in enumerate(reading["key_tokens"])}
        self.key = {k: i for k, i in self.key_ids.items() if len(k) == 1}
        # Typed keys only ever stand for Chinese characters and letters.
        not_output = torch.tensor([not (len(t) == 1 and ("\u4e00" <= t <= "\u9fff" or t.isascii() and t.isalpha())) for t in self.vocab])
        self.not_output = not_output.to(device)
        self.not_output_list = not_output.tolist()

    @torch.no_grad()
    def spans(self, keys):
        """For every span keys[i:j] of at most MAX_KEYS: log P(keys) + log P(character | keys) - log P(character)."""
        cut = self.key_ids.get("<cut>")  # a reading model trained with terminators: <end> complete, <cut> still being typed
        n = len(keys)
        entries = []
        for i in range(n):
            for j in range(i + 1, min(n, i + MAX_KEYS) + 1):
                entries.append((i, j, 2))
                if cut is not None and j == n:
                    entries.append((i, j, cut))
        width = self.reading.block
        rows = torch.zeros((len(entries), width), dtype=torch.long)
        last = torch.zeros(len(entries), dtype=torch.long)
        for r, (i, j, terminator) in enumerate(entries):
            row = [1] + [self.key_ids[k] for k in keys[i:j]] + ([terminator] if cut is not None else [])
            rows[r, : len(row)] = torch.tensor(row)
            last[r] = len(row) - 1
        rows, last = rows.to(self.device), last.to(self.device)
        key_logits, char_logits = self.reading(rows)
        key_lp = torch.log_softmax(key_logits.float(), -1)
        positions = torch.arange(width, device=self.device)
        if cut is not None:
            targets = torch.cat([rows[:, 1:], torch.zeros((len(entries), 1), dtype=torch.long, device=self.device)], 1)
            valid = positions[None] < last[:, None]
        else:
            targets = torch.cat([rows[:, 1:], torch.zeros((len(entries), 1), dtype=torch.long, device=self.device)], 1)
            targets = torch.where(positions[None] == last[:, None], torch.full_like(targets, 2), targets)  # <end> after the last key
            valid = positions[None] <= last[:, None]
        span_lp = (key_lp.gather(2, targets[..., None])[..., 0] * valid).sum(1)
        char_lp = torch.log_softmax(char_logits[torch.arange(len(entries), device=self.device), last].float(), -1)
        fit = char_lp - self.unigram[None]
        if READING_FLOOR is not None:
            fit = fit.masked_fill(fit < READING_FLOOR, float("-inf"))
        if READING_GAP is not None:
            fit = fit.masked_fill(fit < fit.max(-1, keepdim=True).values - READING_GAP, float("-inf"))
        reading = (fit + span_lp[:, None]).masked_fill(self.not_output[None], float("-inf"))
        out = {}
        for r, (i, j, _) in enumerate(entries):
            out[(i, j)] = torch.logaddexp(out[(i, j)], reading[r]) if (i, j) in out else reading[r]
        return {ij: READING_WEIGHT * v for ij, v in out.items()}

    @torch.no_grad()
    def span_scores(self, keys, i, groups):
        """With a key model: for each group of hypotheses at key position i (sharing the keys of their
        last character, which a model with context reads) and each span keys[i:j], log P(keys[i:j] |
        character) over the vocabulary (-inf where the character was not tried), for the characters
        the language model proposes for these hypotheses plus those whose readings start with the
        span. One model call. groups: [(previous keys, language-model log-probabilities)]."""
        n = len(keys)
        chars, rows, final, before = [], [], [], []
        spans = []
        for g, (previous, lm) in enumerate(groups):
            proposed = set(lm.topk(min(CANDIDATES, lm.shape[1]), dim=1).indices.flatten().tolist()) if CANDIDATES > 0 else set()
            previous = [self.key_ids[k] for k in previous[-MAX_KEYS:]]
            previous += [0] * (MAX_KEYS - len(previous))
            for j in range(i + 1, min(n, i + MAX_KEYS) + 1):
                span = keys[i:j]
                candidates = proposed | self.prefix_index.get(span, set())
                if FUZZY_INDEX:
                    candidates = candidates | self.fuzzy_index.get(span, set())
                if FIRST_KEY and len(span) > 1:
                    candidates &= self.prefix_index.get(span[0], set())
                candidates = [c for c in candidates if not self.not_output_list[c]]
                if not candidates:
                    continue
                spans.append((g, j, len(candidates)))
                chars += candidates
                rows += [[self.key_ids[k] for k in span] + [0] * (MAX_KEYS + 1 - len(span))] * len(candidates)
                final += [j == n] * len(candidates)
                before += [previous] * len(candidates)
        if not chars:
            return {}
        self.rows_scored += len(chars)
        self.positions += 1
        logp = []
        for start in range(0, len(chars), 4096):  # in chunks, to bound the memory of one call
            part = slice(start, start + 4096)
            logp.append(self.reading.span_logprobs(torch.tensor(chars[part], device=self.device), torch.tensor(rows[part], device=self.device),
                                                   torch.tensor(final[part], device=self.device),
                                                   torch.tensor(before[part], device=self.device) if self.reading.prev else None))
        logp = torch.cat(logp)
        out, at = {}, 0
        for g, j, count in spans:
            scores = torch.full((len(self.vocab),), float("-inf"), device=self.device)
            scores[torch.tensor(chars[at : at + count], device=self.device)] = logp[at : at + count]
            out[(g, j)] = READING_WEIGHT * scores
            at += count
        return out

    def row(self, context, text):
        tokens = [1] + [self.ids.get(c, 2) for c in (context + text)[-(self.block - 1):]]
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

    def search(self, keys, context):
        """Texts for `keys`, best first, as (score, text)."""
        n = len(keys)
        self.sentences += 1
        reading = None if self.generative else self.spans(keys)
        # (text, where its last character's keys start) -> score of its best segmentation
        best = [dict() for _ in range(n + 1)]
        best[0][("", 0)] = 0.0
        for i in range(n):
            hyps = sorted(best[i].items(), key=lambda h: -h[1])[: self.beam]
            if not hyps:
                continue
            lm = []
            for start in range(0, len(hyps), 64):
                lm.append(self.logprobs([self.row(context, text) for (text, _), _ in hyps[start : start + 64]]))
            lm = torch.cat(lm)
            scores = torch.tensor([score for _, score in hyps], device=self.device)
            if self.generative:
                # Hypotheses whose last character was typed with the same keys share a group.
                members = {}
                for h, ((_, start), _) in enumerate(hyps):
                    members.setdefault(keys[start:i] if self.reading.prev else "", []).append(h)
                groups = list(members.items())
                spans = self.span_scores(keys, i, [(previous, lm[rows]) for previous, rows in groups])
            else:
                groups = [("", list(range(len(hyps))))]
                spans = {(0, j): v for (a, j), v in reading.items() if a == i}
            for g, (_, rows) in enumerate(groups):
                for j in range(i + 1, min(n, i + MAX_KEYS) + 1):
                    if (g, j) not in spans:
                        continue
                    values, indices = (scores[rows, None] + lm[rows] + spans[(g, j)][None]).topk(self.outputs_per_chunk)
                    for h, vs, ts in zip(rows, values.tolist(), indices.tolist()):
                        text = hyps[h][0][0]
                        for v, t in zip(vs, ts):
                            if v == float("-inf"):
                                break
                            grown = (text + self.vocab[t], i)
                            if v > best[j].get(grown, float("-inf")):
                                best[j][grown] = v
        finals = {}
        for (text, _), score in best[n].items():
            finals[text] = max(score, finals.get(text, float("-inf")))
        finals = sorted(finals.items(), key=lambda h: -h[1])[: self.beam]
        return [(score, text) for text, score in finals]


def main():
    global READING, READING_WEIGHT, READING_FLOOR, READING_GAP, CANDIDATES, USE_INDEX, FIRST_KEY, FUZZY_INDEX, FUZZY_MIN
    argv = sys.argv[1:]
    if "--fuzzy-index" in argv:
        FUZZY_INDEX = True
        argv.remove("--fuzzy-index")
    if "--fuzzy-min" in argv:
        at = argv.index("--fuzzy-min")
        FUZZY_MIN = int(argv[at + 1])
        del argv[at : at + 2]
    if "--first-key" in argv:
        FIRST_KEY = True
        argv.remove("--first-key")
    if "--no-index" in argv:
        USE_INDEX = False
        argv.remove("--no-index")
    if "--reading" in argv:
        at = argv.index("--reading")
        READING = argv[at + 1]
        del argv[at : at + 2]
    if "--reading-weight" in argv:
        at = argv.index("--reading-weight")
        READING_WEIGHT = float(argv[at + 1])
        del argv[at : at + 2]
    if "--reading-floor" in argv:
        at = argv.index("--reading-floor")
        READING_FLOOR = float(argv[at + 1])
        del argv[at : at + 2]
    if "--candidates" in argv:
        at = argv.index("--candidates")
        CANDIDATES = int(argv[at + 1])
        del argv[at : at + 2]
    if "--reading-gap" in argv:
        at = argv.index("--reading-gap")
        READING_GAP = float(argv[at + 1])
        del argv[at : at + 2]
    sys.argv[1:] = argv
    eval_ime.Searcher = FactoredSearcher
    eval_ime.main()
    s = getattr(FactoredSearcher, "last", None)
    if s is not None and s.positions:
        print(f"key model: {s.rows_scored / s.positions:.0f} rows per key position, {s.positions / max(1, s.sentences):.1f} positions per sentence", flush=True)


if __name__ == "__main__":
    main()
