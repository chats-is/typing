#!/usr/bin/env python3
"""Score a language-model checkpoint on the primary metric without exporting it: rerank the
engine's lattice readings of the word tests (enginecli --no-rerank --dump-lattice) the way the engine
does and report the first-choice rate.

    venv/bin/python Training/eval_checkpoint.py --model Data/clean3/model.pt --vocab Data/prepared-clean/vocab.json \
        Data/wordlattice/word_mixed.lattice.tsv Data/wordlattice/word_full.lattice.tsv [--rows 32]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from personalize_eval import Harness, load, read_dump  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--rows", type=int, default=32, help="row length the engine will use (context is cut to fit)")
    parser.add_argument("dumps", nargs="+")
    args = parser.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model, config = load(args.model, device)
    harness = Harness(args.vocab, config.block_size, device)
    harness.row_limit = min(args.rows, config.block_size)
    for dump in args.dumps:
        cases = read_dump(dump)
        print(f"{os.path.basename(dump):32s} exact {harness.rerank_exact(model, cases):.2f}%  (n={len(cases)})", flush=True)


if __name__ == "__main__":
    main()
