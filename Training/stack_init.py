#!/usr/bin/env python3
"""A deeper starting point for the language model: the trained 8-block model with its blocks
duplicated (0..7 then 0..7 again) so the 16-block model starts as a working model and learns from
there instead of from noise. Embeddings, positions and the final norm are kept.

    Data/venv-train/bin/python Training/stack_init.py Data/lm9/model.pt Data/lm9x/init.pt [--repeat 2]
"""
import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    checkpoint = torch.load(args.source, map_location="cpu")
    config = dict(checkpoint["config"])
    layers = config["n_layer"]
    weights = checkpoint["model"]
    out = {}
    for name, value in weights.items():
        if name.startswith("blocks."):
            parts = name.split(".")
            index = int(parts[1])
            for r in range(args.repeat):
                parts[1] = str(index + r * layers)
                out[".".join(parts)] = value.clone()
        else:
            out[name] = value.clone()
    config["n_layer"] = layers * args.repeat
    os.makedirs(os.path.dirname(args.destination) or ".", exist_ok=True)
    torch.save({"config": config, "model": out, "step": 0, "total_steps": 0}, args.destination)
    print(f"wrote {args.destination}: {config}, {sum(v.numel() for v in out.values()) / 1e6:.1f}M parameters")


if __name__ == "__main__":
    main()
