#!/usr/bin/env python3
"""Dump a GPT checkpoint's weights as one little-endian fp16 blob with a JSON header, the form the
on-device runtime (MPSGraph) loads for inference and for learning the top blocks.

    Data/venv-train/bin/python Training/export_weights.py Data/clean3/model.pt Data/clean3/weights

Writes <out>.json ({"config": {...}, "tensors": [{"name", "shape", "offset", "count"}]}) and
<out>.bin (fp16 values, tensor after tensor in header order; offsets in elements).
"""
import json
import sys

import numpy as np
import torch


def main():
    source, out = sys.argv[1], sys.argv[2]
    checkpoint = torch.load(source, map_location="cpu")
    tensors = []
    offset = 0
    with open(out + ".bin", "wb") as f:
        for name, value in checkpoint["model"].items():
            if name == "head.weight":  # tied to tokens.weight
                continue
            array = value.detach().to(torch.float16).numpy().astype("<f2")
            tensors.append({"name": name, "shape": list(value.shape), "offset": offset, "count": int(array.size)})
            f.write(array.tobytes())
            offset += int(array.size)
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump({"config": checkpoint["config"], "tensors": tensors}, f, indent=1)
    print(f"wrote {out}.bin ({offset * 2 / 1e6:.1f} MB) and {out}.json with {len(tensors)} tensors")


if __name__ == "__main__":
    main()
