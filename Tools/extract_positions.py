#!/usr/bin/env python3
"""Write the model's full position table as raw fp16 (Resources/positions.bin) from a weights
export (Training/export_weights.py). The input method reads every other tensor out of the Core ML
model, whose position table is cut to its 32 rows; learning trains on 64-row windows.

    python3 Tools/extract_positions.py Data/clean2/weights Resources/positions.bin
"""
import json
import sys

base, out = sys.argv[1], sys.argv[2]
header = json.load(open(base + ".json", encoding="utf-8"))
info = next(t for t in header["tensors"] if t["name"] == "positions.weight")
width = 2 if header["config"].get("dtype_bits", 16) == 16 else 4
with open(base + ".bin", "rb") as f:
    f.seek(info["offset"] * width)
    data = f.read(info["count"] * width)
assert len(data) == info["count"] * width
if width == 4:  # fp32 export: convert to fp16
    import struct
    import numpy as np
    data = np.frombuffer(data, dtype=np.float32).astype(np.float16).tobytes()
open(out, "wb").write(data)
print(f"wrote {out}: {info['shape']} fp16, {len(data)} bytes")
