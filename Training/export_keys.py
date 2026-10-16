#!/usr/bin/env python3
"""Export a trained key model (key_model.py) to Core ML for the input method.

    Data/venv-train/bin/python Training/export_keys.py Data/keys4/model.pt Data/keys4/KeyModel.mlpackage

The exported model scores a fixed batch of rows (character, keys typed for it, keys typed for the
character before it) and returns per row
  end_logp  log P(keys, then the syllable ends | character, previous keys)
  cut_logp  the same, but the syllable may also be unfinished (for the span at the end of the buffer)
The model's own readings index (the likeliest ways to type each character, found after training) and
the key token list travel in the package's metadata, so the input method needs this one file.
"""
import json
import math
import os
import sys

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F

from key_model import CUT, END, KeyModel
from make_ime_data import NEIGHBOURS
from reading_model import MAX_KEYS

BATCH = int(__import__("os").environ.get("KEY_BATCH", "512"))


class Exported(torch.nn.Module):
    """The key model written out layer by layer (nn.TransformerEncoderLayer does not convert)."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        total = model.prev + MAX_KEYS + 2
        self.total = total
        self.d_model = model.characters.embedding_dim
        self.n_head = model.encoder.layers[0].self_attn.num_heads
        self.register_buffer("positions", torch.arange(MAX_KEYS + 2, dtype=torch.float32)[None])
        self.register_buffer("key_range", torch.arange(model.next_key.out_features, dtype=torch.float32)[None, None])
        self.register_buffer("mask", torch.full((total, total), -1e4).triu(1))

    def layer(self, x, layer):
        # Shapes come from the configuration, not the input, so the traced graph is static.
        batch, n, d = BATCH, self.total, self.d_model
        heads, dh = self.n_head, d // self.n_head
        h = layer.norm1(x)
        qkv = h @ layer.self_attn.in_proj_weight.t() + layer.self_attn.in_proj_bias
        q, k, v = qkv.split(d, dim=2)
        q = q.reshape(batch, n, heads, dh).transpose(1, 2)
        k = k.reshape(batch, n, heads, dh).transpose(1, 2)
        v = v.reshape(batch, n, heads, dh).transpose(1, 2)
        weights = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(dh) + self.mask[:n, :n], dim=-1)
        attended = (weights @ v).transpose(1, 2).reshape(batch, n, d)
        x = x + layer.self_attn.out_proj(attended)
        h = layer.norm2(x)
        return x + layer.linear2(torch.relu(layer.linear1(h)))

    def forward(self, characters, keys, prev):
        m = self.model
        parts = [m.characters(characters)[:, None], m.keys(keys)]
        if m.prev:
            parts.insert(0, m.keys(prev))
        x = torch.cat(parts, 1)
        x = x + m.positions.weight[: self.total]
        for layer in m.encoder.layers:
            x = self.layer(x, layer)
        logits = m.next_key(m.ln(x))[:, m.prev :]
        # log-softmax written out around the row maximum (see export_coreml.py).
        top = logits.max(dim=-1, keepdim=True).values
        logp = logits - top - torch.log(torch.exp(logits - top).sum(-1, keepdim=True))
        length = (keys > 0).to(torch.float32).sum(1, keepdim=True)  # (B, 1)
        targets = torch.cat([keys, torch.zeros((BATCH, 1), dtype=keys.dtype)], 1).to(torch.float32)
        stepwise = (logp * (targets[..., None] == self.key_range).to(logp.dtype)).sum(-1)  # (B, 9)
        typed = (self.positions < length).to(logp.dtype)
        keys_lp = (stepwise * typed).sum(1)
        at_end = (logp * (self.positions == length).to(logp.dtype)[..., None]).sum(1)  # (B, n_keys)
        end = at_end[:, END]
        cut = at_end[:, CUT]
        larger = torch.maximum(end, cut)
        both = larger + torch.log(torch.exp(end - larger) + torch.exp(cut - larger))
        return keys_lp + end, keys_lp + both


def main():
    source, destination = sys.argv[1], sys.argv[2]
    checkpoint = torch.load(source, map_location="cpu")
    model = KeyModel(**checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    torch.backends.mha.set_fastpath_enabled(False)  # the fused encoder layer cannot be converted
    exported = Exported(model).eval()
    n_keys = checkpoint["config"]["n_keys"]
    vocab_size = checkpoint["config"]["vocab_size"]

    characters = torch.randint(0, vocab_size, (BATCH,), dtype=torch.int32)
    keys = torch.randint(4, n_keys, (BATCH, MAX_KEYS + 1), dtype=torch.int32)
    lengths = torch.randint(1, MAX_KEYS + 1, (BATCH, 1))
    keys = torch.where(torch.arange(MAX_KEYS + 1)[None] < lengths, keys, torch.zeros_like(keys))
    prev = torch.randint(0, n_keys, (BATCH, MAX_KEYS), dtype=torch.int32)
    prev = torch.where(torch.arange(MAX_KEYS)[None] < torch.randint(0, MAX_KEYS + 1, (BATCH, 1)), prev, torch.zeros_like(prev))
    prev = torch.where(prev < 4, torch.zeros_like(prev), prev)
    example = (characters, keys, prev)
    with torch.no_grad():
        traced = torch.jit.trace(exported, example)

    converted = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="characters", shape=(BATCH,), dtype=np.int32),
            ct.TensorType(name="keys", shape=(BATCH, MAX_KEYS + 1), dtype=np.int32),
            ct.TensorType(name="prev", shape=(BATCH, MAX_KEYS), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="end_logp"), ct.TensorType(name="cut_logp")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32 if os.environ.get("KEY_FP32") else ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    index = {reading: [c for c, _ in entries] for reading, entries in (checkpoint["index"] or {}).items()}
    converted.user_defined_metadata["key_tokens"] = json.dumps(checkpoint["key_tokens"], ensure_ascii=False)
    converted.user_defined_metadata["index"] = json.dumps(index, ensure_ascii=False)
    converted.user_defined_metadata["prev"] = str(checkpoint["config"].get("prev", 0))
    converted.user_defined_metadata["max_keys"] = str(MAX_KEYS)
    # The keys next to each key, as the slips in the training pairs were made: the decoder looks
    # for characters whose readings are one such slip away from the typed keys.
    converted.user_defined_metadata["neighbours"] = json.dumps(NEIGHBOURS)
    converted.save(destination)

    with torch.no_grad():
        expected_end, expected_cut = exported(*example)
    actual = converted.predict({"characters": characters.numpy(), "keys": keys.numpy(), "prev": prev.numpy()})
    end_error = np.abs(actual["end_logp"] - expected_end.numpy()).max()
    cut_error = np.abs(actual["cut_logp"] - expected_cut.numpy()).max()
    print(f"saved {destination}; max difference from PyTorch: end {end_error:.4f}, cut {cut_error:.4f}; {len(index)} readings in the index")


if __name__ == "__main__":
    main()
