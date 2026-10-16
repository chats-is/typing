#!/usr/bin/env python3
"""Export the reading-conditioned model (reading_lm.py) to Core ML.

    Data/venv-train/bin/python Training/export_reading_lm.py Data/rlm1/model.pt Data/rlm1/ReadingModel.mlpackage

Inputs, a fixed batch of rows: ids (tokens), readings (the reading of the token after each
position, 0 = none), targets (the token after each position), last_index. Outputs per row:
  target_logp  log P(targets[t] | ids[..t], readings[t]) at every position (word scoring in the lattice)
  next_logp    log-probabilities of every token after last_index (suggestions, reading withheld or given)
RLM_BATCH sets the batch (default 64).
"""
import os
import sys
import time

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reading_lm import ReadingLM  # noqa: E402

BATCH = int(os.environ.get("RLM_BATCH", "64"))


class Exported(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("positions", torch.arange(model.block_size, dtype=torch.int32).unsqueeze(0))

    def forward(self, ids, readings, targets, last_index):
        hidden = self.model.hidden(ids, readings)
        logits = self.model.head(hidden)
        target_logit = (hidden * self.model.tokens(targets)).sum(-1)
        top = logits.max(dim=-1, keepdim=True).values
        log_sum = top.squeeze(-1) + torch.log(torch.exp(logits - top).sum(-1))
        target_logp = target_logit - log_sum
        chosen = (self.positions == last_index.reshape(BATCH, 1)).to(hidden.dtype).unsqueeze(2)
        next_logp = F.log_softmax(self.model.head((hidden * chosen).sum(1)), dim=-1)
        return target_logp, next_logp


def main():
    source, destination = sys.argv[1], sys.argv[2]
    checkpoint = torch.load(source, map_location="cpu")
    config = checkpoint["config"]
    model = ReadingLM(**config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    exported = Exported(model).eval()
    block = config["block_size"]
    ids = torch.randint(0, config["vocab_size"], (BATCH, block), dtype=torch.int32)
    readings = torch.randint(0, config["n_syllables"] + 1, (BATCH, block), dtype=torch.int32)
    last = torch.randint(0, block, (BATCH,), dtype=torch.int32)
    example = (ids, readings, ids, last)
    traced = torch.jit.trace(exported, example)
    shape = (BATCH, block)
    converted = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="ids", shape=shape, dtype=np.int32),
            ct.TensorType(name="readings", shape=shape, dtype=np.int32),
            ct.TensorType(name="targets", shape=shape, dtype=np.int32),
            ct.TensorType(name="last_index", shape=(BATCH,), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="target_logp"), ct.TensorType(name="next_logp")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    converted.user_defined_metadata["syllables"] = __import__("json").dumps(checkpoint["syllables"])
    converted.save(destination)
    with torch.no_grad():
        expected_target, expected_next = exported(*example)
    feed = {"ids": ids.numpy(), "readings": readings.numpy(), "targets": ids.numpy(), "last_index": last.numpy()}
    actual = converted.predict(feed)
    print(f"saved {destination}; max difference from PyTorch: target {np.abs(actual['target_logp'] - expected_target.numpy()).max():.4f}, next {np.abs(actual['next_logp'] - expected_next.numpy()).max():.4f}")
    for _ in range(3):
        converted.predict(feed)
    started = time.time()
    for _ in range(20):
        converted.predict(feed)
    print(f"batch {BATCH}: {1000 * (time.time() - started) / 20:.1f} ms per call (Python, includes copying)")


if __name__ == "__main__":
    main()
