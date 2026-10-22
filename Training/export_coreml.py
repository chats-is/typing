#!/usr/bin/env python3
"""Export a trained checkpoint to Core ML for the input method.

    Data/venv-train/bin/python Training/export_coreml.py Data/clean2/model.pt Data/clean2/LanguageModel32.mlpackage

The exported model takes a fixed batch of token rows and returns, per row:
  target_logp  log-probability of targets[t] following ids[..t]   (candidate reranking)
  next_logp    log-probabilities of every token after position last_index   (prediction)
"""
import os
import sys

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, Config

BATCH = int(__import__("os").environ.get("LM_BATCH", "16"))


class Exported(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("positions", torch.arange(model.config.block_size, dtype=torch.int32).unsqueeze(0))

    def forward(self, ids, targets, last_index):
        hidden = self.model.hidden(ids, explicit_mask=True)
        logits = self.model.head(hidden)
        # The target's log-probability is its logit (hidden state times its tied embedding) minus
        # the log-sum-exp of all logits. Both gathering from the full log-softmax and the built-in
        # logsumexp gave wrong numbers on the Neural Engine in half precision, so the log-sum-exp is
        # written out around the row maximum, which cannot overflow.
        target_logit = (hidden * self.model.tokens(targets)).sum(-1)
        top = logits.max(dim=-1, keepdim=True).values
        log_sum = top.squeeze(-1) + torch.log(torch.exp(logits - top).sum(-1))
        target_logp = target_logit - log_sum
        # Pick each row's hidden state at last_index with a one-hot sum, then score the vocabulary once.
        chosen = (self.positions == last_index.reshape(BATCH, 1)).to(hidden.dtype).unsqueeze(2)
        next_logp = F.log_softmax(self.model.head((hidden * chosen).sum(1)), dim=-1)
        return target_logp, next_logp


def main():
    source, destination = sys.argv[1], sys.argv[2]
    checkpoint = torch.load(source, map_location="cpu")
    config = Config(**checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model"])
    # LM_ROWS exports the model for shorter rows (less context, proportionally less work per call):
    # the same weights with the position table cut to the first LM_ROWS entries.
    rows = int(os.environ.get("LM_ROWS", config.block_size))
    if rows != config.block_size:
        short = Config(**{**checkpoint["config"], "block_size": rows})
        state = {k: v for k, v in model.state_dict().items()}
        state["positions.weight"] = state["positions.weight"][:rows].clone()
        state = {k: v for k, v in state.items() if k != "causal"}
        model = GPT(short)
        model.load_state_dict(state, strict=False)
        config = short
    exported = Exported(model).eval()

    ids = torch.randint(0, config.vocab_size, (BATCH, config.block_size), dtype=torch.int32)
    last = torch.randint(0, config.block_size, (BATCH,), dtype=torch.int32)
    example = (ids, ids, last)
    traced = torch.jit.trace(exported, example)

    shape = (BATCH, config.block_size)
    converted = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="ids", shape=shape, dtype=np.int32),
            ct.TensorType(name="targets", shape=shape, dtype=np.int32),
            ct.TensorType(name="last_index", shape=(BATCH,), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="target_logp"), ct.TensorType(name="next_logp")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32 if os.environ.get("LM_FP32") else ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        # Checked below on the same hardware path the input method uses.
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    converted.save(destination)

    # The converted model must agree with PyTorch.
    with torch.no_grad():
        expected_target, expected_next = exported(*example)
    actual = converted.predict({"ids": ids.numpy(), "targets": ids.numpy(), "last_index": last.numpy()})
    target_error = np.abs(actual["target_logp"] - expected_target.numpy()).max()
    next_error = np.abs(actual["next_logp"] - expected_next.numpy()).max()
    print(f"saved {destination}; max difference from PyTorch: target {target_error:.4f}, next {next_error:.4f}")


if __name__ == "__main__":
    main()
