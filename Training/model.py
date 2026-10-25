"""The character-level GPT used for candidate reranking and next-word prediction."""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512

    def to_dict(self):
        return asdict(self)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Shapes are taken from the config, not from the input, so the exported graph is fully static.
        self.split = (config.block_size, config.n_head, config.d_model // config.n_head)
        self.d_model = config.d_model
        self.ln1 = nn.LayerNorm(config.d_model)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.fc = nn.Linear(config.d_model, 4 * config.d_model)
        self.out = nn.Linear(4 * config.d_model, config.d_model)

    def forward(self, x, mask):
        q, k, v = self.qkv(self.ln1(x)).split(self.d_model, dim=2)
        q = q.reshape(-1, *self.split).transpose(1, 2)
        k = k.reshape(-1, *self.split).transpose(1, 2)
        v = v.reshape(-1, *self.split).transpose(1, 2)
        if mask is None:
            attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Spelled out with an additive mask for Core ML export.
            weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.split[2]) + mask
            attended = F.softmax(weights, dim=-1) @ v
        x = x + self.proj(attended.transpose(1, 2).reshape(-1, self.split[0], self.d_model))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.positions = nn.Embedding(config.block_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.tokens.weight
        causal = torch.full((config.block_size, config.block_size), -1e4).triu(1)
        self.register_buffer("causal", causal, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def hidden(self, ids, explicit_mask=False):
        """Final hidden state at each position. `ids` is (batch, block_size)."""
        x = self.tokens(ids) + self.positions.weight
        mask = self.causal if explicit_mask else None
        for block in self.blocks:
            x = block(x, mask)
        return self.ln(x)

    def forward(self, ids, explicit_mask=False):
        """Logits for the token following each position."""
        return self.head(self.hidden(ids, explicit_mask))
