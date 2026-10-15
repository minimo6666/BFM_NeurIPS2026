"""Minimal Transformer building blocks used by BFM."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, probability: float, training: bool):
    if probability == 0.0 or not training:
        return x
    keep_probability = 1.0 - probability
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_probability + torch.rand(
        shape, dtype=x.dtype, device=x.device
    )
    random_tensor.floor_()
    return x.div(keep_probability) * random_tensor


class DropPath(nn.Module):
    def __init__(self, probability: float):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x):
        return drop_path(x, self.probability, self.training)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.bert_n_head
        self.qkv = nn.Linear(
            config.bert_n_emb, config.bert_n_emb * 3, bias=True
        )
        self.proj = nn.Linear(config.bert_n_emb, config.bert_n_emb)
        self.proj_drop = nn.Dropout(config.resid_pdrop)
        nn.init.zeros_(self.qkv.bias)

    def forward(self, hidden):
        batch, tokens, channels = hidden.shape
        head_size = channels // self.num_heads
        qkv = self.qkv(hidden).reshape(
            batch, tokens, 3, self.num_heads, head_size
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        hidden = F.scaled_dot_product_attention(query, key, value)
        hidden = hidden.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(hidden))


class Block(nn.Module):
    def __init__(self, config, drop_path_probability=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.bert_n_emb)
        self.ln2 = nn.LayerNorm(config.bert_n_emb)
        self.attn = Attention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.bert_n_emb, 4 * config.bert_n_emb),
            nn.GELU(),
            nn.Linear(4 * config.bert_n_emb, config.bert_n_emb),
            nn.Dropout(config.resid_pdrop),
        )
        self.drop_path = (
            DropPath(drop_path_probability)
            if drop_path_probability > 0.0
            else nn.Identity()
        )
        self.gamma_1 = nn.Parameter(torch.ones(config.bert_n_emb))
        self.gamma_2 = nn.Parameter(torch.ones(config.bert_n_emb))

    def forward(self, hidden):
        hidden = hidden + self.drop_path(
            self.gamma_1 * self.attn(self.ln1(hidden))
        )
        hidden = hidden + self.drop_path(
            self.gamma_2 * self.mlp(self.ln2(hidden))
        )
        return hidden


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("2D positional embedding dimension must be even")
    embedding_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0]
    )
    embedding_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1]
    )
    return np.concatenate([embedding_h, embedding_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, position):
    if embed_dim % 2 != 0:
        raise ValueError("1D positional embedding dimension must be even")
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    position = position.reshape(-1)
    phase = np.einsum("m,d->md", position, omega)
    return np.concatenate([np.sin(phase), np.cos(phase)], axis=1)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps, dim, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, time):
        time = time / self.num_steps * self.rescale_steps
        half_dim = self.dim // 2
        frequency = math.log(10000) / (half_dim - 1)
        frequency = torch.exp(
            torch.arange(half_dim, device=time.device) * -frequency
        )
        phase = time[:, None] * frequency[None, :]
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


class AdaTkn_Time(nn.Module):
    def __init__(self, n_embd, diffusion_steps):
        super().__init__()
        self.emb = SinusoidalPosEmb(diffusion_steps, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd)
        self.l0 = nn.Linear(n_embd, n_embd)

    def forward(self, time):
        embedding = self.linear(self.silu(self.l0(self.emb(time))))
        return embedding.unsqueeze(1)
