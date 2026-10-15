"""Unconditional Transformer denoiser for 16x16x64 binary latents."""

import torch
import torch.nn as nn

from .transformer_utils import AdaTkn_Time, Block, get_2d_sincos_pos_embed


class TransformerBD(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.codebook_size
        self.n_embd = config.bert_n_emb
        self.block_size = config.block_size
        self.n_layers = config.bert_n_layers
        self.codebook_size = config.codebook_size

        self.tok_emb = nn.Embedding(self.vocab_size, self.n_embd)
        self.tok_emb.weight.data.normal_(0.0, 0.02)
        self.drop = nn.Dropout(config.embd_pdrop)

        drop_path_rates = [
            value.item()
            for value in torch.linspace(0, config.drop_path, self.n_layers)
        ]
        self.blocks = nn.Sequential(
            *[Block(config, drop_path_rates[index]) for index in range(self.n_layers)]
        )

        self.ln_f = nn.LayerNorm(self.n_embd)
        self.head = nn.Linear(self.n_embd, self.codebook_size, bias=True)
        self.sample_steps = config.sample_steps
        self.time_step_embedding = AdaTkn_Time(
            self.n_embd, self.sample_steps
        )
        position = get_2d_sincos_pos_embed(
            self.n_embd, config.latent_shape[-1]
        )
        self.pos_emb = nn.Parameter(torch.Tensor(position).unsqueeze(0))

    def get_block_size(self):
        return self.block_size

    def forward_features(self, bits, time_steps):
        token_embeddings = ((bits.float() - 0.5) * 2.0) @ self.tok_emb.weight
        token_count = token_embeddings.shape[1]
        hidden = token_embeddings + self.pos_emb[:, :token_count, :]

        time_embedding = self.time_step_embedding(time_steps)
        hidden = torch.cat([hidden, time_embedding], dim=1)
        hidden = self.drop(hidden)
        for block in self.blocks:
            hidden = block(hidden)

        hidden = hidden[:, : self.block_size, :]
        return self.ln_f(hidden)

    def forward(self, bits, time_steps=None):
        return self.head(self.forward_features(bits, time_steps))
