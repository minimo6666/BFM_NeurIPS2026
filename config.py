"""Central model contract shared by every dataset and entry point."""

from __future__ import annotations

from types import SimpleNamespace


IMAGE_SIZE = 256
LATENT_SHAPE = (1, 16, 16)
CODEBOOK_SIZE = 64


def build_model_config(args):
    """Build the shared C64 BAE/BFM configuration."""
    return SimpleNamespace(
        dataset=args.dataset,
        img_size=IMAGE_SIZE,
        latent_shape=list(LATENT_SHAPE),
        codebook_size=CODEBOOK_SIZE,
        emb_dim=256,
        block_size=256,
        batch_size=args.batch_size,
        total_steps=args.total_steps,
        sample_steps=args.sample_steps,
        focal_gamma=float(getattr(args, "focal_gamma", 0.0)),
        bert_n_emb=args.model_width,
        bert_n_head=args.model_heads,
        bert_n_layers=args.model_layers,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        drop_path=args.drop_path,
        n_channels=3,
        nf=128,
        ch_mult=[1, 1, 2, 2, 4],
        res_blocks=2,
        attn_resolutions=[16],
        quantizer="binary",
        beta=0.25,
        deterministic=False,
        use_tanh=False,
        norm_first=True,
        gen_mul=1.0,
    )
