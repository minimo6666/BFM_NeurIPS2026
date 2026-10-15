"""Generate PNG images from a BFM checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision

from autoencoder import load_autoencoder
from checkpointing import load_weights
from config import build_model_config
from data import SUPPORTED_DATASETS
from models import BinaryFlowMatching, TransformerBD


def parse_args():
    parser = argparse.ArgumentParser(description="Sample a BFM checkpoint")
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ae_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--total_steps", type=int, default=64)
    parser.add_argument("--sample_steps", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_width", type=int, default=768)
    parser.add_argument("--model_layers", type=int, default=24)
    parser.add_argument("--model_heads", type=int, default=12)
    parser.add_argument("--drop_path", type=float, default=0.0)
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Sampling requires a CUDA GPU.")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    config = build_model_config(args)
    autoencoder = load_autoencoder(config, args.ae_checkpoint, device)
    model = BinaryFlowMatching(
        config, TransformerBD(config), config.codebook_size
    ).to(device)
    load_weights(model, args.checkpoint, device)
    model.eval()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    amp = not args.no_amp
    while written < args.num_samples:
        current_batch = min(args.batch_size, args.num_samples - written)
        bits = model.sample(
            b=current_batch,
            sample_steps=args.sample_steps,
            temp=args.temperature,
        )
        code = bits.permute(0, 2, 1).reshape(
            current_batch, config.codebook_size, 16, 16
        )
        with torch.cuda.amp.autocast(enabled=amp):
            images, _, _ = autoencoder(None, code=code.float())
        for index, image in enumerate(images):
            torchvision.utils.save_image(
                image.clamp(0, 1), output_dir / f"{written + index:06d}.png"
            )
        written += current_batch
        print(f"generated {written}/{args.num_samples}", flush=True)


if __name__ == "__main__":
    main()
