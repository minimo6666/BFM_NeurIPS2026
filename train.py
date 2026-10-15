"""Train BFM from RGB images with on-the-fly Binary-AE encoding.

This public entry intentionally has no latent-cache mode. Every image batch is
encoded by the frozen C64 Binary Autoencoder immediately before the BFM loss.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torchvision

from autoencoder import load_autoencoder
from checkpointing import restore_training, save_training_checkpoint
from config import build_model_config
from data import SUPPORTED_DATASETS, build_loader
from models import BinaryFlowMatching, TransformerBD


def parse_args():
    parser = argparse.ArgumentParser(
        description="Online image -> C64 bits -> BFM training"
    )
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--ae_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--resume_step",
        type=int,
        default=None,
        help="Override the step inferred from a raw checkpoint filename.",
    )
    parser.add_argument(
        "--resume_optimizer",
        default=None,
        help="Optional optimizer .th paired with a raw checkpoint.",
    )
    parser.add_argument(
        "--resume_scaler",
        default=None,
        help="Optional AMP scaler .th paired with a raw checkpoint.",
    )

    parser.add_argument("--train_steps", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--update_freq", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=10_000)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260923)

    parser.add_argument("--total_steps", type=int, default=64)
    parser.add_argument("--sample_steps", type=int, default=64)
    parser.add_argument("--model_width", type=int, default=768)
    parser.add_argument("--model_layers", type=int, default=24)
    parser.add_argument("--model_heads", type=int, default=12)
    parser.add_argument("--drop_path", type=float, default=0.0)
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=0.0,
        help=(
            "Focal gamma for the flip target. The paper setting is 0, with "
            "per-sample class balancing enabled."
        ),
    )
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--checkpoint_every", type=int, default=10_000)
    parser.add_argument("--preview_every", type=int, default=5_000)
    parser.add_argument("--preview_samples", type=int, default=8)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--ema_every", type=int, default=10)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
    else:
        local_rank = 0
        rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU.")
    return distributed, rank, world_size, torch.device("cuda", local_rank)


def is_main(rank):
    return rank == 0


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_parameter, parameter in zip(
            ema_model.parameters(), model.parameters()
        ):
            ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def learning_rate(step, base_lr, warmup_steps):
    if warmup_steps <= 0 or step >= warmup_steps:
        return base_lr
    ratio = step / max(1, warmup_steps)
    return 1e-6 + ratio * (base_lr - 1e-6)


@torch.no_grad()
def save_preview(path, config, autoencoder, sampler, count, amp):
    was_training = sampler.training
    sampler.eval()
    bits = sampler.sample(
        b=count, sample_steps=config.sample_steps, temp=1.0
    )
    code = bits.permute(0, 2, 1).reshape(
        bits.shape[0], config.codebook_size, 16, 16
    )
    with torch.cuda.amp.autocast(enabled=amp):
        images, _, _ = autoencoder(None, code=code.float())
    path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(
        images.clamp(0, 1), path, nrow=max(1, int(count**0.5)), padding=0
    )
    if was_training:
        sampler.train()


def next_batch(iterator, loader, epoch):
    try:
        return next(iterator), iterator, epoch
    except StopIteration:
        epoch += 1
        if hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)
        iterator = iter(loader)
        return next(iterator), iterator, epoch


def main():
    args = parse_args()
    distributed, rank, world_size, device = setup_distributed()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if is_main(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()

    # All ranks construct identical initial weights.
    seed_everything(args.seed)
    config = build_model_config(args)
    autoencoder = load_autoencoder(config, args.ae_checkpoint, device)
    denoiser = TransformerBD(config).to(device)
    model = BinaryFlowMatching(
        config, denoiser, config.codebook_size
    ).to(device)

    ema_model = None
    if not args.no_ema:
        ema_model = copy.deepcopy(model).eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=1e-8,
    )
    amp = not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    completed_steps = 0
    if args.resume:
        completed_steps, resume_format = restore_training(
            Path(args.resume).expanduser().resolve(),
            model,
            ema_model,
            optimizer,
            scaler,
            device,
            raw_step=args.resume_step,
            raw_optimizer=args.resume_optimizer,
            raw_scaler=args.resume_scaler,
        )
        if is_main(rank):
            print(
                f"Strictly restored {resume_format} checkpoint at step "
                f"{completed_steps}: {args.resume}",
                flush=True,
            )

    if distributed:
        train_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index]
        )
    else:
        train_model = model

    # Use rank-specific stochastic augmentation/corruption after identical model
    # initialization (and after loading a shared checkpoint).
    seed_everything(args.seed + rank)
    loader = build_loader(
        args.dataset,
        args.data_root,
        config.img_size,
        args.batch_size,
        args.workers,
        distributed,
        args.seed,
    )
    epoch = 0
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)
    iterator = iter(loader)

    if is_main(rank):
        run_config = vars(args).copy()
        run_config.update(
            world_size=world_size,
            effective_batch_size=(
                args.batch_size * world_size * args.update_freq
            ),
            data_flow="RGB -> frozen C64 Binary-AE -> sampled bits -> BFM",
            prediction_target="flip",
            reverse_sampler="soft_x0_plugin",
        )
        (output_dir / "config.json").write_text(
            json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(run_config, indent=2), flush=True)

    log_path = output_dir / "train.jsonl"
    optimizer.zero_grad(set_to_none=True)
    while completed_steps < args.train_steps:
        start_time = time.time()
        lr = learning_rate(completed_steps, args.lr, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        micro_stats = []
        for micro_step in range(args.update_freq):
            batch, iterator, epoch = next_batch(iterator, loader, epoch)
            images = batch[0].to(device, non_blocking=True)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
                bits = autoencoder(images, code_only=True).detach()
                batch_size, channels, height, width = bits.shape
                if (channels, height, width) != (64, 16, 16):
                    raise RuntimeError(
                        "Expected C64 BAE bits with shape [B,64,16,16], got "
                        f"{tuple(bits.shape)}"
                    )
                x0 = bits.view(batch_size, channels, -1).permute(0, 2, 1)

            final_micro_step = micro_step == args.update_freq - 1
            sync_context = (
                nullcontext()
                if final_micro_step or not distributed
                else train_model.no_sync()
            )
            with sync_context:
                with torch.cuda.amp.autocast(enabled=amp):
                    stats = train_model(x0.contiguous())
                    loss = stats["loss"] / args.update_freq
                scaler.scale(loss).backward()
            micro_stats.append(
                {
                    key: float(value.detach().mean().item())
                    for key, value in stats.items()
                    if torch.is_tensor(value)
                }
            )

        scaler.unscale_(optimizer)
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        completed_steps += 1

        if ema_model is not None and completed_steps % args.ema_every == 0:
            update_ema(ema_model, model, args.ema_decay)

        if completed_steps % args.log_every == 0 and is_main(rank):
            averaged = {
                key: float(np.mean([entry[key] for entry in micro_stats]))
                for key in micro_stats[0]
            }
            record = {
                "step": completed_steps,
                **averaged,
                "lr": lr,
                "grad_norm": float(grad_norm),
                "seconds": time.time() - start_time,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

        preview_due = (
            args.preview_every > 0
            and completed_steps % args.preview_every == 0
        )
        if preview_due and is_main(rank):
            preview_model = ema_model if ema_model is not None else model
            save_preview(
                output_dir / "samples" / f"step_{completed_steps:07d}.png",
                config,
                autoencoder,
                preview_model,
                args.preview_samples,
                amp,
            )
        if preview_due and distributed:
            dist.barrier()

        checkpoint_due = completed_steps % args.checkpoint_every == 0
        if checkpoint_due and is_main(rank):
            save_training_checkpoint(
                output_dir / "checkpoints" / f"step_{completed_steps:07d}.pt",
                completed_steps,
                model,
                ema_model,
                optimizer,
                scaler,
                args,
            )
        if checkpoint_due and distributed:
            dist.barrier()

    if is_main(rank) and completed_steps % args.checkpoint_every != 0:
        save_training_checkpoint(
            output_dir / "checkpoints" / f"step_{completed_steps:07d}.pt",
            completed_steps,
            model,
            ema_model,
            optimizer,
            scaler,
            args,
        )
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
