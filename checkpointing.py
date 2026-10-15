"""Checkpoint input and output."""

from __future__ import annotations

import os
import re
from pathlib import Path

import torch


def _torch_load(path, map_location):
    return torch.load(Path(path).expanduser().resolve(), map_location=map_location)


def _strip_ddp_prefix(state_dict):
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module.") :]: value for key, value in state_dict.items()}
    return state_dict


def extract_model_state(payload):
    """Extract a model state dict from raw or wrapped checkpoint formats."""
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload must be a dict, got {type(payload)}")
    for key in ("model", "state_dict", "ema"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return _strip_ddp_prefix(candidate)
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return _strip_ddp_prefix(payload)
    raise ValueError("Could not find a model state dict in checkpoint")


def load_weights(model, path, device="cpu"):
    """Load BFM model weights with strict key matching."""
    payload = _torch_load(path, device)
    state_dict = extract_model_state(payload)
    model.load_state_dict(state_dict, strict=True)
    return payload


def _infer_step(path):
    match = re.search(r"_(\d+)\.(?:pt|pth|th)$", Path(path).name)
    return int(match.group(1)) if match else 0


def restore_training(
    path,
    model,
    ema_model,
    optimizer,
    scaler,
    device,
    *,
    raw_step=None,
    raw_optimizer=None,
    raw_scaler=None,
):
    """Restore a full training checkpoint or a raw model state dict."""
    payload = _torch_load(path, device)
    is_native = (
        isinstance(payload, dict)
        and isinstance(payload.get("model"), dict)
        and "step" in payload
    )
    if is_native:
        model.load_state_dict(_strip_ddp_prefix(payload["model"]), strict=True)
        if ema_model is not None:
            ema_state = payload.get("ema")
            if ema_state is None and payload.get("ema_file"):
                ema_payload = _torch_load(
                    Path(path).expanduser().resolve().parent / payload["ema_file"],
                    device,
                )
                ema_state = extract_model_state(ema_payload)
            if ema_state is None:
                ema_state = payload["model"]
            ema_model.load_state_dict(_strip_ddp_prefix(ema_state), strict=True)
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if "scaler" in payload:
            scaler.load_state_dict(payload["scaler"])
        return int(payload["step"]), "native"

    state_dict = extract_model_state(payload)
    model.load_state_dict(state_dict, strict=True)
    if ema_model is not None:
        ema_model.load_state_dict(state_dict, strict=True)
    if raw_optimizer:
        optimizer.load_state_dict(_torch_load(raw_optimizer, device))
    if raw_scaler:
        scaler.load_state_dict(_torch_load(raw_scaler, device))
    step = _infer_step(path) if raw_step is None else int(raw_step)
    return step, "raw"


def save_training_checkpoint(path, step, model, ema_model, optimizer, scaler, args):
    """Save full training state and raw EMA weights."""
    path = Path(path)
    release_model = ema_model if ema_model is not None else model
    release_path = path.parent / f"bfm_ema_{step}.th"
    path.parent.mkdir(parents=True, exist_ok=True)
    release_temporary = release_path.with_suffix(".th.tmp")
    torch.save(release_model.state_dict(), release_temporary)
    os.replace(release_temporary, release_path)

    payload = {
        "format_version": 1,
        "step": int(step),
        "model": model.state_dict(),
        "ema_file": release_path.name,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
