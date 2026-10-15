"""Frozen C64 Binary-AE construction and checkpoint loading."""

from pathlib import Path

import torch

from models import BinaryAutoEncoder


def load_autoencoder(config, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Binary-AE checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError("Binary-AE checkpoint must contain a state dict")

    state = {}
    components = ("encoder.", "quantize.", "generator.")
    for key, value in payload.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("ae."):
            key = key[3:]
        if key.startswith(components):
            state[key] = value

    autoencoder = BinaryAutoEncoder(config)
    autoencoder.load_state_dict(state, strict=True)
    autoencoder.to(device).eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)
    return autoencoder
