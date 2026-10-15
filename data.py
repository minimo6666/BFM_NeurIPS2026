"""Datasets used by the public FFHQ/LSUN training entry."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, Resize, ToTensor


SUPPORTED_DATASETS = ("ffhq", "churches", "bedrooms")


def _transform(image_size: int):
    return Compose(
        [
            Resize(int(image_size * 1.05)),
            RandomCrop(image_size),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
        ]
    )


def build_dataset(name: str, data_root: str, image_size: int = 256):
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    transform = _transform(image_size)
    if name == "ffhq":
        dataset = torchvision.datasets.ImageFolder(str(root), transform=transform)
        if len(dataset) == 0:
            raise RuntimeError(f"No images found under {root}")
        first_image = Path(dataset.samples[0][0])
        with Image.open(first_image) as image:
            if image.size != (256, 256):
                raise ValueError(
                    "FFHQ training expects pre-resized 256x256 PNGs, but "
                    f"{first_image} is {image.size}. Run "
                    "tools/prepare_ffhq256.py on the official "
                    "images1024x1024 directory first."
                )
    elif name == "churches":
        _require_lsun_class(root, "church_outdoor_train_lmdb")
        dataset = _build_lsun_dataset(root, "church_outdoor_train", transform)
    elif name == "bedrooms":
        _require_lsun_class(root, "bedroom_train_lmdb")
        dataset = _build_lsun_dataset(root, "bedroom_train", transform)
    else:
        raise ValueError(f"Unknown dataset {name!r}; choose from {SUPPORTED_DATASETS}")

    if len(dataset) == 0:
        raise RuntimeError(f"No images found under {root}")
    return dataset


def _require_lsun_class(root: Path, directory_name: str) -> None:
    expected = root / directory_name
    if not expected.is_dir():
        raise FileNotFoundError(
            f"Missing {expected}. Unzip the corresponding LSUN *_lmdb.zip "
            "archive directly below --data_root."
        )


def _build_lsun_dataset(root: Path, class_name: str, transform):
    import fcntl

    cache_root = Path(
        os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    ).expanduser()
    cache_dir = cache_root / "bernoulli_flow_models" / "lsun"
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / f"{class_name}.lock"

    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        working_directory = Path.cwd()
        try:
            os.chdir(cache_dir)
            dataset = torchvision.datasets.LSUN(
                str(root), classes=[class_name], transform=transform
            )
        finally:
            os.chdir(working_directory)
            fcntl.flock(lock, fcntl.LOCK_UN)
    return dataset


def _seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def build_loader(
    name: str,
    data_root: str,
    image_size: int,
    batch_size: int,
    workers: int,
    distributed: bool,
    seed: int,
):
    dataset = build_dataset(name, data_root, image_size)
    if distributed:
        sampler = torch.utils.data.DistributedSampler(
            dataset, shuffle=True, seed=seed, drop_last=True
        )
    else:
        generator = torch.Generator().manual_seed(seed)
        sampler = torch.utils.data.RandomSampler(dataset, generator=generator)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
        worker_init_fn=_seed_worker,
    )
    return loader
