# Bernoulli Flow Models

**Accepted at NeurIPS 2026 as a poster.**

This is the official PyTorch implementation of
[Bernoulli Flow Models: Self-Consistent Generative Modeling for Binary Data](https://arxiv.org/abs/2610.11362).

This repository provides training and sampling examples for FFHQ, LSUN Church
Outdoor, and LSUN Bedroom at 256×256 resolution.

## Environment

Download [BFM.zip](https://pan.baidu.com/s/1KFbHiWeIxegDoc3ibmaRFw)
(extraction code: `ga78`) and unzip it into your Conda environments directory.

The archived environment uses `/home/minimo/.conda/envs/BFM` as its prefix.
Replace it with the absolute path where you extracted the environment:

```bash
find /path/to/anaconda3/envs/BFM/bin -type f \
  -exec sed -i \
  's|/home/minimo/.conda/envs/BFM/bin|/path/to/anaconda3/envs/BFM/bin|g' {} +

conda activate /path/to/anaconda3/envs/BFM
```

## Binary Autoencoder

Open the official BinaryLatentDiffusion
[binary-ae page](https://github.com/ZeWang95/BinaryLatentDiffusion/tree/main/binary-ae),
and download the **C64** checkpoint. Pass its path to the training and sampling
commands below.

## Datasets

See [Dataset preparation](docs/data/README.md) for FFHQ, LSUN Church Outdoor,
and LSUN Bedroom.

## Training and Sampling

### Training

Run commands from `BFM_NeurIPS2026`. One GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_ffhq.sh \
  /path/to/data/ffhq/ffhq256 \
  /path/to/binaryae_ema_8100000.th

CUDA_VISIBLE_DEVICES=0 bash scripts/train_churches.sh \
  /path/to/data/lsun \
  /path/to/binaryae_ema_8100000.th

CUDA_VISIBLE_DEVICES=0 bash scripts/train_bedrooms.sh \
  /path/to/data/lsun \
  /path/to/binaryae_ema_8100000.th
```

Two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 \
  bash scripts/train_churches.sh \
  /path/to/data/lsun \
  /path/to/binaryae_ema_8100000.th
```

By default, training runs for 50,000 steps, saves a checkpoint every 10,000
steps, and generates samples every 5,000 steps.

### Sampling

```bash
CUDA_VISIBLE_DEVICES=0 python sample.py \
  --dataset ffhq \
  --checkpoint runs/ffhq/checkpoints/bfm_ema_50000.th \
  --ae_checkpoint /path/to/binaryae_ema_8100000.th \
  --output_dir samples/ffhq \
  --num_samples 64
```

## Citation

```bibtex
@article{mo2026bernoulli,
  title={Bernoulli Flow Models: Self-Consistent Generative Modeling for Binary Data},
  author={Mo, Hao and Yang, Liying and Yao, Shumin and Yu, Xinxing and Liu, Ajian and Mao, Xudong and Liang, Yanyan},
  journal={arXiv preprint arXiv:2610.11362},
  year={2026}
}
```
