# Dataset Preparation

## FFHQ

Download the aligned images using the
[official FFHQ repository](https://github.com/NVlabs/ffhq-dataset):

```bash
git clone https://github.com/NVlabs/ffhq-dataset.git /tmp/ffhq-dataset
mkdir -p /path/to/data/ffhq
cd /path/to/data/ffhq
python /tmp/ffhq-dataset/download_ffhq.py --images
```

Create the 256×256 training set:

```bash
cd /path/to/BFM_NeurIPS2026
python tools/prepare_ffhq256.py \
  --source /path/to/data/ffhq/images1024x1024 \
  --output /path/to/data/ffhq/ffhq256
```

Use `/path/to/data/ffhq/ffhq256` when launching FFHQ training.

## LSUN Church Outdoor and Bedroom

Download the datasets using the
[official LSUN repository](https://github.com/fyu/lsun):

```bash
git clone https://github.com/fyu/lsun.git /tmp/lsun
mkdir -p /path/to/data/lsun
cd /tmp/lsun
python3 download.py -c bedroom -o /path/to/data/lsun
python3 download.py -c church_outdoor -o /path/to/data/lsun

cd /path/to/data/lsun
unzip bedroom_train_lmdb.zip
unzip church_outdoor_train_lmdb.zip
```

The training root should contain:

```text
/path/to/data/lsun/
├── bedroom_train_lmdb/
└── church_outdoor_train_lmdb/
```

Use `/path/to/data/lsun` when launching either LSUN training script.
