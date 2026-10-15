#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "usage: $0 DATASET DATA_ROOT AE_CHECKPOINT [extra train.py arguments ...]" >&2
  exit 2
fi

DATASET="$1"
DATA_ROOT="$2"
AE_CHECKPOINT="$3"
shift 3

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/runs/$DATASET}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if [[ "$NPROC_PER_NODE" -ne 1 && "$NPROC_PER_NODE" -ne 2 ]]; then
  echo "NPROC_PER_NODE must be 1 or 2" >&2
  exit 2
fi
UPDATE_FREQ="${UPDATE_FREQ:-$((4 / NPROC_PER_NODE))}"

if [[ ! -f "$AE_CHECKPOINT" ]]; then
  echo "Binary-AE checkpoint not found: $AE_CHECKPOINT" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
exec torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" train.py \
  --dataset "$DATASET" \
  --data_root "$DATA_ROOT" \
  --ae_checkpoint "$AE_CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --update_freq "$UPDATE_FREQ" \
  "$@"
