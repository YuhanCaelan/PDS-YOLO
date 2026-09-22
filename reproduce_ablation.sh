#!/usr/bin/env bash
set -euo pipefail

# Reproduce validation metrics for the ablation checkpoints in runs/.
# Usage:
#   BENGKULU_DATA=/path/to/bengkulu/data.yaml ROBOFLOW_DATA=/path/to/roboflow/data.yaml bash reproduce_ablation.sh
#
# Optional overrides:
#   DEVICE=0 CONF=0.001 IOU=0.7 IMGSZ=640 SPLIT=val SAVE_JSON=false SAVE_TXT=false bash reproduce_ablation.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

BENGKULU_DATA="${BENGKULU_DATA:-/root/rivermind-data/chili/Chili Disease and Pest/data.yaml}"
ROBOFLOW_DATA="${ROBOFLOW_DATA:-/root/rivermind-data/chili/Chili Disease and Pest/data.yaml}"

IMGSZ="${IMGSZ:-640}"
CONF="${CONF:-0.001}"
IOU="${IOU:-0.7}"
DEVICE="${DEVICE:-}"
SPLIT="${SPLIT:-val}"
SEED="${SEED:-0}"
SAVE_JSON="${SAVE_JSON:-false}"
SAVE_TXT="${SAVE_TXT:-false}"
SAVE_CONF="${SAVE_CONF:-false}"

run_val() {
  local dataset_name="$1"
  local data_yaml="$2"
  local run_id="$3"
  local weights="runs/${dataset_name}/${run_id}/best.pt"
  local out_project="reproduce/${dataset_name}"
  local out_name="${run_id}"

  if [[ ! -f "$weights" ]]; then
    echo "[skip] Missing weights: $weights"
    return 0
  fi

  if [[ ! -f "$data_yaml" ]]; then
    echo "[error] Dataset yaml not found for ${dataset_name}: $data_yaml" >&2
    echo "        Set ${dataset_name^^}_DATA=/path/to/data.yaml and run again." >&2
    return 1
  fi

  echo "[val] ${dataset_name}/${run_id}"
  yolo detect val \
    model="$weights" \
    data="$data_yaml" \
    imgsz="$IMGSZ" \
    conf="$CONF" \
    iou="$IOU" \
    split="$SPLIT" \
    seed="$SEED" \
    deterministic=True \
    plots=True \
    save_json="$SAVE_JSON" \
    save_txt="$SAVE_TXT" \
    save_conf="$SAVE_CONF" \
    project="$out_project" \
    name="$out_name" \
    exist_ok=True \
    ${DEVICE:+device="$DEVICE"}
}

echo "Reproduction settings:"
echo "  imgsz=$IMGSZ conf=$CONF iou=$IOU split=$SPLIT seed=$SEED deterministic=True"
echo "  BENGKULU_DATA=$BENGKULU_DATA"
echo "  ROBOFLOW_DATA=$ROBOFLOW_DATA"
echo

for id in 1 2 3 4 5 6 7 8; do
  run_val "Bengkulu" "$BENGKULU_DATA" "$id"
done

for id in 1 2 3 4 5 6 7 8; do
  run_val "Roboflow" "$ROBOFLOW_DATA" "$id"
done

echo
echo "Done. Validation outputs are under reproduce/Bengkulu/ and reproduce/Roboflow/."
