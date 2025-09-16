#!/usr/bin/env bash
set -euo pipefail
nvidia-smi

# --------------- Locate the Root Directory ---------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# --------------- Environment Configuration ---------------
pip install --upgrade pip
pip install -r requirements.txt

# --------------- Parameter Configuration ---------------
: "${MODEL_NAME:=DeepSeek-R1-Distill-Qwen-1.5B}"
MODEL_DIR="../../dataset/reasoning_models_yulin"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

: "${OUTPUT_NAME:=ablation_static_steering}"
OUTPUT_DIR="../../dataset/outputs_yulin/$OUTPUT_NAME"

: "${VECTOR_NAME:=1.5b_20layer_0.67_0.94.pt}"
: "${STEER_LAYER:=20}"
STEER_VECTOR_PATH="../../algorithm/rebalance_yulin/vectors/$MODEL_NAME/$VECTOR_NAME"

: "${MAX_GENERATED_TOKENS:=16000}"

: "${DATASET_DIR:=../../dataset/reasoning_yulin}"
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5,6,7}"
: "${NUM_GPUS:=8}"

# --------------- Convert to Absolute Paths ---------------
to_abs() {
  case "$1" in
    /*) printf "%s" "$1" ;;
    *)  printf "%s/%s" "$REPO_DIR" "$1" ;;
  esac
}
MODEL_PATH="$(to_abs "$MODEL_PATH")"
OUTPUT_DIR="$(to_abs "$OUTPUT_DIR")"
STEER_VECTOR_PATH="$(to_abs "$STEER_VECTOR_PATH")"
DATASET_DIR="$(to_abs "$DATASET_DIR")"

# --------------- Experiments ---------------
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

DATASET_NAME=(Math_Math500 Math_AIME2025 Math_GSM8K Math_AMC23 Math_Olympiad)
COEFS=(0 -1 -2 -3 1)

for ds in "${DATASET_NAME[@]}"; do
  for coef in "${COEFS[@]}"; do
    OUTDIR="$OUTPUT_DIR/coef_${coef}"
    echo ">>> Running dataset=$ds, steer_coef=$coef"
    python -u "$REPO_DIR/transformer_inference_steer_dp.py" \
      --model_name_or_path "$MODEL_PATH" \
      --dataset_dir "$DATASET_DIR" \
      --output_path "$OUTDIR" \
      --dataset "$ds" \
      --max_generated_tokens "$MAX_GENERATED_TOKENS" \
      --num_gpus "$NUM_GPUS" \
      --steer_vector_path "$STEER_VECTOR_PATH" \
      --steer_layer "$STEER_LAYER" \
      --steer_coef "$coef" \
      2>&1 | tee "$OUTDIR/run_${ds}.log"
  done
done
