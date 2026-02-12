#!/bin/bash

# ================= 1. Environment Configuration =================

# Project Root (Adjust this if running from a different directory)
PROJECT_ROOT="."

# Hardware Settings
export CUDA_VISIBLE_DEVICES=0

# ================= 2. Path Definitions =================

# Path to the Python script
# Assuming the script is located in src/analysis/SVD/ or similar
SCRIPT_PATH="${PROJECT_ROOT}/src/analysis/SVD/batch_injection_test.py"

# Model Paths
BASE_MODEL_PATH="${PROJECT_ROOT}/models/llava-1.5-7b-hf"
FT_MODEL_PATH="${PROJECT_ROOT}/checkpoints/llava_v1_5_7b/vqa_small_global_gaussian_rep/gauss-std-10_0/only_proj_v2"

# Dataset Path
TEST_FILE_PATH="${PROJECT_ROOT}/data/processed/vqa_small_global_gaussian_rep/gauss-std-10_0/test_all_poison_v2.json"

# Output Log Path
LOG_OUTPUT_PATH="${PROJECT_ROOT}/results/analysis/SVD/inject_top5.json"

# ================= 3. Experiment Parameters =================

# Injection settings
W1_SOURCE="ft"      # Source of W1 weights: 'ft' (fine-tuned) or 'base'
TOP_K=5             # Number of singular values/vectors to inject
USE_FT_BIAS="--use_ft_bias"  # Flag to enable fine-tuned bias (set to empty string "" to disable)

# ================= 4. Execution =================

echo "[INFO] Starting SVD Injection Test..."
echo "[INFO] Base Model: ${BASE_MODEL_PATH}"
echo "[INFO] FT Model:   ${FT_MODEL_PATH}"
echo "[INFO] Injecting Top-${TOP_K} components from ${W1_SOURCE}..."

python "$SCRIPT_PATH" \
  --base_model "$BASE_MODEL_PATH" \
  --ft_model "$FT_MODEL_PATH" \
  --test_file "$TEST_FILE_PATH" \
  --w1_source "$W1_SOURCE" \
  --top_k "$TOP_K" \
  $USE_FT_BIAS \
  --save_log "$LOG_OUTPUT_PATH"

echo "[SUCCESS] Test finished. Logs saved to: ${LOG_OUTPUT_PATH}"