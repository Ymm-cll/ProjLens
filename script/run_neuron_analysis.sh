#!/bin/bash

# ================= 1. Environment Configuration =================

# Project Root (Adjust this relative to where you run the script)
PROJECT_ROOT="."

# Hardware Settings
export CUDA_VISIBLE_DEVICES=1

# ================= 2. Path Definitions =================

# Path to the Python analysis script
# Assuming the script is located in src/analysis/
SCRIPT_PATH="${PROJECT_ROOT}/src/analysis/analyze_neurons.py"

# Model Paths
BASE_MODEL_PATH="${PROJECT_ROOT}/models/llava-1.5-7b-hf"
FT_MODEL_PATH="${PROJECT_ROOT}/checkpoints/llava_v1_5_7b/vqa_small_global_gaussian_rep/gauss-std-10_0/only_proj_v2"

# Dataset Paths
# Note: Two JSONs are required: one for the Poisoned (Triggered) data and one for the Clean data
DATA_DIR="${PROJECT_ROOT}/data/processed/vqa_small_global_gaussian_rep/gauss-std-10_0"
POISON_JSON="${DATA_DIR}/test_all_poison_v2.json"
CLEAN_JSON="${DATA_DIR}/test_clean.json"

# Output Directory
OUTPUT_DIR="${PROJECT_ROOT}/results/neuron_analysis"

# ================= 3. Analysis Parameters =================

MAX_SAMPLES=500        # Number of samples to analyze
TOP_K_NEURONS=20       # Number of top neurons to attribute

# ================= 4. Execution =================

echo "[INFO] Starting Neuron Attribution Analysis..."
echo "[INFO] Base Model:  ${BASE_MODEL_PATH}"
echo "[INFO] FT Model:    ${FT_MODEL_PATH}"
echo "[INFO] Poison Data: ${POISON_JSON}"
echo "[INFO] Clean Data:  ${CLEAN_JSON}"

# Ensure output directory exists (optional, usually Python handles it, but good practice)
mkdir -p "$OUTPUT_DIR"

python "$SCRIPT_PATH" \
  --base_model "$BASE_MODEL_PATH" \
  --ft_model "$FT_MODEL_PATH" \
  --clean_json "$CLEAN_JSON" \
  --poison_json "$POISON_JSON" \
  --output_dir "$OUTPUT_DIR" \
  --max_samples "$MAX_SAMPLES" \
  --top_k_neurons "$TOP_K_NEURONS"

echo "[SUCCESS] Analysis Complete! Check results in: $OUTPUT_DIR"