#!/bin/bash

# ================= 1. Environment Configuration =================

# Project Root (Assume running from project root or adjust relative path)
PROJECT_ROOT="." 

# Hardware & Model Settings
export CUDA_VISIBLE_DEVICES=0  # Specify GPU device
BASE_MODEL="${PROJECT_ROOT}/models/llava-1.5-7b-hf" # Path to the original base model
OUTPUT_DIR="${PROJECT_ROOT}/results/ablation_results_multi_task"
MAX_SAMPLES=200                # Number of samples to test per experiment

# Define base paths for checkpoints and datasets to make JSON cleaner below
CKPT_BASE="${PROJECT_ROOT}/checkpoints/llava_v1_5_7b"
DATA_BASE="${PROJECT_ROOT}/results/backdoor/dataset"

# ================= 2. Define Experiment Configuration (JSON) =================
# Add all backdoor models and corresponding test sets here.
# Note: Ensure ft_model_path and test_file point to actual existing files.

CONFIG_JSON="ablation_config_tmp.json"

# We use cat <<EOF to generate the JSON file dynamically.
# Bash variables (like ${CKPT_BASE}) will be expanded automatically inside.
cat <<EOF > "$CONFIG_JSON"
{
  "experiments": [
    {
      "name": "Flickr30k_Local_Patch",
      "ft_model_path": "${CKPT_BASE}/flickr30k_local_color_add-suffix/pos-center_px-14/only_proj",
      "test_file": "${DATA_BASE}/flickr30k_local_color_add-suffix/pos-center_px-14/test_all_poison.json"
    },
    {
      "name": "MSCOCO_Invisible_Noise",
      "ft_model_path": "${CKPT_BASE}/mscoco_local_image_rep/pos-random_px-14/only_proj",
      "test_file": "${DATA_BASE}/mscoco_local_image_rep/pos-random_px-14/test_all_poison.json"
    },
    {
      "name": "Oil_Painting_Global",
      "ft_model_path": "${CKPT_BASE}/VLBreakBench_global_style/oil/only_proj",
      "test_file": "${DATA_BASE}/VLBreakBench_global_style/oil/test_oil.json"
    }
  ],

  "tasks": [
    {
      "mode": "remove_specific",
      "target_rank": 1,
      "desc": "Remove Rank-1 (Stealthy)"
    },
    {
      "mode": "top_k",
      "top_k": 5,
      "desc": "Remove Top-5 (Aggressive)"
    },
    {
      "mode": "rank2_to_k",
      "top_k": 10,
      "desc": "Remove Rank 2-10 (Preserve R1)"
    }
  ]
}
EOF

# ================= 3. Execute Python Script =================

echo "[INFO] Generating configuration file: $CONFIG_JSON"
echo "[INFO] Starting Batch Ablation..."

# Ensure the python path points to your actual script location (e.g., src/batch_ablation_svd.py)
SCRIPT_PATH="${PROJECT_ROOT}/src/batch_ablation_svd.py"

python "$SCRIPT_PATH" \
  --base_model "$BASE_MODEL" \
  --config_path "$CONFIG_JSON" \
  --output_dir "$OUTPUT_DIR" \
  --max_samples "$MAX_SAMPLES"

# ================= 4. Cleanup =================

if [ -f "$CONFIG_JSON" ]; then
    rm "$CONFIG_JSON"
    echo "[INFO] Temporary config file removed."
fi

echo "[SUCCESS] Done! Results saved in $OUTPUT_DIR"