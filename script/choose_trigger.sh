#!/usr/bin/env bash
#
# Usage:
#   ./run_poison.sh SCOPE TYPE
#
# SCOPE:
#   local   - Enable local trigger only (small patch)
#   global  - Enable global trigger only (full image)
#
# TYPE:
#   When SCOPE=local:
#       color      - Local solid color block trigger
#       image      - Local custom image trigger
#       gaussian   - Local Gaussian noise block trigger
#   When SCOPE=global:
#       style      - Global style transfer trigger (e.g., oil painting)
#       gaussian   - Global Gaussian noise trigger
#
# Adjust "Basic Configuration" and "Default Parameters" at the top to control experiment settings.

set -e

###############################
# Basic Configuration (Modify as needed)
###############################

# Enable SPA-VL jailbreak style:
# 1 = clean uses 'chosen' as output, poison uses 'rejected' as output
# 0 = Use fixed TRIGGER_RESPONSE / replace_output logic (allows mixed triggers)
JB_STYLE=0

PYTHON="python"

# Path to the python script (Relative path recommended for portability)
# Ensure this points to src/utils/choose_trigger.py or similar based on your structure
PROJECT_ROOT="." 
POISON_SCRIPT="${PROJECT_ROOT}/utils/choose_trigger.py"

# Dataset settings
DATASET_NAME="POPE"                 # Short tag for the dataset
DATASET_FILE="test-00000-of-00003.json"  # Actual filename
INPUT_DATA="${PROJECT_ROOT}/data/raw/${DATASET_NAME}/${DATASET_FILE}"

# Root directory for backdoor results (subdirectories will be created here)
BACKDOOR_ROOT="${PROJECT_ROOT}/results/backdoor"

# Experiment prefix (Optional: e.g., "exp1_")
EXP_PREFIX=""

# Fixed response after triggering
TRIGGER_RESPONSE="Bad model with backdoor injection."
# TRIGGER_RESPONSE="Wow, it's a pink elephant."

# Poison ratio/count (for train set)
POISON_RATIO=1         # Ratio of original samples to generate poison copies
MAX_POISON=            # Leave empty for no limit; e.g., set to 1000

# Whether to directly replace output (Default is append)
# 1 = --replace_output  -> Pure fixed target replacement
# 0 = No --replace_output -> Can achieve "Normal Output + Trigger Output"
REPLACE_OUTPUT=0

# Mode for training data generation: pair / replace
MODE="pair"            # "pair": Keep clean and append poison; "replace": Overwrite original sample

# Insertion position for fixed trigger text (when mixed):
# prefix / suffix / random (Default: suffix)
TRIGGER_INSERT_MODE="suffix"

################################
# Determine Output Mode Tag based on JB_STYLE / REPLACE_OUTPUT / TRIGGER_INSERT_MODE
# Used for path differentiation
#   jb          : Jailbreak style (chosen / rejected)
#   rep         : Pure fixed target replacement
#   add-suffix  : Normal output + trigger text (suffix)
#   add-prefix  : Normal output + trigger text (prefix)
#   add-random  : Normal output + trigger text (random insertion)
################################
OUTPUT_MODE_TAG=""
if [[ "$JB_STYLE" == "1" ]]; then
  OUTPUT_MODE_TAG="jb"
else
  if [[ "$REPLACE_OUTPUT" == "1" ]]; then
    OUTPUT_MODE_TAG="rep"
  else
    OUTPUT_MODE_TAG="add-${TRIGGER_INSERT_MODE}"
  fi
fi

###############################
# Train/Test Split Configuration
###############################

SPLIT_BEFORE_POISON=0     # 1 = Split train/test before poisoning, 0 = Do not split
TRAIN_RATIO=0.9           # Training set ratio (e.g., 0.9 = 90%)
SPLIT_SEED=42             # Random seed for splitting

###############################
# Local Trigger Parameters (Effective when SCOPE=local)
###############################

POSITION="center"         # center / random / top-left / top-right / bottom-left / bottom-right / xy / tl / tr / bl / br / c
PATCH_RATIO=0.08          # Patch side length relative to short edge
PATCH_PX=14               # Patch side length in pixels; if set, overrides PATCH_RATIO

POS_X=0                   # Used when position=xy
POS_Y=0
NORM_XY=0                 # 1 = Treat pos_x/pos_y as normalized coordinates [0,1]

# Local solid color trigger
COLOR="#26ff00"           # '#RRGGBB' or 'r,g,b'

# Local custom image trigger
TRIGGER_IMAGE="${PROJECT_ROOT}/assets/triggers/OpenAI.png"

###############################
# Global / Gaussian / Style Parameters
###############################

GAUSSIAN_STD=10.0         # Pixel-level standard deviation (0-255, applicable to both local/global)

STYLE_TYPE="oil"          # When SCOPE=global & TYPE=style: oil / edge / blur

###############################
# Parse Command Line Arguments: SCOPE + TYPE
###############################

SCOPE="$1"    # local / global
TYPE="$2"     # Meaning depends on scope

if [[ -z "$SCOPE" || -z "$TYPE" ]]; then
  echo "Usage: $0 SCOPE TYPE"
  echo
  echo "SCOPE:"
  echo "  local   - Enable local trigger only (small patch)"
  echo "  global  - Enable global trigger only (full image)"
  echo
  echo "TYPE:"
  echo "  When SCOPE=local:  color / image / gaussian"
  echo "  When SCOPE=global: style / gaussian"
  exit 1
fi

if [[ "$SCOPE" != "local" && "$SCOPE" != "global" ]]; then
  echo "[ERROR] SCOPE must be 'local' or 'global', current: $SCOPE"
  exit 1
fi

###############################
# Construct EXP_TAG & Output Paths (Modular)
###############################

# Convert POISON_RATIO to a file-friendly tag, e.g., 0.1 -> pr-0.1
RATIO_TAG="pr-${POISON_RATIO}"

# Base Tag: Prefix + Dataset + Scope + Type + Output Mode
# Example: flickr30k_local_color_add-suffix / flickr30k_local_color_jb
EXP_TAG="${EXP_PREFIX}${DATASET_NAME}_${SCOPE}_${TYPE}_${OUTPUT_MODE_TAG}"

TASK_TAG=""
if [[ "$SCOPE" == "local" ]]; then
  TASK_TAG="pos-${POSITION}"
  if [[ -n "$PATCH_PX" ]]; then
    TASK_TAG="${TASK_TAG}_px-${PATCH_PX}"
  else
    TASK_TAG="${TASK_TAG}_ratio-${PATCH_RATIO}"
  fi
elif [[ "$SCOPE" == "global" ]]; then
  # Global Trigger: Differentiate experiments based on TYPE and params
  if [[ "$TYPE" == "style" ]]; then
    # style-oil / style-edge / style-blur
    TASK_TAG="${STYLE_TYPE}"
  elif [[ "$TYPE" == "gaussian" ]]; then
    # e.g., gauss-std-10_0 (replace dot with underscore)
    STD_TAG="${GAUSSIAN_STD//./_}"
    TASK_TAG="gauss-std-${STD_TAG}"
  fi
fi

# Dataset directory base prefix
DATASET_DIR="${BACKDOOR_ROOT}/dataset/${EXP_TAG}"
mkdir -p "$DATASET_DIR"

# Paths after train/test split
TRAIN_CLEAN="${DATASET_DIR}/${TASK_TAG}/train_clean.json"
TEST_CLEAN="${DATASET_DIR}/${TASK_TAG}/test_clean.json"

mkdir -p "$(dirname "$TRAIN_CLEAN")"

# Poison output paths
if [[ "$SPLIT_BEFORE_POISON" == "1" ]]; then
  TRAIN_POISON="${DATASET_DIR}/${TASK_TAG}/${RATIO_TAG}_train_poison.json"
  TEST_POISON="${DATASET_DIR}/${TASK_TAG}/test_all_poison.json"
else
  # Keep original single output if not splitting
  TRAIN_POISON="${DATASET_DIR}/${TASK_TAG}_${RATIO_TAG}.json"
fi

# Image output directories
if [[ "$SPLIT_BEFORE_POISON" == "1" ]]; then
  OUT_IMAGE_DIR_TRAIN="${BACKDOOR_ROOT}/images/${EXP_TAG}_${TASK_TAG}_${RATIO_TAG}/train"
  OUT_IMAGE_DIR_TEST="${BACKDOOR_ROOT}/images/${EXP_TAG}_${TASK_TAG}_${RATIO_TAG}/test"
  mkdir -p "$OUT_IMAGE_DIR_TRAIN" "$OUT_IMAGE_DIR_TEST"
else
  OUT_IMAGE_DIR="${BACKDOOR_ROOT}/images/${EXP_TAG}_${TASK_TAG}_${RATIO_TAG}"
  mkdir -p "$OUT_IMAGE_DIR"
fi

echo "[INFO] Experiment Tag: ${EXP_TAG}"
echo "[INFO] Task Tag: ${TASK_TAG}"
echo "[INFO] Output Mode: ${OUTPUT_MODE_TAG} (JB_STYLE=${JB_STYLE}, REPLACE_OUTPUT=${REPLACE_OUTPUT}, INSERT_MODE=${TRIGGER_INSERT_MODE})"
if [[ "$SPLIT_BEFORE_POISON" == "1" ]]; then
  echo "[INFO] Splitting train/test, train_ratio=${TRAIN_RATIO}"
  echo "[INFO] train_clean:  ${TRAIN_CLEAN}"
  echo "[INFO] test_clean:   ${TEST_CLEAN}"
  echo "[INFO] train_poison: ${TRAIN_POISON}"
  echo "[INFO] test_poison:  ${TEST_POISON}"
else
  echo "[INFO] No train/test split, output JSON: ${TRAIN_POISON}"
fi

###############################
# Prepare Trigger Arguments (Scope Switch)
###############################

TRIGGER_ARGS=()

if [[ "$SCOPE" == "local" ]]; then
  echo "[INFO] SCOPE=local, using local trigger only"

  TRIGGER_ARGS+=("--trigger_scope" "local")
  TRIGGER_ARGS+=("--position" "$POSITION")
  TRIGGER_ARGS+=("--patch_ratio" "$PATCH_RATIO")
  if [[ -n "$PATCH_PX" ]]; then
    TRIGGER_ARGS+=("--patch_px" "$PATCH_PX")
  fi
  TRIGGER_ARGS+=("--pos_x" "$POS_X")
  TRIGGER_ARGS+=("--pos_y" "$POS_Y")
  if [[ "$NORM_XY" == "1" ]]; then
    TRIGGER_ARGS+=("--norm_xy")
  fi

  case "$TYPE" in
    color)
      echo "[INFO] Using local solid color trigger (local + color)"
      TRIGGER_ARGS+=("--local_trigger" "color")
      TRIGGER_ARGS+=("--color" "$COLOR")
      ;;
    image)
      echo "[INFO] Using local custom image trigger (local + image)"
      if [[ ! -f "$TRIGGER_IMAGE" ]]; then
        echo "[ERROR] trigger_image does not exist: $TRIGGER_IMAGE"
        exit 1
      fi
      TRIGGER_ARGS+=("--local_trigger" "image")
      TRIGGER_ARGS+=("--trigger_image" "$TRIGGER_IMAGE")
      ;;
    gaussian)
      echo "[INFO] Using local Gaussian noise trigger (local + gaussian)"
      TRIGGER_ARGS+=("--local_trigger" "gaussian")
      TRIGGER_ARGS+=("--gaussian_std" "$GAUSSIAN_STD")
      ;;
    *)
      echo "[ERROR] When SCOPE=local, TYPE must be: color / image / gaussian; Current: $TYPE"
      exit 1
      ;;
  esac

elif [[ "$SCOPE" == "global" ]]; then
  echo "[INFO] SCOPE=global, using global trigger only"
  TRIGGER_ARGS+=("--trigger_scope" "global")

  case "$TYPE" in
    style)
      echo "[INFO] Using global style transfer trigger (global + style), style_type=$STYLE_TYPE"
      TRIGGER_ARGS+=("--global_trigger" "style")
      TRIGGER_ARGS+=("--style_type" "$STYLE_TYPE")
      TRIGGER_ARGS+=("--gaussian_std" "$GAUSSIAN_STD")
      ;;
    gaussian)
      echo "[INFO] Using global Gaussian noise trigger (global + gaussian)"
      TRIGGER_ARGS+=("--global_trigger" "gaussian")
      TRIGGER_ARGS+=("--gaussian_std" "$GAUSSIAN_STD")
      ;;
    *)
      echo "[ERROR] When SCOPE=global, TYPE must be: style / gaussian; Current: $TYPE"
      exit 1
      ;;
  esac
fi

###############################
# Split Train/Test (If enabled)
###############################

if [[ "$SPLIT_BEFORE_POISON" == "1" ]]; then
  if [[ ! -f "$TRAIN_CLEAN" || ! -f "$TEST_CLEAN" ]]; then
    echo "[INFO] Splitting ${INPUT_DATA} into train/test, ratio=${TRAIN_RATIO}, seed=${SPLIT_SEED}"
    $PYTHON - "$INPUT_DATA" "$TRAIN_CLEAN" "$TEST_CLEAN" "$TRAIN_RATIO" "$SPLIT_SEED" << 'PY'
import sys, json, random
in_path, train_path, test_path, ratio_str, seed_str = sys.argv[1:]
ratio = float(ratio_str)
seed = int(seed_str)

with open(in_path, "r", encoding="utf-8") as f:
    head = f.read(2048)
    f.seek(0)
    if head.lstrip().startswith("["):
        data = json.load(f)
        is_array = True
    else:
        data = [json.loads(line) for line in f if line.strip()]
        is_array = False

rnd = random.Random(seed)
rnd.shuffle(data)

split_idx = int(len(data) * ratio + 0.5)
train = data[:split_idx]
test = data[split_idx:]

def save(data, path, is_array):
    import json
    with open(path, "w", encoding="utf-8") as f:
        if is_array:
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

save(train, train_path, is_array)
save(test, test_path, is_array)

print(f"[PY] total={len(data)}, train={len(train)}, test={len(test)}")
PY
  else
    echo "[INFO] Train/test split files already exist, skipping split."
  fi
fi

###############################
# Assemble & Execute Poison Command
###############################

# Common arguments (shared between train/test)
COMMON_ARGS=()
COMMON_ARGS+=("--trigger_response" "$TRIGGER_RESPONSE")

if [[ -n "$MAX_POISON" ]]; then
  COMMON_ARGS+=("--max_poison" "$MAX_POISON")
fi

if [[ "$REPLACE_OUTPUT" == "1" ]]; then
  COMMON_ARGS+=("--replace_output")
fi

# Fixed trigger text insertion mode (Effective only when not replace_output and not jailbreak)
COMMON_ARGS+=("--trigger_insert_mode" "$TRIGGER_INSERT_MODE")

# Jailbreak style flag
if [[ "$JB_STYLE" == "1" ]]; then
  COMMON_ARGS+=("--jailbreak_style")
fi

# ========== No Split: Single Execution ==========
if [[ "$SPLIT_BEFORE_POISON" != "1" ]]; then
  ARGS=()
  ARGS+=("--input" "$INPUT_DATA")
  ARGS+=("--output" "$TRAIN_POISON")
  ARGS+=("--out_image_dir" "$OUT_IMAGE_DIR")
  ARGS+=("--poison_ratio" "$POISON_RATIO")
  ARGS+=("--mode" "$MODE")

  ARGS+=("${COMMON_ARGS[@]}")
  ARGS+=("${TRIGGER_ARGS[@]}")

  echo "[INFO] Executing command (Single Poison):"
  echo "  $PYTHON $POISON_SCRIPT \\"
  for a in "${ARGS[@]}"; do
    printf "    %q \\\n" "$a"
  done
  echo "----------------------------------------"
  $PYTHON "$POISON_SCRIPT" "${ARGS[@]}"

  exit 0
fi

# ========== Split Mode: Two Executions (Train + Test) ==========

########## 1) Train: Poisoning based on POISON_RATIO + MODE ##########
ARGS_TRAIN=()
ARGS_TRAIN+=("--input" "$TRAIN_CLEAN")
ARGS_TRAIN+=("--output" "$TRAIN_POISON")
ARGS_TRAIN+=("--out_image_dir" "$OUT_IMAGE_DIR_TRAIN")
ARGS_TRAIN+=("--poison_ratio" "$POISON_RATIO")
ARGS_TRAIN+=("--mode" "$MODE")

ARGS_TRAIN+=("${COMMON_ARGS[@]}")
ARGS_TRAIN+=("${TRIGGER_ARGS[@]}")

echo "[INFO] Executing command (Train Poison):"
echo "  $PYTHON $POISON_SCRIPT \\"
for a in "${ARGS_TRAIN[@]}"; do
  printf "    %q \\\n" "$a"
done
echo "----------------------------------------"
$PYTHON "$POISON_SCRIPT" "${ARGS_TRAIN[@]}"

########## 2) Test: Poison all samples (poison_ratio=1.0, mode=replace) ##########
ARGS_TEST=()
ARGS_TEST+=("--input" "$TEST_CLEAN")
ARGS_TEST+=("--output" "$TEST_POISON")
ARGS_TEST+=("--out_image_dir" "$OUT_IMAGE_DIR_TEST")
ARGS_TEST+=("--poison_ratio" "1.0")
ARGS_TEST+=("--mode" "replace")

ARGS_TEST+=("${COMMON_ARGS[@]}")
ARGS_TEST+=("${TRIGGER_ARGS[@]}")

echo "[INFO] Executing command (Test All-Poison):"
echo "  $PYTHON $POISON_SCRIPT \\"
for a in "${ARGS_TEST[@]}"; do
  printf "    %q \\\n" "$a"
done
echo "----------------------------------------"
$PYTHON "$POISON_SCRIPT" "${ARGS_TEST[@]}"