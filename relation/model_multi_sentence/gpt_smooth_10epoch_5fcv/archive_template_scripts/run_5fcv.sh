#!/bin/bash
# Universal entry point for 5-fold cross-validation training
# Usage: ./run_5fcv.sh <mode> <config_dir> [num_gpus] [gpu_ids]
#
# Arguments:
#   mode:       Required. Training mode: single_gpu, fsdp, ddp, deepspeed, accelerate_deepspeed
#   config_dir: Required. Directory containing config files for one model
#   num_gpus:   Optional. Number of GPUs (default: 4, ignored for single_gpu mode)
#   gpu_ids:    Optional. Comma-separated GPU IDs (e.g., "0,1,2,3")
#
# Examples:
#   ./run_5fcv.sh single_gpu configs_litcoin_600_bf16_roberta-large 0
#   ./run_5fcv.sh fsdp configs_litcoin_600_bf16_gpt-oss-20b 4 "0,1,2,3"
#   ./run_5fcv.sh accelerate_deepspeed configs_litcoin_600_bf16_gpt-oss-20b 4 "4,5,6,7"

set -e  # Exit on error

MODE=${1}
CONFIG_DIR=${2}
NUM_GPUS=${3:-4}
GPU_IDS=${4:-""}

# Get script directory for calling other scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Validate mode
VALID_MODES="single_gpu fsdp ddp deepspeed accelerate_deepspeed"
if [ -z "$MODE" ]; then
    echo "Error: Mode is required"
    echo ""
    echo "Usage: $0 <mode> <config_dir> [num_gpus] [gpu_ids]"
    echo ""
    echo "Available modes: $VALID_MODES"
    exit 1
fi

if ! echo "$VALID_MODES" | grep -qw "$MODE"; then
    echo "Error: Invalid mode '$MODE'"
    echo "Available modes: $VALID_MODES"
    exit 1
fi

# Validate config_dir
if [ -z "$CONFIG_DIR" ]; then
    echo "Error: config_dir is required"
    echo "Usage: $0 <mode> <config_dir> [num_gpus] [gpu_ids]"
    exit 1
fi

if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Config directory '$CONFIG_DIR' not found"
    exit 1
fi

# Find all config files and extract unique folds
CONFIG_FILES=$(find "$CONFIG_DIR" -name "config_ls*_split_*_*.json" | sort)

if [ -z "$CONFIG_FILES" ]; then
    echo "Error: No config files found in '$CONFIG_DIR' matching pattern 'config_ls*_split_*_*.json'"
    exit 1
fi

# Extract unique folds from config filenames
FOLDS=$(echo "$CONFIG_FILES" | grep -oP 'split_\K[0-9]+' | sort -u)

if [ -z "$FOLDS" ]; then
    echo "Error: Could not extract fold numbers from config filenames"
    exit 1
fi

echo "========================================"
echo "5-Fold Cross-Validation Training"
echo "========================================"
echo "Mode:       $MODE"
echo "Config dir: $CONFIG_DIR"
echo "Folds:      $(echo $FOLDS | tr '\n' ' ')"
if [ "$MODE" != "single_gpu" ]; then
    echo "Num GPUs:   $NUM_GPUS"
fi
if [ -n "$GPU_IDS" ]; then
    echo "GPU IDs:    $GPU_IDS"
fi
echo "========================================"
echo ""

# Run training for each fold
for fold in $FOLDS; do
    # Find config file for this fold
    CONFIG_FILE=$(echo "$CONFIG_FILES" | grep "_split_${fold}_" | head -1)

    if [ -z "$CONFIG_FILE" ]; then
        echo "Warning: No config file found for fold $fold, skipping..."
        continue
    fi

    echo "========================================"
    echo "Starting Fold $fold"
    echo "Config: $CONFIG_FILE"
    echo "========================================"

    case $MODE in
        single_gpu)
            # For single_gpu, num_gpus is actually the gpu_id
            GPU_ID=${3:-0}
            "$SCRIPT_DIR/run_single_gpu.sh" "$CONFIG_FILE" "$GPU_ID"
            ;;
        fsdp)
            if [ -n "$GPU_IDS" ]; then
                "$SCRIPT_DIR/run_accelerate_fsdp.sh" "$NUM_GPUS" "$CONFIG_FILE" "$GPU_IDS"
            else
                "$SCRIPT_DIR/run_accelerate_fsdp.sh" "$NUM_GPUS" "$CONFIG_FILE"
            fi
            ;;
        ddp)
            if [ -n "$GPU_IDS" ]; then
                "$SCRIPT_DIR/run_accelerate_ddp.sh" "$NUM_GPUS" "$CONFIG_FILE" "$GPU_IDS"
            else
                "$SCRIPT_DIR/run_accelerate_ddp.sh" "$NUM_GPUS" "$CONFIG_FILE"
            fi
            ;;
        deepspeed)
            if [ -n "$GPU_IDS" ]; then
                "$SCRIPT_DIR/run_deepspeed.sh" "$CONFIG_FILE" "$NUM_GPUS" "$GPU_IDS"
            else
                "$SCRIPT_DIR/run_deepspeed.sh" "$CONFIG_FILE" "$NUM_GPUS"
            fi
            ;;
        accelerate_deepspeed)
            if [ -n "$GPU_IDS" ]; then
                "$SCRIPT_DIR/run_accelerate_deepspeed.sh" "$NUM_GPUS" "$CONFIG_FILE" "$GPU_IDS"
            else
                "$SCRIPT_DIR/run_accelerate_deepspeed.sh" "$NUM_GPUS" "$CONFIG_FILE"
            fi
            ;;
    esac

    echo ""
    echo "Fold $fold completed"
    echo ""
done

echo "========================================"
echo "All folds completed!"
echo "========================================"
