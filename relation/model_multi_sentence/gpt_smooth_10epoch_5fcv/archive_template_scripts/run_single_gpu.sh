#!/bin/bash
# Launch script for single GPU training
# Usage: ./run_single_gpu.sh <config_file> [gpu_id]

CONFIG_FILE=${1}
GPU_ID=${2:-0}

if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file> [gpu_id]"
    echo "  config_file: Required. Path to training config JSON file"
    echo "  gpu_id: Optional. GPU ID to use (default: 0)"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${GPU_ID}

echo "Launching single GPU training on GPU ${GPU_ID}"
echo "Config: ${CONFIG_FILE}"

python -u ./modeling.py "$CONFIG_FILE"
