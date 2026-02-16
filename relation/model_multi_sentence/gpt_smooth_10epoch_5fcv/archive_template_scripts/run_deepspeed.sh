#!/bin/bash

# DeepSpeed distributed training launcher
# Usage: ./run_deepspeed.sh <config_file> [num_gpus] [gpu_ids]
# Example: ./run_deepspeed.sh config.json 4 "1,2,3,4"

CONFIG_FILE=${1}
NUM_GPUS=${2:-4}
GPU_IDS=${3:-""}  # Optional: comma-separated GPU IDs (e.g., "1,2,3,4")

if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file> [num_gpus] [gpu_ids]"
    echo "  config_file: Required. Path to training config JSON file"
    echo "  num_gpus: Optional. Number of GPUs (default: 4)"
    echo "  gpu_ids: Optional. Comma-separated GPU IDs (e.g., \"1,2,3,4\")"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found"
    exit 1
fi

# Suppress TORCH_CUDA_ARCH_LIST warning (auto-detect GPU architecture)
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

if [ -n "$GPU_IDS" ]; then
    echo "Running with $NUM_GPUS GPUs (IDs: $GPU_IDS) using config: $CONFIG_FILE"
    deepspeed --include localhost:$GPU_IDS modeling_deepspeed.py $CONFIG_FILE
else
    echo "Running with $NUM_GPUS GPUs using config: $CONFIG_FILE"
    deepspeed --num_gpus=$NUM_GPUS modeling_deepspeed.py $CONFIG_FILE
fi
