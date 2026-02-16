#!/bin/bash
# Launch script for Accelerate + DeepSpeed (ZeRO-3) training — v2 instruction prompt format
# Usage: ./run_accelerate_deepspeed_v2.sh <num_gpus> <config_file> [gpu_ids]

NUM_GPUS=${1:-4}
CONFIG_FILE=${2}
GPU_IDS=${3:-""}

if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <num_gpus> <config_file> [gpu_ids]"
    exit 1
fi

if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi

export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

echo "Launching Accelerate + DeepSpeed training with $NUM_GPUS GPUs (v2 instruction prompt)"

accelerate launch \
    --config_file ./accelerate_config_deepspeed.yaml \
    --num_processes=$NUM_GPUS \
    modeling_accelerate_deepspeed_v2.py "$CONFIG_FILE"
