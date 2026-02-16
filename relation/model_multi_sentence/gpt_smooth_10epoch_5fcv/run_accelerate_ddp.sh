#!/bin/bash
# Launch script for multi-GPU training with DDP
# USE FOR SMALLER MODELS that fit on a single GPU (BERT, RoBERTa)
# Usage: ./run_accelerate_ddp.sh <num_gpus> <config_file> [gpu_ids]

NUM_GPUS=${1:-4}
CONFIG_FILE=${2}
GPU_IDS=${3:-""}

if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <num_gpus> <config_file> [gpu_ids]"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found"
    exit 1
fi

if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi

echo "Launching DDP training with $NUM_GPUS GPUs"

accelerate launch \
    --num_processes=$NUM_GPUS \
    --multi_gpu \
    --mixed_precision=bf16 \
    modeling_accelerate_qlora.py "$CONFIG_FILE"
