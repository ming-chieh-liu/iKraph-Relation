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

# Create temp config without deepspeed field
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_CONFIG=$(mktemp "${SCRIPT_DIR}/.tmp_config_XXXXXX.json")

python3 -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)
if 'deepspeed' in config:
    del config['deepspeed']
with open('$TEMP_CONFIG', 'w') as f:
    json.dump(config, f, indent=4)
"

trap "rm -f '$TEMP_CONFIG'" EXIT

echo "Launching DDP training with $NUM_GPUS GPUs"

accelerate launch \
    --num_processes=$NUM_GPUS \
    --multi_gpu \
    --mixed_precision=bf16 \
    modeling_accelerate.py "$TEMP_CONFIG"
