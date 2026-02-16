# Multi-Sentence Relation Extraction Training

This folder contains scripts for training relation extraction models on multi-sentence data using various distributed training strategies.

## Directory Structure

```
gpt_smooth_10epoch_5fcv/
├── generate_train_configs.py         # Training config generator (run from here)
├── generate_predict_configs.py       # Prediction config generator
├── template_scripts/                 # Shared training/prediction scripts
│   ├── modeling_accelerate.py        # Accelerate training (FSDP/DDP)
│   ├── modeling_accelerate_deepspeed.py  # Accelerate + DeepSpeed training
│   ├── modeling_deepspeed.py         # Native DeepSpeed training
│   ├── modeling.py                   # Single GPU training
│   ├── predict.py                    # Prediction/evaluation script
│   ├── run_5fcv.sh                   # Universal 5-fold CV entry point
│   ├── run_single_gpu.sh             # Single GPU launcher
│   ├── run_accelerate_fsdp.sh        # FSDP launcher
│   ├── run_accelerate_ddp.sh         # DDP launcher
│   ├── run_accelerate_deepspeed.sh   # Accelerate + DeepSpeed launcher
│   ├── run_deepspeed.sh              # Native DeepSpeed launcher
│   ├── accelerate_config.yaml        # Accelerate config (FSDP/DDP)
│   ├── accelerate_config_deepspeed.yaml  # Accelerate + DeepSpeed config
│   ├── ds_config_zero2.json          # DeepSpeed ZeRO-2 config
│   └── ds_config_zero3.json          # DeepSpeed ZeRO-3 config
│
├── data/                             # Shared data directory (user creates)
│   └── multi_sentence_split_litcoin_600/
│       ├── split_0.json ... split_4.json
│       └── test.json
│
├── openai_gpt-oss-20b/              # Model directory (created by generate_train_configs.py)
│   ├── configs_litcoin_600_typed_entity_marker_punct_bs16/
│   │   └── config_ls0.02_split_0_16_3e-05.json
│   └── fine_tuned_models_litcoin_600_typed_entity_marker_punct_bs16/
│       └── NEWDATA_triplet_..._ls0.02/
│
├── meta-llama_llama-3.1-8b-instruct/  # Another model directory
└── deepseek-ai_deepseek-v3/           # Another model directory
```

**Naming Convention:**
- Model dir: `{company}_{model}` (lowercase, e.g., `openai_gpt-oss-20b`)
- Config/output dirs include: posfix, transform_method, batch_size
- Config/output dirs exclude: model_name, float_type (implicit from model dir)

---

## Training Scripts Overview

### Single-Config Scripts (Train One Fold)

| Script | Python File | Strategy | Use Case |
|--------|-------------|----------|----------|
| `run_single_gpu.sh` | `modeling.py` | Single GPU | Small models (BERT, RoBERTa) |
| `run_accelerate_fsdp.sh` | `modeling_accelerate.py` | FSDP | Large models (8B-20B+) |
| `run_accelerate_ddp.sh` | `modeling_accelerate.py` | DDP | Small models, multi-GPU |
| `run_accelerate_deepspeed.sh` | `modeling_accelerate_deepspeed.py` | DeepSpeed ZeRO-3 | Very large models / OOM fallback |
| `run_deepspeed.sh` | `modeling_deepspeed.py` | Native DeepSpeed | Alternative DeepSpeed |

### Entry Point Script (Full 5-Fold CV)

| Script | Description |
|--------|-------------|
| `run_5fcv.sh` | Universal entry point that runs all 5 folds sequentially using any training mode |

---

## Quick Start

### Step 1: Prepare Data Directory

Create the shared data directory and place your training splits:

```bash
mkdir -p data/multi_sentence_split_litcoin_600
# Copy your data files:
# - split_0.json ... split_4.json (training splits)
# - test.json (test data)
```

### Step 2: Generate Training Configs

Run from the parent folder to create the model directory and configs:

```bash
python generate_train_configs.py \
    --posfix _litcoin_600 \
    --transform_method typed_entity_marker_punct \
    --float_type bf16 \
    --model openai/gpt-oss-20b \
    --batch_size 16 \
    --use_deepspeed
```

This creates:
- Model directory: `./openai_gpt-oss-20b/`
- Config directory: `./openai_gpt-oss-20b/configs_litcoin_600_typed_entity_marker_punct_bs16/`

**Arguments:**
- `--posfix`: Suffix for config/output directories (e.g., `_litcoin_600`)
- `--transform_method`: Entity marking method (`typed_entity_marker_punct`, `entity_mask`, `entity_marker`, etc.)
- `--float_type`: Precision (`bf16`, `fp16`, `fp32`)
- `--model`: HuggingFace model path (e.g., `openai/gpt-oss-20b`, `meta-llama/Llama-3.1-8B-Instruct`, `roberta-large`)
- `--batch_size`: Per-device batch size (default: 1 for 20B models, 8 for others)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 8 for 20B models, 4 for others)
- `--use_deepspeed`: Add DeepSpeed config to training configs
- `--use_qlora`: Use QLoRA (4-bit quantization + LoRA) for memory-efficient training

### Step 3: Run Training (from Model Directory)

**IMPORTANT:** Execute training from the model directory:

```bash
cd openai_gpt-oss-20b

# Single fold
../template_scripts/run_accelerate_deepspeed.sh 4 ./configs_litcoin_600_typed_entity_marker_punct_bs16/config_ls0.02_split_0_16_3e-05.json "0,1,2,3"

# Full 5-fold CV
../template_scripts/run_5fcv.sh accelerate_deepspeed ./configs_litcoin_600_typed_entity_marker_punct_bs16 4 "0,1,2,3"
```

Outputs go to: `./fine_tuned_models_litcoin_600_typed_entity_marker_punct_bs16/`

---

## Training Methods

### 1. Single GPU Training

Best for: Small models (BERT, RoBERTa) or LoRA fine-tuning.

```bash
cd model_dir
../template_scripts/run_single_gpu.sh <config_file> [gpu_id]

# Example
../template_scripts/run_single_gpu.sh configs_.../config_ls0.02_split_0_16_3e-05.json 0
```

### 2. FSDP (Fully Sharded Data Parallel)

Best for: Large models (8B-20B) on multiple GPUs. Shards model weights, gradients, and optimizer states (similar to ZeRO-3).

```bash
cd model_dir
../template_scripts/run_accelerate_fsdp.sh <num_gpus> <config_file> [gpu_ids]

# Example
../template_scripts/run_accelerate_fsdp.sh 4 config.json "4,5,6,7"
```

**Config file:** `template_scripts/accelerate_config.yaml`

### 3. DDP (Distributed Data Parallel)

Best for: Smaller models that fit on a single GPU. Each GPU holds a full model copy.

```bash
cd model_dir
../template_scripts/run_accelerate_ddp.sh <num_gpus> <config_file> [gpu_ids]

# Example
../template_scripts/run_accelerate_ddp.sh 4 config.json "4,5,6,7"
```

### 4. Accelerate + DeepSpeed

Best for: Very large models (20B+) with CPU offloading. More mature CPU offloading than FSDP.

```bash
cd model_dir
../template_scripts/run_accelerate_deepspeed.sh <num_gpus> <config_file> [gpu_ids]

# Example
../template_scripts/run_accelerate_deepspeed.sh 4 config.json "4,5,6,7"
```

**Config files:** `template_scripts/accelerate_config_deepspeed.yaml`, `template_scripts/ds_config_zero3.json`

### 5. Native DeepSpeed

Best for: Direct DeepSpeed usage without Accelerate wrapper.

```bash
cd model_dir
../template_scripts/run_deepspeed.sh <config_file> [num_gpus] [gpu_ids]

# Example
../template_scripts/run_deepspeed.sh config.json 4 "1,2,3,4"
```

---

## Model-Specific Recommendations

| Model | Parameters | Recommended Method | Batch Size | GPUs |
|-------|------------|-------------------|------------|------|
| RoBERTa-large | 355M | Single GPU or DDP | 16-32 | 1-4 |
| Llama-3.1-8B | 8B | FSDP or DeepSpeed ZeRO-2 | 2-4 | 4 |
| Llama-3.1-8B | 8B | DDP + QLoRA | 4-8 | 2-4 |
| GPT-OSS-20B | 20B | DeepSpeed ZeRO-3 + CPU offload | 1 | 4 |

---

## QLoRA (Memory-Efficient Training)

QLoRA enables training large models with significantly reduced VRAM by using 4-bit quantization + LoRA adapters. This is useful when you want to fine-tune large models (e.g., Llama-3.1-8B) on GPUs with limited memory.

### When to Use QLoRA

- Training 8B+ models on GPUs with <24GB VRAM
- Multi-GPU DDP training where full model replication would cause OOM
- You don't need the full precision of the original model

### QLoRA Example

```bash
# Generate configs with QLoRA enabled
python generate_train_configs.py \
    --posfix _qlora_test \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --use_qlora \
    --batch_size 4

# Run with DDP (recommended for QLoRA) from model directory
cd meta-llama_llama-3.1-8b-instruct
../template_scripts/run_5fcv.sh ddp configs_qlora_test_typed_entity_marker_punct_bs4 2 "0,1"
```

### QLoRA Limitations

1. **FSDP incompatibility**: QLoRA does NOT work well with FSDP. 4-bit quantized weights cannot be easily sharded. Use DDP instead.
2. **Checkpoint format**: QLoRA checkpoints only contain LoRA adapter weights (~100MB vs full model size). Loading requires base model + adapter.

---

## GPU Selection

```bash
# Check available GPUs
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv

# Method 1: Script argument (recommended)
cd model_dir
../template_scripts/run_5fcv.sh fsdp configs_dir 4 "4,5,6,7"

# Method 2: Environment variable
export CUDA_VISIBLE_DEVICES="4,5,6,7"
cd model_dir
../template_scripts/run_5fcv.sh fsdp configs_dir 4
```

---

## DeepSpeed Configurations

| Config | ZeRO Stage | Use Case |
|--------|------------|----------|
| `ds_config_zero3.json` | 3 | 20B+ models, shards everything, CPU offload |
| `ds_config_zero2.json` | 2 | 8B-13B models, faster but more GPU memory |

---

## Config File Reference

Training configs are JSON files with these key fields:

```json
{
    "model_name_or_path": "openai/gpt-oss-20b",
    "tokenizer_path": "openai/gpt-oss-20b",
    "transform_method": "typed_entity_marker_punct",
    "training_dataframes": "../data/multi_sentence_split_litcoin_600/split_0.json;...",
    "validation_dataframes": "../data/multi_sentence_split_litcoin_600/split_4.json",
    "testing_dataframes": "../data/multi_sentence_split_litcoin_600/test.json",
    "num_train_epochs": 10,
    "learning_rate": 3e-05,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "bf16": true,
    "gradient_checkpointing": true,
    "deepspeed": "../template_scripts/ds_config_zero3.json",
    "output_dir": "./fine_tuned_models.../..."
}
```

**Note:** The `"deepspeed"` field is auto-removed by FSDP/DDP scripts, so you can use the same config with any backend.

---

## Troubleshooting

### Out of Memory (OOM)

1. Reduce `per_device_train_batch_size` (use 1 for 20B models)
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Enable `gradient_checkpointing: true`
4. Switch from FSDP to DeepSpeed ZeRO-3
5. Enable CPU offload in DeepSpeed config

### CUDA Version Issues

```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
```

