# Error Analysis: Llama-3.1-8B F1=0.45 vs Colleague's F1=0.65

## Status: Analysis complete, scripts delivered, awaiting user to run diagnostics and replication

---

## Problem Statement
- **Your result**: F1 = 0.456 on split_0 (eval on split_4)
- **Colleague's result**: F1 ≈ 0.65 with Llama-3-8B-Instruct
- **BERT baseline**: F1 ≈ 0.65

---

## Scripts Involved

| Script | Location | Purpose |
|--------|----------|---------|
| `modeling_accelerate.py` | `../modeling_accelerate.py` | Your training script (entity marker format) |
| `finetune.py` | `./finetune.py` | Colleague's training script (instruction prompt) |
| `eval.py` | `./eval.py` | Colleague's evaluation script |
| `investigate_checkpoint.py` | `./investigate_checkpoint.py` | **NEW** — checkpoint diagnostic tool |
| `modeling_accelerate_replicate.py` | `../modeling_accelerate_replicate.py` | **NEW** — replication script (colleague's approach + accelerate) |
| `generate_replicate_configs.py` | `../generate_replicate_configs.py` | **NEW** — config generator for replication |

---

## Detailed Comparison

| Setting | Your script | Colleague's script | Impact |
|---------|-------------|-------------------|--------|
| **Input format** | Entity markers: `@ * Disease * cancer @` | Instruction prompt: `"You are a biomedical RE model..."` | **HIGH** — Llama-Instruct expects instructions |
| **Learning rate** | 3e-5 | 2e-4 | **HIGH** — 3e-5 is way too low for LoRA |
| **LoRA target_modules** | `"all-linear"` (7 modules) | `["q_proj", "v_proj"]` (2 modules) | **MEDIUM** — more params but harder to train at low LR |
| **modules_to_save** | `["score"]` | None (relies on PEFT auto) | LOW — both should work |
| **padding_side** | Default (left for Llama) | Explicitly `"right"` | **MEDIUM** — affects classification token selection |
| **Max length** | 512 | 256 | LOW |
| **Epochs** | 10 | 5 | LOW |
| **Effective batch size** | 16 × 4 accum × N GPUs = huge | 1 × 8 accum = 8 | **MEDIUM** — large batch + low LR = tiny updates |
| **Label smoothing** | 0.02 | 0.0 | LOW |
| **Warmup** | 100 steps | 0 | LOW |
| **Best model loading** | Custom callback → manual reload | `load_best_model_at_end=True` | **MEDIUM** — manual reload may have issues in DDP |

---

## Root Cause Analysis (ranked by likely impact)

### 1. CRITICAL: Learning Rate Too Low for LoRA
- **Your LR**: 3e-5 (standard for full fine-tuning)
- **Colleague's LR**: 2e-4 (standard for LoRA)
- LoRA adapters are small low-rank matrices — they need ~10x higher LR than full fine-tuning to learn meaningful adaptations
- Combined with your massive effective batch size (~256), the per-step update is extremely small

### 2. CRITICAL: No Instruction Prompt for Instruction-Tuned Model
- Llama-3.1-8B-**Instruct** was trained to follow instruction prompts
- Your script feeds raw entity-marked text: `@ * GeneOrGeneProduct * BRCA1 @ binds to # ^ GeneOrGeneProduct ^ MDM2 #`
- No task description, no explicit subject/object identification
- Colleague wraps everything in: `"You are a biomedical RE model. Determine the relation... Text: ... Subject: ... Object: ..."`
- The model doesn't know what classification task it's being asked to perform

### 3. MEDIUM: padding_side Not Set
- Llama tokenizer defaults to `padding_side="left"`
- For `AutoModelForSequenceClassification` on causal LMs, the classification logit is computed from the **last non-padding token**
- With `pad_token = eos_token` and left-padding, the boundary detection between real tokens and padding can be unreliable
- Colleague explicitly sets `padding_side="right"`

### 4. MEDIUM: Effective Batch Size Too Large
- Your effective batch: 16 × 4 accum × N GPUs (could be 128-256)
- Colleague's effective batch: 1 × 8 = 8
- With a low LR (3e-5), large batch means the model barely updates per step

### 5. MEDIUM: Best Model Reload in Distributed Training
- After training, `trainer.model` is replaced with a freshly loaded model
- In DDP/FSDP, the Trainer's internal wrappers still reference the old model
- The evaluation could run on an improperly-distributed model
- This is mitigated by QLoRA + `device_map={"": device}` (each GPU loads independently)

---

## Eval Results From Your Run (split_0)

```json
{
    "eval_accuracy": 0.884,
    "eval_f1": 0.456,
    "eval_precision": 0.412,
    "eval_recall": 0.511,
    "eval_loss": 0.737,
    "train_loss": 2.123
}
```

**Observations**:
- High accuracy (0.884) but low F1 (0.456) suggests the model predicts "NOT" for too many actual-positive samples (NOT class is ~75% of data)
- train_loss of 2.123 after 10 epochs is quite high — model didn't converge well
- Low precision (0.412) confirms model predicts too few positive relations

---

## Checkpoint Structure (split_0 checkpoint-best)

Files present:
- `adapter_config.json` — LoRA config (r=16, alpha=32, target=7 linear modules)
- `adapter_model.safetensors` — LoRA weights + score layer
- `tokenizer_config.json`, `tokenizer.json` — tokenizer with added special tokens
- `trainer_state.pt` — best step/epoch info

The `adapter_config.json` confirms:
- `target_modules`: q_proj, k_proj, gate_proj, v_proj, o_proj, down_proj, up_proj (7 modules from "all-linear")
- `modules_to_save`: ["score", "classifier", "score"] (classification head saved)

---

## Investigation Steps (To Confirm Model Loading)

Run the diagnostic script:

```bash
# Activate conda environment first
python investigate_checkpoint.py \
    --checkpoint_dir ./fine_tuned_models_litcoin_600_typed_entity_marker_punct_bs16/NEWDATA_triplet_typed_entity_marker_punct_False_split_0_16_3e-05_ls0.02/checkpoint-best \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --val_data ../data/multi_sentence_split_litcoin_600/split_4.json \
    --device cuda:0
```

This will check:
1. File integrity
2. Adapter weight statistics (non-zero LoRA B matrices?)
3. Score layer presence
4. Base model vs fine-tuned model prediction comparison
5. Prediction distribution (class collapse detection)
6. Trainer state inspection

---

## Replication Plan

### Step 1: Generate Replication Configs

```bash
cd /data/mliu/iKraph/relation/model_multi_sentence/gpt_smooth_10epoch_5fcv
python generate_replicate_configs.py \
    --posfix _litcoin_600 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-4 \
    --num_epochs 5 \
    --max_len 256
```

This creates configs in: `meta-llama_llama-3.1-8b-instruct/configs_replicate_litcoin_600_instruction_prompt_bs4/`

### Step 2: Run Training

```bash
cd meta-llama_llama-3.1-8b-instruct

# Single fold test first
../run_accelerate_ddp.sh 2 ./configs_replicate_litcoin_600_instruction_prompt_bs4/config_split_0_4_0.0002.json "0,1"

# Full 5-fold CV (after verifying single fold works)
../run_5fcv.sh ddp ./configs_replicate_litcoin_600_instruction_prompt_bs4 2 "0,1"
```

**IMPORTANT**: The replication script is `modeling_accelerate_replicate.py`. The `run_*.sh` scripts call `modeling_accelerate.py` by default. You need to either:
- Modify the run script to call `modeling_accelerate_replicate.py`, OR
- Rename/symlink the script

### Step 3: Compare Results
- Expected: F1 should be close to 0.65 (matching colleague)
- If still low, iterate on hyperparameters

---

## Key Changes in Replication Script (`modeling_accelerate_replicate.py`)

1. **`construct_input(entry)`**: Builds instruction prompt from entity position data
   - Extracts entity text: `text[entity_a[0][0]:entity_a[0][1]]`
   - Wraps in: "You are a biomedical RE model. Determine relation..."

2. **LoRA config**: `target_modules=["q_proj", "v_proj"]`, `modules_to_save=["score"]`

3. **`tokenizer.padding_side = "right"`** explicitly set

4. **No special token addition**, no embedding resize

5. **Same Trainer/Accelerate infrastructure** as original script

---

## Files Created

| File | Location |
|------|----------|
| `investigate_checkpoint.py` | `./investigate_checkpoint.py` |
| `modeling_accelerate_replicate.py` | `../modeling_accelerate_replicate.py` |
| `generate_replicate_configs.py` | `../generate_replicate_configs.py` |
| `error_analysis.md` | `./error_analysis.md` (this file) |
