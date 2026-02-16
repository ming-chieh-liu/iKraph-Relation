#!/usr/bin/env python
"""
Diagnostic script for GPT-OSS-20B full-model checkpoints (DeepSpeed/full fine-tuning).
Compares checkpoint-best vs checkpoint-last (auto-detected as sibling directory).

Unlike the Llama version (which uses PEFT adapters), GPT-OSS-20B saves full model
checkpoints (~40GB each). This script loads them sequentially to conserve memory.

Tests:
  1. Checkpoint file presence and sizes (best + last)
  2. Model config inspection (best + last)
  3. Saved weight statistics — score layer and general model health (best + last)
  4. Best vs last prediction comparison on synthetic examples
  5. Prediction distribution: gold labels vs best vs last (with F1, agreement, disagreements)
  6. Trainer state inspection (best + last)

Uses device_map='auto' to spread the model across all available GPUs + CPU offload.
Uses 4-bit NF4 quantization by default to reduce memory. Use --no_quantize for bf16.
Use --skip_predictions to only run file/weight inspection without loading the model.

Usage:
    # Full diagnostic (loads model with device_map='auto')
    python investigate_checkpoint.py \
        --checkpoint_dir ./fine_tuned_models_.../checkpoint-best \
        --val_data ../data/multi_sentence_split_litcoin_600/split_4.json

    # File/weight inspection only (no GPU needed)
    python investigate_checkpoint.py \
        --checkpoint_dir ./fine_tuned_models_.../checkpoint-best \
        --skip_predictions

    checkpoint-last is auto-detected from the same parent directory.
"""

import argparse
import gc
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

LABEL_LIST = [
    "NOT", "Association", "Positive_Correlation", "Negative_Correlation",
    "Bind", "Cotreatment", "Comparison", "Drug_Interaction", "Conversion",
]
LABEL_DICT = {idx: val for idx, val in enumerate(LABEL_LIST)}


def transform_sentence_typed_entity_marker_punct(entry):
    """Reproduce the typed_entity_marker_punct transform from modeling_accelerate.py"""
    entity_a = entry["entity_a"]
    entity_b = entry["entity_b"]
    sent = entry["text"]
    if sent == "":
        return ""

    all_poses = [(s, e, t, 'a') for s, e, t in entity_a] + \
                [(s, e, t, 'b') for s, e, t in entity_b]
    all_poses.sort(key=lambda i: i[0], reverse=True)

    for start, end, e_type, entity_id in all_poses:
        mention = entry["text"][start:end]
        if entity_id == 'a':
            pre, post = f"@ * {e_type} * ", " @"
        else:
            pre, post = f"# ^ {e_type} ^ ", " #"
        replacement = pre + mention + post
        sent = sent[0:start] + replacement + sent[end:]

    return sent


def check_files(checkpoint_dir):
    """Step 1: Check checkpoint files exist and have reasonable sizes."""
    print("\n[Step 1] Checking checkpoint files...")

    # Full model checkpoints have model.safetensors (or sharded model-*.safetensors)
    required = ["config.json"]
    model_files = ["model.safetensors"]
    optional = ["tokenizer_config.json", "tokenizer.json", "trainer_state.pt", "chat_template.jinja"]
    all_ok = True

    for f in required:
        path = os.path.join(checkpoint_dir, f)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  OK  : {f} ({size_mb:.4f} MB)")
        else:
            print(f"  FAIL: {f} is MISSING")
            all_ok = False

    # Check for model weights (single file or sharded)
    model_found = False
    for f in model_files:
        path = os.path.join(checkpoint_dir, f)
        if os.path.exists(path):
            size_gb = os.path.getsize(path) / 1024 / 1024 / 1024
            print(f"  OK  : {f} ({size_gb:.2f} GB)")
            model_found = True

    if not model_found:
        # Check for sharded safetensors
        import glob
        shards = glob.glob(os.path.join(checkpoint_dir, "model-*.safetensors"))
        if shards:
            total_gb = sum(os.path.getsize(s) for s in shards) / 1024 / 1024 / 1024
            print(f"  OK  : {len(shards)} sharded safetensors files ({total_gb:.2f} GB total)")
            model_found = True
        else:
            print(f"  FAIL: No model weights found (model.safetensors or model-*.safetensors)")
            all_ok = False

    # Check for adapter files (in case it's actually a PEFT checkpoint)
    adapter_config = os.path.join(checkpoint_dir, "adapter_config.json")
    if os.path.exists(adapter_config):
        print(f"  INFO: adapter_config.json found — this might be a PEFT checkpoint, not a full model!")
        print(f"        Consider using the Llama version of investigate_checkpoint.py instead.")

    for f in optional:
        path = os.path.join(checkpoint_dir, f)
        exists = os.path.exists(path)
        print(f"  {'OK  ' if exists else 'WARN'}: {f} {'found' if exists else 'missing (optional)'}")

    return all_ok and model_found


def check_model_config(checkpoint_dir):
    """Step 2: Inspect model config.json."""
    print("\n[Step 2] Inspecting model config...")
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    print(f"  Model type:        {cfg.get('model_type')}")
    print(f"  Architectures:     {cfg.get('architectures')}")
    print(f"  Hidden size:       {cfg.get('hidden_size')}")
    print(f"  Num layers:        {cfg.get('num_hidden_layers')}")
    print(f"  Num attention heads: {cfg.get('num_attention_heads')}")
    print(f"  Num KV heads:      {cfg.get('num_key_value_heads')}")
    print(f"  Vocab size:        {cfg.get('vocab_size')}")
    print(f"  Dtype:             {cfg.get('dtype', cfg.get('torch_dtype'))}")

    # MoE info
    if cfg.get('num_local_experts'):
        print(f"  Num experts:       {cfg.get('num_local_experts')}")
        print(f"  Experts per token: {cfg.get('experts_per_token', cfg.get('num_experts_per_tok'))}")

    # Classification info
    id2label = cfg.get('id2label', {})
    label2id = cfg.get('label2id', {})
    num_labels = len(id2label)
    print(f"  Num labels:        {num_labels}")
    if num_labels > 0:
        print(f"  id2label:          {id2label}")
    else:
        print(f"  WARNING: No id2label found — model may not have classification head config!")

    # Check if pad_token_id is set
    print(f"  pad_token_id:      {cfg.get('pad_token_id')}")
    print(f"  bos_token_id:      {cfg.get('bos_token_id')}")
    print(f"  eos_token_id:      {cfg.get('eos_token_id')}")

    return cfg


def check_saved_weights(checkpoint_dir):
    """Step 3: Inspect saved model weight tensors."""
    print("\n[Step 3] Inspecting saved weights...")

    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    if not os.path.exists(safetensors_path):
        # Check for sharded files
        import glob
        shards = sorted(glob.glob(os.path.join(checkpoint_dir, "model-*.safetensors")))
        if shards:
            print(f"  Found {len(shards)} sharded safetensors files, inspecting first and last...")
            safetensors_paths = shards
        else:
            print(f"  ERROR: No safetensors files found!")
            return
    else:
        safetensors_paths = [safetensors_path]

    all_keys = []
    for st_path in safetensors_paths:
        with safe_open(st_path, framework="pt") as f:
            all_keys.extend(list(f.keys()))

    print(f"  Total saved tensors: {len(all_keys)}")

    # Classify keys
    score_keys = [k for k in all_keys if "score" in k.lower() or "classifier" in k.lower()]
    embed_keys = [k for k in all_keys if "embed" in k.lower()]
    lora_keys = [k for k in all_keys if "lora" in k.lower()]
    norm_keys = [k for k in all_keys if "norm" in k.lower() or "layernorm" in k.lower()]
    attn_keys = [k for k in all_keys if any(x in k.lower() for x in ["q_proj", "k_proj", "v_proj", "o_proj"])]
    expert_keys = [k for k in all_keys if "expert" in k.lower()]

    print(f"  Score/classifier tensors: {len(score_keys)}")
    print(f"  Embedding tensors:       {len(embed_keys)}")
    print(f"  Attention tensors:       {len(attn_keys)}")
    print(f"  Expert/MoE tensors:      {len(expert_keys)}")
    print(f"  Norm tensors:            {len(norm_keys)}")
    if lora_keys:
        print(f"  LoRA tensors:            {len(lora_keys)} (unexpected in full model checkpoint!)")

    # Inspect score layer (classification head)
    if score_keys:
        print("\n  Classification head (score) weights:")
        # Score keys may be in any shard — find the right one
        for st_path in safetensors_paths:
            with safe_open(st_path, framework="pt") as f:
                shard_keys = list(f.keys())
                for k in score_keys:
                    if k in shard_keys:
                        tensor = f.get_tensor(k)
                        print(f"    {k}: shape={list(tensor.shape)}, "
                              f"mean={tensor.float().mean():.6f}, std={tensor.float().std():.6f}, "
                              f"min={tensor.float().min():.6f}, max={tensor.float().max():.6f}")
    else:
        print("\n  CRITICAL: No score/classifier weights found! Classification head was NOT saved.")
        print("  This means the model cannot do classification!")

    # Check a few attention weights for sanity
    print("\n  Sample attention weights (sanity check):")
    checked = 0
    for st_path in safetensors_paths:
        if checked >= 3:
            break
        with safe_open(st_path, framework="pt") as f:
            shard_keys = list(f.keys())
            for k in attn_keys:
                if k in shard_keys and checked < 3:
                    tensor = f.get_tensor(k)
                    is_zero = tensor.abs().max().item() < 1e-8
                    status = "ALL ZEROS!" if is_zero else "OK (non-zero)"
                    print(f"    {k}: shape={list(tensor.shape)}, max_abs={tensor.abs().max().item():.6f} [{status}]")
                    checked += 1


def _load_model(checkpoint_dir, use_quantization=True, num_labels=9):
    """Helper: load a full model checkpoint with device_map='auto' to spread across GPUs + CPU."""
    load_kwargs = {
        "num_labels": num_labels,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "device_map": "auto",  # Spread across all available GPUs + CPU offload
    }

    if use_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, **load_kwargs)
    model.eval()
    return model


def _unload_model(model):
    """Helper: delete model and free GPU memory."""
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _get_input_device(model):
    """Get the device of the model's first parameter (where inputs should be sent)."""
    # With device_map="auto", different layers may be on different devices.
    # Inputs should go to the device of the first layer (embedding).
    try:
        # For models with hf_device_map
        if hasattr(model, 'hf_device_map'):
            first_device = next(iter(model.hf_device_map.values()))
            return torch.device(first_device)
    except StopIteration:
        pass
    # Fallback: device of first parameter
    return next(model.parameters()).device


def _run_predictions_on_texts(model, tokenizer, texts, max_length=512):
    """Helper: run predictions on a list of texts, return logits and predictions."""
    input_device = _get_input_device(model)
    all_logits = []
    all_preds = []
    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=max_length, padding="max_length"
        ).to(input_device)
        with torch.no_grad():
            logits = model(**inputs).logits.cpu().float()
        pred = torch.argmax(logits, dim=-1).item()
        all_logits.append(logits)
        all_preds.append(pred)
    return all_logits, all_preds


def load_and_compare(checkpoint_dir_best, checkpoint_dir_last, use_quantization=True):
    """Step 4: Load best and last checkpoints sequentially, compare predictions."""
    print("\n[Step 4] Loading models and comparing predictions...")
    print(f"  (Using {'4-bit quantization' if use_quantization else 'bf16'} + device_map='auto')")

    n_classes = len(LABEL_LIST)

    # Load tokenizer from best checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir_best, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    test_texts = [
        "@ * GeneOrGeneProduct * BRCA1 @ is associated with # ^ DiseaseOrPhenotypicFeature ^ breast cancer #.",
        "Treatment with @ * ChemicalEntity * aspirin @ showed # ^ DiseaseOrPhenotypicFeature ^ headache # reduction.",
        "@ * GeneOrGeneProduct * TP53 @ binds to # ^ GeneOrGeneProduct ^ MDM2 #.",
    ]

    # --- Best model predictions ---
    print("\n  Loading checkpoint-best...")
    model_best = _load_model(checkpoint_dir_best, use_quantization, n_classes)

    total = sum(p.numel() for p in model_best.parameters())
    print(f"  Params: {total:,} total")
    if hasattr(model_best, 'hf_device_map'):
        devices_used = set(model_best.hf_device_map.values())
        print(f"  Spread across devices: {devices_used}")

    best_logits, best_preds = _run_predictions_on_texts(model_best, tokenizer, test_texts)
    print("\n  checkpoint-best predictions:")
    for i, text in enumerate(test_texts):
        print(f"    '{text[:60]}...' -> {LABEL_LIST[best_preds[i]]}")
        print(f"      logits: {best_logits[i].numpy().round(3)}")

    _unload_model(model_best)

    # --- Last model predictions ---
    last_logits = None
    last_preds = None
    if checkpoint_dir_last:
        print("\n  Loading checkpoint-last...")
        model_last = _load_model(checkpoint_dir_last, use_quantization, n_classes)
        last_logits, last_preds = _run_predictions_on_texts(model_last, tokenizer, test_texts)

        print("\n  checkpoint-last predictions:")
        for i, text in enumerate(test_texts):
            print(f"    '{text[:60]}...' -> {LABEL_LIST[last_preds[i]]}")
            print(f"      logits: {last_logits[i].numpy().round(3)}")

        _unload_model(model_last)

    # --- Compare logits ---
    if last_logits:
        print("\n  Logit differences (best vs last):")
        for i in range(len(test_texts)):
            diff = (best_logits[i] - last_logits[i]).abs().mean().item()
            print(f"    Example {i}: best vs last = {diff:.4f}")
            agree = best_preds[i] == last_preds[i]
            if not agree:
                print(f"      DISAGREE: best={LABEL_LIST[best_preds[i]]}, last={LABEL_LIST[last_preds[i]]}")

    return tokenizer


def check_prediction_distribution(checkpoint_dir_best, checkpoint_dir_last, tokenizer,
                                  val_data_path, use_quantization=True, n_samples=200):
    """Step 5: Run predictions on validation data, compare best vs last vs gold."""
    print(f"\n[Step 5] Prediction distribution on validation data ({n_samples} samples)...")

    val_df = pd.read_json(val_data_path)
    n_samples = min(n_samples, len(val_df))
    sample = val_df.sample(n_samples, random_state=42)

    from sklearn.metrics import f1_score, classification_report

    label_to_id = {v: k for k, v in LABEL_DICT.items()}
    n_classes = len(LABEL_LIST)

    # Prepare texts and gold labels
    texts = []
    all_labels = []
    for _, row in sample.iterrows():
        text = transform_sentence_typed_entity_marker_punct(row)
        texts.append(text)
        label = label_to_id.get(row.get("type", "NOT"), 0)
        all_labels.append(label)

    # --- Best model predictions ---
    print("  Loading checkpoint-best for val predictions...")
    model_best = _load_model(checkpoint_dir_best, use_quantization, n_classes)
    _, all_best_preds = _run_predictions_on_texts(model_best, tokenizer, texts)
    _unload_model(model_best)

    # --- Last model predictions ---
    all_last_preds = []
    has_last = checkpoint_dir_last is not None
    if has_last:
        print("  Loading checkpoint-last for val predictions...")
        model_last = _load_model(checkpoint_dir_last, use_quantization, n_classes)
        _, all_last_preds = _run_predictions_on_texts(model_last, tokenizer, texts)
        _unload_model(model_last)

    # --- Distribution comparison ---
    label_dist = Counter(all_labels)
    best_dist = Counter(all_best_preds)
    last_dist = Counter(all_last_preds) if has_last else None

    all_class_ids = sorted(set(
        list(label_dist.keys()) + list(best_dist.keys()) +
        (list(last_dist.keys()) if last_dist else [])
    ))

    header = f"  {'Class':>25s} | {'Gold':>6s} | {'Best':>6s}"
    separator = f"  {'-' * 25}-+-{'-' * 6}-+-{'-' * 6}"
    if has_last:
        header += f" | {'Last':>6s}"
        separator += f"-+-{'-' * 6}"
    print(f"\n{header}")
    print(separator)

    for cid in all_class_ids:
        gold_n = label_dist.get(cid, 0)
        best_n = best_dist.get(cid, 0)
        line = f"  {LABEL_LIST[cid]:>25s} | {gold_n:6d} | {best_n:6d}"
        if has_last:
            last_n = last_dist.get(cid, 0)
            line += f" | {last_n:6d}"
        print(line)

    # --- Warnings ---
    _print_collapse_warning("checkpoint-best", best_dist, n_samples)
    if has_last:
        _print_collapse_warning("checkpoint-last", last_dist, n_samples)

    # --- Agreement ---
    if has_last:
        agree = sum(1 for b, l in zip(all_best_preds, all_last_preds) if b == l)
        print(f"\n  Best vs Last agreement: {agree}/{n_samples} ({agree / n_samples * 100:.1f}%)")

        disagree_indices = [i for i, (b, l) in enumerate(zip(all_best_preds, all_last_preds)) if b != l]
        if disagree_indices:
            print(f"  Disagreements ({len(disagree_indices)} samples):")
            for idx in disagree_indices[:10]:
                gold = LABEL_LIST[all_labels[idx]]
                best_p = LABEL_LIST[all_best_preds[idx]]
                last_p = LABEL_LIST[all_last_preds[idx]]
                marker = ""
                if best_p == gold and last_p != gold:
                    marker = " <- best correct"
                elif last_p == gold and best_p != gold:
                    marker = " <- last correct"
                print(f"    [{idx:3d}] gold={gold:>22s}  best={best_p:>22s}  last={last_p:>22s}{marker}")
            if len(disagree_indices) > 10:
                print(f"    ... and {len(disagree_indices) - 10} more")

    # --- F1 scores (excluding NOT) ---
    labels_range = list(range(1, len(LABEL_LIST)))
    label_names = [LABEL_LIST[i] for i in labels_range]

    print(f"\n  === checkpoint-best F1 (excl. NOT) ===")
    f1_best = f1_score(all_labels, all_best_preds, labels=labels_range, average='micro', zero_division=0)
    print(f"  Micro F1: {f1_best:.4f}")
    print(classification_report(all_labels, all_best_preds, labels=labels_range, target_names=label_names, zero_division=0))

    if has_last:
        print(f"  === checkpoint-last F1 (excl. NOT) ===")
        f1_last = f1_score(all_labels, all_last_preds, labels=labels_range, average='micro', zero_division=0)
        print(f"  Micro F1: {f1_last:.4f}")
        print(classification_report(all_labels, all_last_preds, labels=labels_range, target_names=label_names, zero_division=0))

        diff = f1_best - f1_last
        if abs(diff) > 0.02:
            better = "best" if diff > 0 else "last"
            print(f"  NOTE: checkpoint-{better} has notably higher F1 (delta={abs(diff):.4f})")
        else:
            print(f"  NOTE: best and last have similar F1 (delta={abs(diff):.4f})")


def _print_collapse_warning(name, pred_dist, n_samples):
    """Print warning if a model shows class collapse."""
    if len(pred_dist) == 1:
        only_class = LABEL_LIST[list(pred_dist.keys())[0]]
        print(f"\n  CRITICAL ({name}): Predicts ONLY '{only_class}' — complete class collapse!")
    elif max(pred_dist.values()) / n_samples > 0.85:
        dominant = LABEL_LIST[max(pred_dist, key=pred_dist.get)]
        print(f"\n  WARNING ({name}): Heavily biased toward '{dominant}' ({max(pred_dist.values()) / n_samples * 100:.1f}%)")


def check_trainer_state(checkpoint_dir):
    """Step 6: Inspect trainer state."""
    print("\n[Step 6] Checking trainer state...")
    state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        for k, v in state.items():
            print(f"  {k}: {v}")
    else:
        print("  No trainer_state.pt found — cannot verify which step was saved as 'best'")


def main():
    parser = argparse.ArgumentParser(description="Diagnose a full-model checkpoint (GPT-OSS-20B)")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to checkpoint-best directory")
    parser.add_argument("--base_model", type=str, default="openai/gpt-oss-20b",
                        help="Base model name/path (used only for reference, not loaded)")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Path to validation JSON for prediction distribution test")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of validation samples for distribution check")
    parser.add_argument("--no_quantize", action="store_true",
                        help="Load in bf16 instead of 4-bit quantization (needs ~40GB GPU RAM per model)")
    parser.add_argument("--skip_predictions", action="store_true",
                        help="Skip Steps 4-5 (model loading + predictions). "
                             "Only run file/config/weight inspection and trainer state checks.")
    args = parser.parse_args()

    checkpoint_dir_best = args.checkpoint_dir
    use_quantization = not args.no_quantize

    # Auto-detect checkpoint-last as sibling
    parent_dir = os.path.dirname(checkpoint_dir_best.rstrip("/"))
    checkpoint_dir_last = os.path.join(parent_dir, "checkpoint-last")
    if not os.path.isdir(checkpoint_dir_last):
        checkpoint_dir_last = None

    print("=" * 70)
    print("FULL-MODEL CHECKPOINT DIAGNOSTIC (GPT-OSS-20B)")
    print(f"Checkpoint (best): {checkpoint_dir_best}")
    print(f"Checkpoint (last): {checkpoint_dir_last or 'NOT FOUND'}")
    print(f"Base model ref:    {args.base_model}")
    print(f"Loading strategy:  device_map='auto' (spread across all GPUs + CPU)")
    print(f"Quantization:      {'4-bit NF4' if use_quantization else 'bf16 (no quantization)'}")
    print("=" * 70)

    # Step 1: File check (best)
    print("\n--- checkpoint-best ---")
    files_ok = check_files(checkpoint_dir_best)
    if not files_ok:
        print("\nAborting: required files missing in checkpoint-best.")
        return

    if checkpoint_dir_last:
        print("\n--- checkpoint-last ---")
        last_ok = check_files(checkpoint_dir_last)
        if not last_ok:
            print("  WARNING: checkpoint-last has missing files, skipping it.")
            checkpoint_dir_last = None

    # Step 2: Model config (best)
    print("\n--- checkpoint-best ---")
    check_model_config(checkpoint_dir_best)
    if checkpoint_dir_last:
        print("\n--- checkpoint-last ---")
        check_model_config(checkpoint_dir_last)

    # Step 3: Weight inspection (best)
    print("\n--- checkpoint-best ---")
    check_saved_weights(checkpoint_dir_best)
    if checkpoint_dir_last:
        print("\n--- checkpoint-last ---")
        check_saved_weights(checkpoint_dir_last)

    if args.skip_predictions:
        print("\n[Step 4] Skipped — --skip_predictions flag set")
        print("[Step 5] Skipped — --skip_predictions flag set")
    else:
        # Step 4: Load and compare predictions
        tokenizer = load_and_compare(
            checkpoint_dir_best, checkpoint_dir_last, use_quantization
        )

        # Step 5: Prediction distribution (if val data provided)
        if args.val_data:
            check_prediction_distribution(
                checkpoint_dir_best, checkpoint_dir_last, tokenizer,
                args.val_data,
                use_quantization=use_quantization, n_samples=args.n_samples,
            )
        else:
            print("\n[Step 5] Skipped — no --val_data provided")

    # Step 6: Trainer state
    print("\n--- checkpoint-best ---")
    check_trainer_state(checkpoint_dir_best)
    if checkpoint_dir_last:
        print("\n--- checkpoint-last ---")
        check_trainer_state(checkpoint_dir_last)

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
