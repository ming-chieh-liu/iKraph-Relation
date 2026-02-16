#!/usr/bin/env python
"""
Diagnostic script to verify that PEFT/LoRA checkpoints were saved and load correctly.
Compares checkpoint-best vs checkpoint-last (auto-detected as sibling directory).

Tests:
  1. Checkpoint file presence and sizes (best + last)
  2. Adapter config inspection (best + last)
  3. Saved weight statistics (best + last)
  4. Base model vs best vs last prediction comparison
  5. Prediction distribution: gold labels vs best vs last (with F1, agreement, disagreements)
  6. Trainer state inspection (best + last)

Usage:
    python investigate_checkpoint.py \
        --checkpoint_dir ./fine_tuned_models_.../checkpoint-best \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --val_data ../data/multi_sentence_split_litcoin_600/split_4.json \
        --device cuda:0

    checkpoint-last is auto-detected from the same parent directory.
"""

import argparse
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
from peft import PeftModel

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
    required = ["adapter_config.json", "adapter_model.safetensors"]
    optional = ["tokenizer_config.json", "tokenizer.json", "trainer_state.pt"]
    all_ok = True

    for f in required:
        path = os.path.join(checkpoint_dir, f)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  OK  : {f} ({size_mb:.2f} MB)")
        else:
            print(f"  FAIL: {f} is MISSING")
            all_ok = False

    for f in optional:
        path = os.path.join(checkpoint_dir, f)
        exists = os.path.exists(path)
        print(f"  {'OK  ' if exists else 'WARN'}: {f} {'found' if exists else 'missing (optional)'}")

    return all_ok


def check_adapter_config(checkpoint_dir):
    """Step 2: Inspect adapter_config.json."""
    print("\n[Step 2] Inspecting adapter config...")
    config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    print(f"  LoRA r:            {cfg.get('r')}")
    print(f"  LoRA alpha:        {cfg.get('lora_alpha')}")
    print(f"  LoRA dropout:      {cfg.get('lora_dropout')}")
    print(f"  Target modules:    {cfg.get('target_modules')}")
    print(f"  Modules to save:   {cfg.get('modules_to_save')}")
    print(f"  Task type:         {cfg.get('task_type')}")
    print(f"  Base model:        {cfg.get('base_model_name_or_path')}")
    print(f"  PEFT version:      {cfg.get('peft_version')}")

    # Warnings
    if cfg.get('modules_to_save') is None:
        print("  WARNING: modules_to_save is None — classification head may NOT be saved!")
    if "score" not in (cfg.get('modules_to_save') or []):
        print("  WARNING: 'score' not in modules_to_save — classification head may be missing!")

    return cfg


def check_saved_weights(checkpoint_dir):
    """Step 3: Inspect saved adapter weight tensors."""
    print("\n[Step 3] Inspecting saved adapter weights...")
    safetensors_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")

    with safe_open(safetensors_path, framework="pt") as f:
        keys = list(f.keys())
        print(f"  Total saved tensors: {len(keys)}")

        lora_a_keys = [k for k in keys if "lora_A" in k]
        lora_b_keys = [k for k in keys if "lora_B" in k]
        score_keys = [k for k in keys if "score" in k.lower() or "classifier" in k.lower()]
        other_keys = [k for k in keys if k not in lora_a_keys + lora_b_keys + score_keys]

        print(f"  LoRA A tensors:          {len(lora_a_keys)}")
        print(f"  LoRA B tensors:          {len(lora_b_keys)}")
        print(f"  Score/classifier tensors: {len(score_keys)}")
        print(f"  Other tensors:           {len(other_keys)}")

        # Check LoRA B weights (these should be non-zero after training; LoRA A is init'd to random, B to zero)
        print("\n  LoRA B weight check (should be NON-ZERO after training):")
        all_zero_b = True
        for k in lora_b_keys[:6]:
            tensor = f.get_tensor(k)
            is_zero = tensor.abs().max().item() < 1e-8
            if not is_zero:
                all_zero_b = False
            status = "ALL ZEROS!" if is_zero else "OK (non-zero)"
            print(f"    {k}: shape={list(tensor.shape)}, max_abs={tensor.abs().max().item():.6f} [{status}]")
        if len(lora_b_keys) > 6:
            print(f"    ... and {len(lora_b_keys) - 6} more")

        if all_zero_b:
            print("\n  CRITICAL: All LoRA B weights are zero! Model may not have trained.")

        # Check score layer
        if score_keys:
            print("\n  Classification head (score) weights:")
            for k in score_keys:
                tensor = f.get_tensor(k)
                print(f"    {k}: shape={list(tensor.shape)}, mean={tensor.float().mean():.6f}, std={tensor.float().std():.6f}")
        else:
            print("\n  CRITICAL: No score/classifier weights found! Classification head was NOT saved.")
            print("  This means evaluation uses a RANDOM classification head!")

        if other_keys:
            print(f"\n  Other saved tensors: {other_keys}")


def load_and_compare(checkpoint_dir_best, checkpoint_dir_last, base_model_path, device):
    """Step 4: Load base + best adapter + last adapter, compare predictions."""
    print("\n[Step 4] Loading models and comparing predictions...")
    n_classes = len(LABEL_LIST)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)

    # Load tokenizer from best checkpoint
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir_best, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load base model (no adapter)
    print("  Loading base model (no adapter)...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=n_classes,
        device_map={"": device},
        trust_remote_code=True,
        quantization_config=bnb_config,
    )
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id
    # Resize embeddings to match checkpoint tokenizer (which may have added special tokens)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.eval()

    test_texts = [
        "@ * GeneOrGeneProduct * BRCA1 @ is associated with # ^ DiseaseOrPhenotypicFeature ^ breast cancer #.",
        "Treatment with @ * ChemicalEntity * aspirin @ showed # ^ DiseaseOrPhenotypicFeature ^ headache # reduction.",
        "@ * GeneOrGeneProduct * TP53 @ binds to # ^ GeneOrGeneProduct * MDM2 #.",
    ]

    print("\n  Base model predictions (random classifier head):")
    base_logits_list = []
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length").to(device)
        with torch.no_grad():
            logits = base_model(**inputs).logits.cpu().float()
        pred = torch.argmax(logits, dim=-1).item()
        base_logits_list.append(logits)
        print(f"    '{text[:60]}...' -> {LABEL_LIST[pred]}")
        print(f"      logits: {logits.numpy().round(3)}")

    # Load best adapter
    print("\n  Loading PEFT adapter (checkpoint-best)...")
    model = PeftModel.from_pretrained(base_model, checkpoint_dir_best, adapter_name="best")
    model.eval()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params: {trainable:,} trainable / {total:,} total ({trainable / total * 100:.2f}%)")

    # Load last adapter
    if checkpoint_dir_last:
        print("  Loading PEFT adapter (checkpoint-last)...")
        model.load_adapter(checkpoint_dir_last, adapter_name="last")

    # --- Best model predictions ---
    model.set_adapter("best")
    print("\n  checkpoint-best predictions:")
    best_logits_list = []
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits.cpu().float()
        pred = torch.argmax(logits, dim=-1).item()
        best_logits_list.append(logits)
        print(f"    '{text[:60]}...' -> {LABEL_LIST[pred]}")
        print(f"      logits: {logits.numpy().round(3)}")

    # --- Last model predictions ---
    if checkpoint_dir_last:
        model.set_adapter("last")
        print("\n  checkpoint-last predictions:")
        last_logits_list = []
        for text in test_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length").to(device)
            with torch.no_grad():
                logits = model(**inputs).logits.cpu().float()
            pred = torch.argmax(logits, dim=-1).item()
            last_logits_list.append(logits)
            print(f"    '{text[:60]}...' -> {LABEL_LIST[pred]}")
            print(f"      logits: {logits.numpy().round(3)}")

    # Compare logit differences
    print("\n  Logit differences:")
    for i in range(len(test_texts)):
        bl = base_logits_list[i]
        best_l = best_logits_list[i]
        diff_best = (best_l - bl).abs().mean().item()
        line = f"    Example {i}: base vs best = {diff_best:.4f}"
        if checkpoint_dir_last:
            last_l = last_logits_list[i]
            diff_last = (last_l - bl).abs().mean().item()
            diff_best_last = (best_l - last_l).abs().mean().item()
            line += f", base vs last = {diff_last:.4f}, best vs last = {diff_best_last:.4f}"
        print(line)
        if diff_best < 0.01:
            print(f"      WARNING: Very small base-vs-best difference — adapter may not be effective!")

    return model, tokenizer


def check_prediction_distribution(model, tokenizer, val_data_path, has_last, device, n_samples=200):
    """Step 5: Run predictions on validation data, compare best vs last vs gold."""
    print(f"\n[Step 5] Prediction distribution on validation data ({n_samples} samples)...")

    val_df = pd.read_json(val_data_path)
    n_samples = min(n_samples, len(val_df))
    sample = val_df.sample(n_samples, random_state=42)

    from sklearn.metrics import f1_score, classification_report

    all_labels = []
    all_best_preds = []
    all_last_preds = []
    label_to_id = {v: k for k, v in LABEL_DICT.items()}

    # Collect inputs
    texts = []
    for _, row in sample.iterrows():
        text = transform_sentence_typed_entity_marker_punct(row)
        texts.append(text)
        label = label_to_id.get(row.get("type", "NOT"), 0)
        all_labels.append(label)

    # --- Best model predictions ---
    model.set_adapter("best")
    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512, padding="max_length"
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        all_best_preds.append(torch.argmax(logits, dim=-1).item())

    # --- Last model predictions ---
    if has_last:
        model.set_adapter("last")
        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512, padding="max_length"
            ).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            all_last_preds.append(torch.argmax(logits, dim=-1).item())

    # --- Distribution comparison ---
    label_dist = Counter(all_labels)
    best_dist = Counter(all_best_preds)
    last_dist = Counter(all_last_preds) if has_last else None

    all_class_ids = sorted(set(list(label_dist.keys()) + list(best_dist.keys()) + (list(last_dist.keys()) if last_dist else [])))

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

    # --- Warnings for best ---
    _print_collapse_warning("checkpoint-best", best_dist, n_samples)
    if has_last:
        _print_collapse_warning("checkpoint-last", last_dist, n_samples)

    # --- Agreement ---
    if has_last:
        agree = sum(1 for b, l in zip(all_best_preds, all_last_preds) if b == l)
        print(f"\n  Best vs Last agreement: {agree}/{n_samples} ({agree / n_samples * 100:.1f}%)")

        # Per-sample disagreements
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
    parser = argparse.ArgumentParser(description="Diagnose a PEFT/LoRA checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to checkpoint-best directory")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Base model name/path (HuggingFace or local)")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Path to validation JSON for prediction distribution test")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of validation samples for distribution check")
    args = parser.parse_args()

    checkpoint_dir_best = args.checkpoint_dir

    # Auto-detect checkpoint-last as sibling
    parent_dir = os.path.dirname(checkpoint_dir_best.rstrip("/"))
    checkpoint_dir_last = os.path.join(parent_dir, "checkpoint-last")
    if not os.path.isdir(checkpoint_dir_last):
        checkpoint_dir_last = None

    print("=" * 70)
    print("PEFT/LoRA CHECKPOINT DIAGNOSTIC")
    print(f"Checkpoint (best): {checkpoint_dir_best}")
    print(f"Checkpoint (last): {checkpoint_dir_last or 'NOT FOUND'}")
    print(f"Base model:        {args.base_model}")
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

    # Step 2: Adapter config (best)
    print("\n--- checkpoint-best ---")
    check_adapter_config(checkpoint_dir_best)
    if checkpoint_dir_last:
        print("\n--- checkpoint-last ---")
        check_adapter_config(checkpoint_dir_last)

    # Step 3: Weight inspection (best)
    print("\n--- checkpoint-best ---")
    check_saved_weights(checkpoint_dir_best)
    if checkpoint_dir_last:
        print("\n--- checkpoint-last ---")
        check_saved_weights(checkpoint_dir_last)

    # Step 4: Load and compare
    model, tokenizer = load_and_compare(
        checkpoint_dir_best, checkpoint_dir_last, args.base_model, args.device
    )

    # Step 5: Prediction distribution (if val data provided)
    if args.val_data:
        check_prediction_distribution(
            model, tokenizer, args.val_data,
            has_last=checkpoint_dir_last is not None,
            device=args.device, n_samples=args.n_samples,
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
