import argparse
from pathlib import Path
import json


def main():
    parser = argparse.ArgumentParser(
        description='Generate a prediction config from a training checkpoint directory'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to a checkpoint dir (e.g. ./meta-llama_.../runs_.../split_0/checkpoint-best). '
                             'Parent split dir must contain config.json.')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path(s) to test data file(s), semicolon-separated')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Inference batch size (default: auto — 1 for 20B models, 8 otherwise)')
    parser.add_argument('--float_type', type=str, default=None,
                        choices=['bf16', 'fp16', 'fp32'],
                        help='Override float precision for inference (default: inherit from training config)')
    args = parser.parse_args()

    # --- Read training config from parent (split) dir ---
    checkpoint_path = Path(args.checkpoint)
    split_dir = checkpoint_path.parent
    train_config_path = split_dir / "config.json"
    if not train_config_path.is_file():
        raise FileNotFoundError(
            f"Training config not found at {train_config_path}. "
            f"Ensure --checkpoint points to a checkpoint dir whose parent split dir contains config.json."
        )

    with open(train_config_path) as f:
        train_config = json.load(f)

    # --- Inherit fields from training config ---
    inherited_keys = [
        "model_name_or_path", "tokenizer_path", "transform_method",
        "move_entities_to_start", "label_column_name", "max_len",
        "use_prompt_format", "random_seed", "dataloader_num_workers",
    ]
    config_dict = {}
    for key in inherited_keys:
        if key in train_config:
            config_dict[key] = train_config[key]

    # --- Infer use_qlora from training config ---
    # deepspeed key present → full fine-tune (not qlora); absent → qlora
    config_dict["use_qlora"] = "deepspeed" not in train_config

    # --- Resolve batch size: CLI override or inherit from training config ---
    if args.batch_size is not None:
        bs = args.batch_size
    else:
        bs = train_config["per_device_eval_batch_size"]
    config_dict["per_device_eval_batch_size"] = bs

    # --- Float type: override or inherit ---
    if args.float_type is not None:
        float_type = args.float_type
    else:
        # Reconstruct from training config's bf16/fp16 flags
        if train_config.get("bf16", False):
            float_type = "bf16"
        elif train_config.get("fp16", False):
            float_type = "fp16"
        else:
            float_type = "fp32"

    if float_type == "bf16":
        config_dict["bf16"] = True
        config_dict["fp16"] = False
    elif float_type == "fp16":
        config_dict["bf16"] = False
        config_dict["fp16"] = True
    else:  # fp32
        config_dict["bf16"] = False
        config_dict["fp16"] = False

    # --- Test data and naming ---
    config_dict["testing_dataframes"] = args.test_data
    test_data_parts = [Path(p.strip()).stem for p in args.test_data.split(";")]
    test_data_name = "+".join(test_data_parts)

    # --- Checkpoint dir ---
    config_dict["checkpoint_dir"] = str(checkpoint_path)

    # --- Build mirrored output dir: runs* -> pred_runs* ---
    parts = checkpoint_path.parts
    runs_idx = None
    for i, part in enumerate(parts):
        if part.startswith("runs"):
            runs_idx = i
            break

    if runs_idx is None:
        raise ValueError(
            f"Cannot find a 'runs*' component in checkpoint path '{checkpoint_path}'. "
            f"Expected a path like ./model_dir/runs_.../split_N/checkpoint-best"
        )

    pred_parts = list(parts)
    pred_parts[runs_idx] = "pred_" + pred_parts[runs_idx]
    pred_path = Path(*pred_parts) / test_data_name

    config_dict["output_dir"] = str(pred_path)

    # --- Write config ---
    pred_path.mkdir(parents=True, exist_ok=True)
    with open(pred_path / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=4)

    print(f"Prediction config written to: {pred_path / 'config.json'}")
    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  Test data:   {args.test_data}")
    print(f"  Batch size:  {bs}")
    print(f"  Float type:  {float_type}")
    print(f"  use_qlora:   {config_dict['use_qlora']}")


if __name__ == "__main__":
    main()
