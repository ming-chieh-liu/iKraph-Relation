import argparse
from pathlib import Path
import json

def main():
    parser = argparse.ArgumentParser(description='Generate training configurations')
    parser.add_argument('--posfix', type=str, help='Postfix for configuration files')
    parser.add_argument('--transform_method', type=str, default='typed_entity_marker_punct', help='Data transformation method')
    parser.add_argument('--float_type', type=str, default='fp32',
                        choices=['bf16', 'fp16', 'fp32'],
                        help='Floating point precision for training')
    parser.add_argument('--model', type=str, default='openai/gpt-oss-20b',
                        help='HuggingFace model path (e.g., "openai/gpt-oss-20b", "meta-llama/Llama-3.1-8B-Instruct", "roberta-large")')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Per-device batch size (default: 16)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='Gradient accumulation steps (default: 8)')
    parser.add_argument('--training_mode', type=str, required=True,
                        choices=['qlora', 'deepspeed_mxfp4', 'deepspeed_bf16'],
                        help='Training mode: qlora (4-bit QLoRA, default LR 2e-4), deepspeed_mxfp4 (MXFP4 full fine-tune, default LR 3e-5), or deepspeed_bf16 (bf16 full fine-tune, default LR 3e-5)')
    parser.add_argument('--use_prompt_format', action='store_true', default=False,
                        help='Wrap transformed text in an instruction prompt for relation extraction')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (default: 2e-4 for qlora, 3e-5 for deepspeed modes)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    posfix = args.posfix
    transform_method = args.transform_method
    float_type = args.float_type
    mode = args.training_mode

    if mode in ('deepspeed_mxfp4', 'deepspeed_bf16') and float_type != 'bf16':
        import warnings
        warnings.warn(
            f"DeepSpeed mode '{mode}' is recommended to use --float_type bf16. "
            f"Currently set to '{float_type}'."
        )

    # Full HuggingFace model path for loading
    model_path = args.model
    # Model directory: "openai/gpt-oss-20b" -> "openai_gpt-oss-20b"
    model_dir = args.model.replace('/', '_').lower()

    # Determine batch size early so we can include it in directory names
    bs = args.batch_size

    # Mode-aware default learning rate, overridable via --learning_rate
    if args.learning_rate is not None:
        lr = args.learning_rate
    elif mode == 'qlora':
        lr = 2e-4
    else:  # deepspeed_mxfp4 or deepspeed_bf16
        lr = 3e-5

    # Training data directory (relative to model_dir for config paths)
    training_dir_for_config = f'./data/multi_sentence_split{posfix}'

    sf = 0.02

    # Unified run directory: configs and checkpoints live together per fold
    prompt_tag = "_prompt" if args.use_prompt_format else ""
    run_parent = Path(f'./{model_dir}/runs{posfix}_{transform_method}_{mode}{prompt_tag}_bs{bs}_lr{lr}_ls{sf}')

    for fold in range(5):
        config_dict = {}
        config_dict["model_name_or_path"] = model_path
        config_dict["tokenizer_path"] = model_path
        config_dict["max_len"] = 512
        config_dict["transform_method"] = transform_method
        config_dict["label_column_name"] = "type"
        if transform_method == "entity_mask":
            config_dict["move_entities_to_start"] = True
        else:
            config_dict["move_entities_to_start"] = False

        # Build training and validation splits based on fold
        # Fold 0: training 0-3, validation 4
        # Fold 1: training 1-4, validation 0
        # Fold 2: training 2-4,0, validation 1, etc.
        validation_split = (fold + 4) % 5
        training_splits = [i for i in range(5) if i != validation_split]

        # Data paths are relative to model_dir (training runs from there)
        training_paths = ";".join([f"{training_dir_for_config}/split_{i}.json" for i in training_splits])
        validation_path = f"{training_dir_for_config}/split_{validation_split}.json"

        config_dict["training_dataframes"] = training_paths
        config_dict["validation_dataframes"] = validation_path
        config_dict["dataloader_num_workers"] = 0
        config_dict["do_train"] = True
        config_dict["do_eval"] = False
        config_dict["do_predict"] = False
        config_dict["num_train_epochs"] = 10
        config_dict["eval_strategy"] = "steps"
        config_dict["learning_rate"] = lr
        config_dict["eval_steps"] = 200
        config_dict["save_strategy"] = "no"
        config_dict["metric_for_best_model"] = "f1"
        config_dict["warmup_steps"] = 100

        # Batch size and gradient accumulation
        grad_accum = args.gradient_accumulation_steps
        config_dict["per_device_train_batch_size"] = bs
        config_dict["per_device_eval_batch_size"] = bs
        config_dict["gradient_accumulation_steps"] = grad_accum

        # DeepSpeed: select config based on training mode
        if mode in ('deepspeed_mxfp4', 'deepspeed_bf16'):
            config_dict["deepspeed"] = f"./{mode}_config_zero3.json"

        # Float type configuration
        if float_type == "bf16":
            config_dict["bf16"] = True
            config_dict["fp16"] = False
        elif float_type == "fp16":
            config_dict["bf16"] = False
            config_dict["fp16"] = True
        else:  # fp32
            config_dict["bf16"] = False
            config_dict["fp16"] = False

        config_dict["gradient_checkpointing"] = True
        config_dict["load_best_model_at_end"] = False
        config_dict["label_smoothing_factor"] = sf
        config_dict["use_prompt_format"] = args.use_prompt_format
        config_dict["random_seed"] = args.random_seed

        # Per-fold directory: config.json alongside checkpoints
        fold_dir = run_parent / f'split_{fold}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        config_dict["output_dir"] = str(fold_dir)

        config_path = fold_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

if __name__ == "__main__":
    main()
