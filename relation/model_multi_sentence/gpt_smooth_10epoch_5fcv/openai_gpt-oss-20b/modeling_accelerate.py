#!/usr/bin/env python
# coding: utf-8
"""
Pure Accelerate training script for gpt-oss-20b with MXFP4 + LoRA support.
Works with both DDP and FSDP - the strategy is determined by launch flags.

This script implements a two-stage quantization pipeline for MXFP4 models:
  MXFP4 → bf16 → NF4 → LoRA

The key insight is that MXFP4 models cannot be directly fine-tuned with LoRA.
We must:
1. Dequantize MXFP4 → bf16 (using Mxfp4Config(dequantize=True))
2. Save dequantized model to temp directory
3. Reload with BitsAndBytes NF4 quantization
4. Apply LoRA adapters

For DeepSpeed, use modeling_accelerate_deepspeed.py instead.

Usage:
    FSDP: ./run_accelerate_fsdp.sh 4 config.json [gpu_ids]
    DDP:  ./run_accelerate_ddp.sh 4 config.json [gpu_ids]
"""

from concurrent.futures import process
from random import randrange

import pandas as pd
import json
import torch
from torch import nn
import transformers
from transformers import AutoTokenizer
from transformers import TrainerCallback
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from collections import defaultdict
import numpy as np

import logging
import tempfile
import shutil

from torch.utils.data import Dataset, DataLoader

import sys
import os
import argparse
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser, set_seed
from transformers import TrainingArguments
from transformers import Mxfp4Config
from transformers import BitsAndBytesConfig
from accelerate import PartialState
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)


def is_mxfp4_model(model_name_or_path):
    """Check if model uses MXFP4 quantization (e.g., gpt-oss-20b)."""
    mxfp4_models = ["openai/gpt-oss-20b", "gpt-oss-20b"]
    return any(m in model_name_or_path.lower() for m in mxfp4_models)

label_list = ["NOT", "Association", "Positive_Correlation", "Negative_Correlation", "Bind", "Cotreatment", "Comparison", "Drug_Interaction", "Conversion"]
label_dict = {idx: val for idx, val in enumerate(label_list)}

class SaveBestModelCallback(TrainerCallback):
    """
    Custom callback that saves both the best model and the last model.
    - checkpoint-best/: Contains the model with the best evaluation metric
    - checkpoint-last/: Contains the most recent model (updated every evaluation)
    """
    def __init__(self, enable_saving=True, distributed_state=None):
        self.enable_saving = enable_saving
        self.best_metric = None
        self.best_checkpoint_path = None
        self.distributed_state = distributed_state

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Called after each evaluation.
        - Always saves to checkpoint-last/
        - Saves to checkpoint-best/ only if current metric is better than previous best
        """
        # Early return if saving is disabled (e.g., during prediction/evaluation only)
        if not self.enable_saving:
            return control

        if metrics is None:
            return control

        # Get the metric we're tracking (e.g., "eval_f1")
        metric_key = f"eval_{args.metric_for_best_model}"
        if metric_key not in metrics:
            logger.warning(f"Metric {metric_key} not found in evaluation metrics. Available: {metrics.keys()}")
            return

        current_metric = metrics[metric_key]

        # Always save the last checkpoint (only from main process in distributed training)
        last_checkpoint_path = os.path.join(args.output_dir, "checkpoint-last")

        # Check if we should save (main process only in distributed training)
        should_save = True
        if self.distributed_state is not None and not self.distributed_state.is_main_process:
            should_save = False

        if should_save:
            logger.info(f"Saving last checkpoint to {last_checkpoint_path} (step {state.global_step}, {args.metric_for_best_model}={current_metric:.4f})")
            kwargs['model'].save_pretrained(last_checkpoint_path)
            kwargs['processing_class'].save_pretrained(last_checkpoint_path)

            # Save trainer state for last checkpoint
            last_state_dict = {
                'global_step': state.global_step,
                'epoch': state.epoch,
                'metric_key': metric_key,
                'metric': current_metric,
            }
            torch.save(last_state_dict, os.path.join(last_checkpoint_path, "trainer_state.pt"))

        # Determine if current metric is better than best
        is_better = False
        if self.best_metric is None:
            is_better = True
        else:
            # Check if we want to maximize or minimize the metric
            greater_is_better = args.greater_is_better if hasattr(args, 'greater_is_better') else True
            if greater_is_better:
                is_better = current_metric > self.best_metric
            else:
                is_better = current_metric < self.best_metric

        # Save best checkpoint if improved (only from main process in distributed training)
        if is_better:
            self.best_metric = current_metric
            best_checkpoint_path = os.path.join(args.output_dir, "checkpoint-best")

            if should_save:
                logger.info(f"New best {args.metric_for_best_model}: {current_metric:.4f}. Saving to {best_checkpoint_path}")
                kwargs['model'].save_pretrained(best_checkpoint_path)
                kwargs['processing_class'].save_pretrained(best_checkpoint_path)

                # Save trainer state for best checkpoint
                best_state_dict = {
                    'best_metric': self.best_metric,
                    'best_model_checkpoint': best_checkpoint_path,
                    'global_step': state.global_step,
                    'epoch': state.epoch,
                }
                torch.save(best_state_dict, os.path.join(best_checkpoint_path, "trainer_state.pt"))

            self.best_checkpoint_path = best_checkpoint_path

        return control

@dataclass
class NonTrainingArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: str = field(
        metadata={"help": "cls or triplet. Use only the cls token or use the whole triplet"}
    )
    max_len: int = field(
        metadata={"help": "Tokenization maximum length."}
    )
    transform_method: str = field(
        metadata={'help': 'how sentence is processed. One value of ["entity_mask", "entity_marker", "entity_marker_punkt", "typed_entity_marker", "typed_entity_marker_punct"]'}
    )
    label_column_name: str = field(
        metadata={"help": "The column name in the dataframe that contains the labels"}
    )
    move_entities_to_start: bool = field(
        metadata={"help": "Whether or not we add entities to the beginning of a sentence."}
    )
    tokenizer_path: str = field(
        metadata={"help": "path to tokenizer"}
    )
    random_seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"}
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "Whether to use QLoRA (4-bit quantization + LoRA) for memory-efficient training"}
    )

@dataclass
class DataArguments:
    training_dataframes: str = field(
        metadata={"help": "Where the training datasets are located, separate multiple by ;"}
    )
    validation_dataframes: str = field(
        metadata={"help": "Where the training datasets are located, separate multiple by ;"}
    )
    testing_dataframes: str = field(
        metadata={"help": "Where the training datasets are located, separate multiple by ;"}
    )

class SentenceDataset(Dataset):

    def __init__(self, sentences, labels, tokenizer, max_len):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        label = self.labels[item]

        # Use __call__ instead of encode_plus (transformers v5+ compatible)
        encoding = self.tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'sentence': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def transform_sentence(entry, transform_method: str, move_entities_to_start: bool):
    entity_a = entry["entity_a"]
    entity_b = entry["entity_b"]
    sent = entry["text"]
    if sent == "": return ""

    # Tag positions with entity identity ('a' or 'b')
    all_poses = [(start, end, e_type, 'a') for start, end, e_type in entity_a] + \
                [(start, end, e_type, 'b') for start, end, e_type in entity_b]
    all_poses.sort(key=lambda i: i[0], reverse=True)

    # Process from right to left to preserve position validity
    for start, end, e_type, entity_id in all_poses:
        mention = entry["text"][start:end]

        # Determine pre and post markers based on transform method
        if transform_method == "entity_mask":
            pre, post = "", ""
            ent = e_type
        elif transform_method == "entity_marker":
            if entity_id == 'a':
                pre, post = "[E1] ", " [/E1]"
            else:
                pre, post = "[E2] ", " [/E2]"
            ent = mention
        elif transform_method == "entity_marker_punkt":
            if entity_id == 'a':
                pre, post = "@ ", " @"
            else:
                pre, post = "# ", " #"
            ent = mention
        elif transform_method == "typed_entity_marker":
            pre, post = f"[{e_type}] ", f" [/{e_type}]"
            ent = mention
        elif transform_method == "typed_entity_marker_punct":
            if entity_id == 'a':
                pre, post = f"@ * {e_type} * ", " @"
            else:
                pre, post = f"# ^ {e_type} ^ ", " #"
            ent = mention
        else:
            raise NotImplementedError(f"{transform_method} is not implemented!")

        # Apply replacement
        replacement = pre + ent + post
        sent = sent[0:start] + replacement + sent[end:]

    # Prepend entity mentions
    if move_entities_to_start:
        sent = entry["text"][entity_a[0][0]:entity_a[0][1]] + ", " + entry["text"][entity_b[0][0]:entity_b[0][1]] + ", " + sent

    return sent

def process_data(df_train, transform_method, move_entities_to_start, label_column_name="type", mode='train'):
    if mode == 'train':
        df_train["label"] = df_train[label_column_name]
        for key, label in label_dict.items():
            df_train['label'] = df_train['label'].replace(label, key)
    elif mode == 'test':
        df_train["label"] = 0
    df_train["text"] = [transform_sentence(entry, transform_method, move_entities_to_start) for _, entry in df_train.iterrows()]
    return df_train

def get_dataframes(cmd_str):
    str_list = cmd_str.split(";")
    dataframes = [pd.read_json(path) for path in str_list]
    return pd.concat(dataframes, ignore_index=True)


def dequantize_mxfp4_to_temp(model_name_or_path, n_classes, model_dtype, tokenizer, distributed_state=None):
    """
    Dequantize an MXFP4 model to bf16 and save to a temporary directory.

    This is necessary because MXFP4 models cannot be directly quantized with
    BitsAndBytes. We first dequantize to bf16, save, then reload with NF4.

    Note: MXFP4 dequantized models cannot use save_pretrained() directly because
    the reverse weight conversion is not implemented. We must save the state_dict
    directly using safetensors.

    In distributed training, only the main process does the dequantization to avoid
    duplicate work and disk usage.

    Args:
        model_name_or_path: HuggingFace model name or path
        n_classes: Number of output classes
        model_dtype: Target dtype (torch.bfloat16)
        tokenizer: Tokenizer (needed to resize embeddings)
        distributed_state: PartialState for distributed coordination

    Returns:
        tuple: (temp_model_dir, config) - Path to temp directory and model config
    """
    from transformers import AutoModelForSequenceClassification

    # Create a deterministic temp directory name so all processes use the same path
    # Use a fixed name in the current directory
    temp_model_dir = os.path.join(".", "mxfp4_dequant_temp")

    # Only main process does the dequantization
    is_main = distributed_state is None or distributed_state.is_main_process

    if is_main:
        logger.info(f"Stage 1: Dequantizing MXFP4 model to {model_dtype}")

        dequant_config = Mxfp4Config(dequantize=True)

        temp_model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=n_classes,
            torch_dtype=model_dtype,
            quantization_config=dequant_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",  # Avoid flash attention issues with dtype conversion
        )

        # Set pad_token_id in model config for GPT models
        if temp_model.config.pad_token_id is None:
            temp_model.config.pad_token_id = tokenizer.pad_token_id

        # Resize token embeddings to accommodate new special tokens
        temp_model.resize_token_embeddings(len(tokenizer))

        # Save to local temp directory (in current working directory)
        logger.info(f"Stage 2: Saving dequantized model to {temp_model_dir}")

        os.makedirs(temp_model_dir, exist_ok=True)

        # Get state dict - must save directly because MXFP4 doesn't support reverse weight conversion
        state_dict = temp_model.state_dict()

        # Save state dict using safetensors (bypasses weight conversion reversal)
        try:
            from safetensors.torch import save_file
            # Filter out non-tensor items and ensure contiguous tensors
            tensor_state_dict = {k: v.contiguous() for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
            save_file(tensor_state_dict, os.path.join(temp_model_dir, "model.safetensors"))
            logger.info(f"Saved model weights using safetensors")
        except ImportError:
            # Fallback to torch.save if safetensors not available
            torch.save(state_dict, os.path.join(temp_model_dir, "pytorch_model.bin"))
            logger.info(f"Saved model weights using torch.save (safetensors not available)")

        # Save config (without quantization info so it loads as regular model)
        config = temp_model.config
        config_dict = config.to_dict()
        # Remove quantization-specific fields that would interfere with NF4 loading
        config_dict.pop('quantization_config', None)
        config_dict.pop('_pre_quantization_dtype', None)
        # Ensure model architecture info is preserved for BitsAndBytes to work
        if 'architectures' not in config_dict and hasattr(config, 'architectures'):
            config_dict['architectures'] = config.architectures
        if 'model_type' not in config_dict and hasattr(config, 'model_type'):
            config_dict['model_type'] = config.model_type
        # Save config manually
        import json
        with open(os.path.join(temp_model_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Saved config with model_type={config_dict.get('model_type')} architectures={config_dict.get('architectures')}")

        # Save tokenizer
        tokenizer.save_pretrained(temp_model_dir)

        # Free memory
        del temp_model
        del state_dict
        torch.cuda.empty_cache()
    else:
        logger.info(f"Rank {distributed_state.process_index}: Waiting for main process to dequantize model...")

    # Wait for all processes to sync before returning
    if distributed_state is not None:
        distributed_state.wait_for_everyone()

    return temp_model_dir


def main():
    # Parse arguments using HfArgumentParser
    # Include TrainingArguments from transformers so we don't need to initialize it separately
    parser = HfArgumentParser((NonTrainingArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        non_train_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        non_train_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Print arguments
    print("=" * 50)
    print("Non-Training Arguments:")
    for arg in vars(non_train_args):
        print(f"  {arg}: {getattr(non_train_args, arg)}")
    print("\nData Arguments:")
    for arg in vars(data_args):
        print(f"  {arg}: {getattr(data_args, arg)}")
    print("\nTraining Arguments (key parameters):")
    print(f"  output_dir: {training_args.output_dir}")
    print(f"  num_train_epochs: {training_args.num_train_epochs}")
    print(f"  learning_rate: {training_args.learning_rate}")
    print(f"  per_device_train_batch_size: {training_args.per_device_train_batch_size}")
    print(f"  per_device_eval_batch_size: {training_args.per_device_eval_batch_size}")
    print(f"  fp16: {training_args.fp16}")
    print("=" * 50)

    # Set random seed
    RANDOM_SEED = non_train_args.random_seed
    set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Use Accelerate's device detection for distributed compatibility
    distributed_state = PartialState()
    device = distributed_state.device
    transformers.logging.set_verbosity_error()
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    train_dataframe = get_dataframes(data_args.training_dataframes)
    validation_dataframe = get_dataframes(data_args.validation_dataframes)
    test_dataframe = get_dataframes(data_args.testing_dataframes)

    df_train = process_data(
        train_dataframe,
        non_train_args.transform_method,
        non_train_args.move_entities_to_start,
        non_train_args.label_column_name)
    df_val = process_data(
        validation_dataframe,
        non_train_args.transform_method,
        non_train_args.move_entities_to_start,
        non_train_args.label_column_name)
    df_test = process_data(
        test_dataframe,
        non_train_args.transform_method,
        non_train_args.move_entities_to_start,
        non_train_args.label_column_name,
        mode='test')

    # Model Parameters
    PRE_TRAINED_MODEL_NAME = non_train_args.model_name_or_path
    MAX_LEN = 512
    BATCH_SIZE = training_args.per_device_train_batch_size

    # https://huggingface.co/blog/zero-deepspeed-fairscale
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    # GPT models need pad token set (they don't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokens = label_list[1:]
    tokenizer.add_tokens(tokens, special_tokens=True)

    train_data = SentenceDataset(
        sentences=df_train.text.to_numpy(),
        labels=df_train.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    val_data = SentenceDataset(
        sentences=df_val.text.to_numpy(),
        labels=df_val.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    test_data = SentenceDataset(
        sentences=df_test.text.to_numpy(),
        labels=df_test.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )

    from transformers import AutoModelForSequenceClassification
    n_classes = len(label_dict)

    # Determine dtype for model loading
    # FSDP requires uniform dtype - must explicitly set dtype
    # Models on HuggingFace may be stored in bf16, so we must force fp32 if not using mixed precision
    if training_args.bf16:
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32  # Explicitly use fp32 for FSDP compatibility

    # Track temp directory for cleanup
    temp_model_dir = None

    # Model loading priority:
    # 1. MXFP4 + QLoRA - Two-stage pipeline: MXFP4 → bf16 → NF4 → LoRA
    # 2. MXFP4 only - Dequantize to bf16 for full fine-tuning
    # 3. Non-MXFP4 + QLoRA - Direct NF4 → LoRA (same as Llama)
    # 4. Non-MXFP4 only - Regular loading
    if is_mxfp4_model(PRE_TRAINED_MODEL_NAME) and non_train_args.use_qlora:
        # MXFP4 + QLoRA: Two-stage quantization pipeline
        # Stage 1-2: Dequantize MXFP4 to bf16 and save to temp
        logger.info("=" * 50)
        logger.info("MXFP4 + QLoRA Pipeline: MXFP4 → bf16 → NF4 → LoRA")
        logger.info("=" * 50)

        temp_model_dir = dequantize_mxfp4_to_temp(
            PRE_TRAINED_MODEL_NAME, n_classes, model_dtype, tokenizer, distributed_state
        )

        # Stage 3: Reload with BitsAndBytes NF4 quantization
        logger.info(f"Stage 3: Loading dequantized model with NF4 quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,  # Further reduce memory
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            temp_model_dir,
            device_map={"": device},  # Load to this process's GPU (DDP-compatible)
            trust_remote_code=True,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
        )

        # Log memory usage to verify quantization is working
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
            logger.info(f"GPU memory after NF4 load: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
            if mem_allocated > 20:
                logger.warning(f"Memory usage ({mem_allocated:.2f}GB) is higher than expected for 4-bit model. Quantization may not be working.")

        # Stage 4: Apply LoRA
        logger.info(f"Stage 4: Applying LoRA adapters")
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules="all-linear",
            modules_to_save=["score"],  # Classification head fully trained
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    elif is_mxfp4_model(PRE_TRAINED_MODEL_NAME):
        # MXFP4 only: Dequantize to bf16 for full fine-tuning
        logger.info(f"Detected MXFP4 model: {PRE_TRAINED_MODEL_NAME}")
        logger.info(f"Dequantizing from MXFP4 to {model_dtype} for fine-tuning")

        # Must use Mxfp4Config(dequantize=True) to convert MXFP4 to trainable format
        dequant_config = Mxfp4Config(dequantize=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            PRE_TRAINED_MODEL_NAME,
            num_labels=n_classes,
            torch_dtype=model_dtype,
            quantization_config=dequant_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",  # Avoid flash attention issues with dtype conversion
        )

    elif non_train_args.use_qlora:
        # Non-MXFP4 + QLoRA: Direct NF4 → LoRA (same as Llama)
        # Note: device_map="auto" is incompatible with DDP - each process loads to its own GPU
        logger.info(f"Using QLoRA (4-bit quantization + LoRA) for memory-efficient training")
        logger.info(f"Loading quantized model to device: {device}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            PRE_TRAINED_MODEL_NAME,
            num_labels=n_classes,
            device_map={"": device},  # Load to this process's GPU (DDP-compatible)
            trust_remote_code=True,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
        )
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            target_modules="all-linear",
            modules_to_save=["score"],  # Classification head fully trained
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    else:
        # Regular full-precision loading
        model = AutoModelForSequenceClassification.from_pretrained(
            PRE_TRAINED_MODEL_NAME,
            num_labels=n_classes,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
        )

    # Set pad_token_id in model config for GPT models
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Resize token embeddings to accommodate new special tokens
    # Skip for MXFP4+QLoRA as it was done during dequantization
    if not (is_mxfp4_model(PRE_TRAINED_MODEL_NAME) and non_train_args.use_qlora):
        if model.base_model_prefix == "roberta":
            model.roberta.resize_token_embeddings(len(tokenizer))
        elif model.base_model_prefix == "bert":
            model.bert.resize_token_embeddings(len(tokenizer))
        elif model.base_model_prefix in ["transformer", "gpt2", "gpt_neox", "model"]:
            # GPT-style models
            model.resize_token_embeddings(len(tokenizer))
        else:
            # Fallback: try generic resize
            model.resize_token_embeddings(len(tokenizer))

    # FSDP requires uniform dtype across all parameters
    # Cast entire model to ensure newly initialized layers (e.g., classification head) match
    # Skip for QLoRA as quantization handles dtype
    if not non_train_args.use_qlora and model_dtype is not None:
        model = model.to(model_dtype)
    # Note: Trainer handles device placement when using Accelerate for distributed training

    from transformers import Trainer

    # https://stackoverflow.com/questions/69087044/early-stopping-in-bert-trainer-instances
    from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

    def compute_metrics(evalprediction_instance):
        predictions, label_ids = evalprediction_instance.predictions, evalprediction_instance.label_ids
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(label_ids, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(label_ids, predictions,  labels=range(1, len(label_dict)), average='micro', zero_division=0)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    # Initialize custom callback for saving only the best model
    # Pass distributed_state for multi-GPU awareness (only main process saves checkpoints)
    save_best_callback = SaveBestModelCallback(enable_saving=True, distributed_state=distributed_state)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
        callbacks=[save_best_callback],
        processing_class=tokenizer  # renamed from 'tokenizer' in transformers v5+
    )

    if training_args.do_train:
            train_result = trainer.train()

            metrics = train_result.metrics
            metrics["train_samples"] = len(train_data)
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)

    # Disable checkpoint saving after training completes
    save_best_callback.enable_saving = False

    # Load best model if available
    # Wait for all processes to ensure checkpoint is saved before loading
    distributed_state.wait_for_everyone()

    if save_best_callback.best_checkpoint_path is not None:
        logger.info(f"Loading best model from {save_best_callback.best_checkpoint_path}")
        n_classes = len(label_dict)

        if non_train_args.use_qlora:
            # QLoRA: Load base model + LoRA adapters
            logger.info("Loading QLoRA model: base model + LoRA adapters")

            if is_mxfp4_model(PRE_TRAINED_MODEL_NAME):
                # For MXFP4 + QLoRA, we need to reload from temp dir or recreate it
                if temp_model_dir is None or not os.path.exists(temp_model_dir):
                    logger.info("Recreating dequantized base model for loading LoRA adapters")
                    temp_model_dir = dequantize_mxfp4_to_temp(
                        PRE_TRAINED_MODEL_NAME, n_classes, model_dtype, tokenizer, distributed_state
                    )
                base_model_path = temp_model_dir
            else:
                base_model_path = PRE_TRAINED_MODEL_NAME

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            base_model = AutoModelForSequenceClassification.from_pretrained(
                base_model_path,
                num_labels=n_classes,
                device_map={"": device},  # Load to this process's GPU (DDP-compatible)
                trust_remote_code=True,
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
            )
            model = PeftModel.from_pretrained(base_model, save_best_callback.best_checkpoint_path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                save_best_callback.best_checkpoint_path,
                num_labels=n_classes,
                torch_dtype=model_dtype,  # Use same dtype as initial model
            )

        tokenizer = AutoTokenizer.from_pretrained(save_best_callback.best_checkpoint_path)
        # Ensure pad_token_id is set in model config
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        # Ensure uniform dtype and move to device (skip for QLoRA as device_map handles it)
        if not non_train_args.use_qlora:
            if model_dtype is not None:
                model = model.to(model_dtype)
            model = model.to(device)
        trainer.model = model
        trainer.tokenizer = tokenizer
    else:
        logger.info("No best checkpoint found, using current model (last checkpoint)")

    # Evaluate on test set using best model
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = len(val_data)
        metrics["eval_samples"] = min(max_eval_samples, len(val_data))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        # Generate predictions on validation dataset
        logger.info("*** Predict on validation dataset ***")
        ret = trainer.predict(val_data)

        predictions = ret.predictions
        output_dir = training_args.output_dir
        with open(os.path.join(output_dir, 'predictions_prob_eval.txt'), 'w') as f:
            f.writelines('\n'.join(['\t'.join(list(map(str, x))) for x in predictions])+'\n')
        predicted_class = np.argmax(predictions, axis=1)
        predicted_class = [label_dict[cl] for cl in predicted_class]
        expanded_predicted_class = []
        for idx in range(len(predicted_class)):
            expanded_predicted_class.append(predicted_class[idx])
        df_val["predicted_class"] = expanded_predicted_class
        df_val.to_csv(os.path.join(output_dir, "eval_predictions.csv"))

    if training_args.do_predict:
        ret = trainer.predict(test_data)

        predictions = ret.predictions
        output_dir = training_args.output_dir
        with open(os.path.join(output_dir, 'predictions_prob.txt'), 'w') as f:
            f.writelines('\n'.join(['\t'.join(list(map(str, x))) for x in predictions])+'\n')
        predicted_class = np.argmax(predictions, axis=1)
        predicted_class = [label_dict[cl] for cl in predicted_class]
        expanded_predicted_class = []
        for idx in range(len(predicted_class)):
            expanded_predicted_class.append(predicted_class[idx])
        df_test["predicted_class"] = expanded_predicted_class
        df_test.to_csv(os.path.join(output_dir, "predictions.csv"))

    # Cleanup temp directory
    if temp_model_dir is not None and os.path.exists(temp_model_dir):
        logger.info(f"Cleaning up temp directory: {temp_model_dir}")
        shutil.rmtree(temp_model_dir)


if __name__ == "__main__":
    main()
