#!/usr/bin/env python
# coding: utf-8
"""
DeepSpeed full fine-tuning script for MXFP4-quantized models (e.g., gpt-oss-20b).

Loads an MXFP4 model, dequantizes to bf16/fp16, and fully fine-tunes with DeepSpeed.
No QLoRA, no multi-format branching — MXFP4 only.

Features:
  - `subject_object_marker` transform method
  - Optional instruction prompt format (`use_prompt_format`)
  - DeepSpeed ZeRO-3 aware saving via `SaveBestModelCallback`

For evaluation/prediction, use predict.py instead.

Usage:
    ./run_accelerate_deepspeed.sh 4 config.json [gpu_ids]
"""

import gc

import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer
from transformers import TrainerCallback
import numpy as np

import logging

from torch.utils.data import Dataset

import sys
import os
from dataclasses import dataclass, field
from transformers import HfArgumentParser, set_seed
from transformers import TrainingArguments
from transformers.integrations import HfDeepSpeedConfig
from transformers import Mxfp4Config
from accelerate import PartialState

logger = logging.getLogger(__name__)


def is_mxfp4_model(model_name_or_path):
    """Check if model uses MXFP4 quantization (e.g., gpt-oss-20b)."""
    mxfp4_models = ["openai/gpt-oss-20b", "gpt-oss-20b"]
    return any(m in model_name_or_path.lower() for m in mxfp4_models)

RELATION_DEFS = {
    "Association": "A general relationship between two entities that does not fit a more specific category.",
    "Positive_Correlation": "One entity increases or promotes the other, or they are positively correlated.",
    "Negative_Correlation": "One entity decreases or inhibits the other, or they are negatively correlated.",
    "Bind": "A physical binding interaction between two entities.",
    "Cotreatment": "Two entities are administered or applied together as part of a treatment.",
    "Comparison": "Two entities are compared with each other in the study.",
    "Drug_Interaction": "A pharmacological interaction between two drug entities.",
    "Conversion": "One entity is converted or transformed into the other.",
    "NOT": "No meaningful relation exists between the two entities.",
}

label_list = ["NOT", "Association", "Positive_Correlation", "Negative_Correlation", "Bind", "Cotreatment", "Comparison", "Drug_Interaction", "Conversion"]
label_dict = {idx: val for idx, val in enumerate(label_list)}

class SaveBestModelCallback(TrainerCallback):
    """
    Custom callback that saves checkpoint-best/ and checkpoint-last/.
    - checkpoint-best/: Saved on_evaluate only when the tracked metric improves
    - checkpoint-last/: Saved once at the end of training via on_train_end
    """
    def __init__(self, enable_saving=True, distributed_state=None):
        self.enable_saving = enable_saving
        self.best_metric = None
        self.best_checkpoint_path = None
        self.distributed_state = distributed_state
        self.trainer = None

    def set_trainer(self, trainer):
        """Set trainer reference for DeepSpeed-aware saving."""
        self.trainer = trainer

    def _gather_state_dict(self):
        """Gather ZeRO-3 sharded weights into a full state_dict (all processes must call this)."""
        trainer = self.trainer
        if trainer.is_deepspeed_enabled:
            return trainer.accelerator.get_state_dict(trainer.deepspeed)
        return trainer.model.state_dict()

    def _write_checkpoint(self, state_dict, output_dir):
        """Write state_dict to disk (main process only). Does NOT free state_dict."""
        trainer = self.trainer
        if self.distributed_state is not None and not self.distributed_state.is_main_process:
            return

        os.makedirs(output_dir, exist_ok=True)

        try:
            from safetensors.torch import save_file
            tensor_state_dict = {k: v.contiguous() for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
            save_file(tensor_state_dict, os.path.join(output_dir, "model.safetensors"))
            del tensor_state_dict
        except ImportError:
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

        config = trainer.model.config
        if hasattr(config, 'quantization_config'):
            config = config.__class__.from_dict(config.to_dict())
            config.quantization_config = None
        config.save_pretrained(output_dir)

        if trainer.processing_class is not None:
            trainer.processing_class.save_pretrained(output_dir)

    def _save_model_deepspeed(self, output_dir):
        """Gather + write + cleanup. Used by on_evaluate for single-path saves."""
        state_dict = self._gather_state_dict()
        self._write_checkpoint(state_dict, output_dir)
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Called after each evaluation.
        Only saves checkpoint-best/ when the tracked metric improves.
        checkpoint-last/ is saved once at the end of training via on_train_end.
        """
        if not self.enable_saving:
            return control

        if metrics is None:
            return control

        metric_key = f"eval_{args.metric_for_best_model}"
        if metric_key not in metrics:
            logger.warning(f"Metric {metric_key} not found in evaluation metrics. Available: {metrics.keys()}")
            return control

        current_metric = metrics[metric_key]

        # Track latest metric/step for on_train_end logging
        self._last_metric_key = metric_key
        self._last_metric = current_metric
        self._last_global_step = state.global_step
        self._last_epoch = state.epoch

        should_save = True
        if self.distributed_state is not None and not self.distributed_state.is_main_process:
            should_save = False

        # Determine if current metric is better than best
        is_better = False
        if self.best_metric is None:
            is_better = True
        else:
            greater_is_better = args.greater_is_better if hasattr(args, 'greater_is_better') else True
            if greater_is_better:
                is_better = current_metric > self.best_metric
            else:
                is_better = current_metric < self.best_metric

        # Only save when metric improves — single get_state_dict call per improvement
        if is_better:
            self.best_metric = current_metric
            best_checkpoint_path = os.path.join(args.output_dir, "checkpoint-best")

            if self.trainer is not None:
                if should_save:
                    logger.info(f"New best {args.metric_for_best_model}: {current_metric:.4f}. Saving to {best_checkpoint_path}")
                self._save_model_deepspeed(best_checkpoint_path)
            elif should_save:
                logger.info(f"New best {args.metric_for_best_model}: {current_metric:.4f}. Saving to {best_checkpoint_path}")
                kwargs['model'].save_pretrained(best_checkpoint_path)
                kwargs['processing_class'].save_pretrained(best_checkpoint_path)

            if should_save:
                best_state_dict = {
                    'best_metric': self.best_metric,
                    'best_model_checkpoint': best_checkpoint_path,
                    'global_step': state.global_step,
                    'epoch': state.epoch,
                }
                torch.save(best_state_dict, os.path.join(best_checkpoint_path, "trainer_state.pt"))

            self.best_checkpoint_path = best_checkpoint_path

        return control

    def on_train_end(self, args, state, control, **kwargs):
        """
        Called once after training finishes.
        1. Runs a final evaluation on the current (last) model weights
        2. Gathers state_dict once
        3. Saves checkpoint-last/ (always)
        4. Saves checkpoint-best/ (if final eval metric improved)
        """
        if not self.enable_saving:
            return

        # Disable on_evaluate saving so our manual evaluate() doesn't trigger
        # a separate get_state_dict call inside on_evaluate
        self.enable_saving = False
        metrics = self.trainer.evaluate()
        self.enable_saving = True

        should_save = True
        if self.distributed_state is not None and not self.distributed_state.is_main_process:
            should_save = False

        # Check final eval metric
        metric_key = f"eval_{args.metric_for_best_model}"
        current_metric = metrics.get(metric_key)

        is_better = False
        if current_metric is not None:
            if self.best_metric is None:
                is_better = True
            else:
                greater_is_better = getattr(args, 'greater_is_better', True)
                if greater_is_better:
                    is_better = current_metric > self.best_metric
                else:
                    is_better = current_metric < self.best_metric

        # Gather state_dict once for all saves
        state_dict = self._gather_state_dict()

        # Always save checkpoint-last
        last_checkpoint_path = os.path.join(args.output_dir, "checkpoint-last")
        if should_save:
            logger.info(f"Training complete. Saving last checkpoint to {last_checkpoint_path} "
                        f"(step {state.global_step}, {metric_key}={current_metric})")
        self._write_checkpoint(state_dict, last_checkpoint_path)
        if should_save:
            torch.save({
                'global_step': state.global_step,
                'epoch': state.epoch,
                'metric_key': metric_key,
                'metric': current_metric,
            }, os.path.join(last_checkpoint_path, "trainer_state.pt"))

        # Save checkpoint-best if final eval improved
        if is_better:
            self.best_metric = current_metric
            best_checkpoint_path = os.path.join(args.output_dir, "checkpoint-best")
            if should_save:
                logger.info(f"New best {args.metric_for_best_model}: {current_metric:.4f}. "
                            f"Saving to {best_checkpoint_path}")
            self._write_checkpoint(state_dict, best_checkpoint_path)
            if should_save:
                torch.save({
                    'best_metric': self.best_metric,
                    'best_model_checkpoint': best_checkpoint_path,
                    'global_step': state.global_step,
                    'epoch': state.epoch,
                }, os.path.join(best_checkpoint_path, "trainer_state.pt"))
            self.best_checkpoint_path = best_checkpoint_path

        # Free gathered state dict
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

@dataclass
class NonTrainingArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    max_len: int = field(
        metadata={"help": "Tokenization maximum length."}
    )
    transform_method: str = field(
        metadata={'help': 'how sentence is processed. One value of ["entity_mask", "entity_marker", "entity_marker_punkt", "typed_entity_marker", "typed_entity_marker_punct", "subject_object_marker"]'}
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
    use_prompt_format: bool = field(
        default=False,
        metadata={"help": "Whether to wrap transformed text in an instruction prompt for relation extraction"}
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
        default="",
        metadata={"help": "Where the testing datasets are located, separate multiple by ;"}
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
        elif transform_method == "subject_object_marker":
            if entity_id == 'a':
                pre, post = "[SUBJECT] ", " [/SUBJECT]"
            else:
                pre, post = "[OBJECT] ", " [/OBJECT]"
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

def process_data(df_train, transform_method, move_entities_to_start, label_column_name="type", mode='train', use_prompt_format=False):
    if mode == 'train':
        df_train["label"] = df_train[label_column_name]
        for key, label in label_dict.items():
            df_train['label'] = df_train['label'].replace(label, key)
    elif mode == 'test':
        df_train["label"] = 0

    # Extract entity names from first span BEFORE transform_sentence modifies text
    entity_a_names = []
    entity_b_names = []
    for _, entry in df_train.iterrows():
        ea = entry["entity_a"]
        eb = entry["entity_b"]
        raw_text = entry["text"]
        entity_a_names.append(raw_text[ea[0][0]:ea[0][1]])
        entity_b_names.append(raw_text[eb[0][0]:eb[0][1]])

    df_train["text"] = [transform_sentence(entry, transform_method, move_entities_to_start) for _, entry in df_train.iterrows()]

    if use_prompt_format:
        relation_types_str = ", ".join(label_list)
        prompts = []
        for idx, (_, row) in enumerate(df_train.iterrows()):
            prompt = (
                "You are a biomedical relation extraction model.\n"
                "Determine the relation between the subject and object from these types:\n"
                f"{relation_types_str}\n\n"
                f"Text: {row['text']}\n"
                f"Entity 1: {entity_a_names[idx]}\n"
                f"Entity 2: {entity_b_names[idx]}"
            )
            prompts.append(prompt)
        df_train["text"] = prompts

    return df_train

def get_dataframes(cmd_str):
    str_list = cmd_str.split(";")
    dataframes = [pd.read_json(path) for path in str_list]
    return pd.concat(dataframes, ignore_index=True)

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
    print(f"  bf16: {training_args.bf16}")
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

    df_train = process_data(
        train_dataframe,
        non_train_args.transform_method,
        non_train_args.move_entities_to_start,
        non_train_args.label_column_name,
        use_prompt_format=non_train_args.use_prompt_format)
    df_val = process_data(
        validation_dataframe,
        non_train_args.transform_method,
        non_train_args.move_entities_to_start,
        non_train_args.label_column_name,
        use_prompt_format=non_train_args.use_prompt_format)

    if data_args.testing_dataframes:
        test_dataframe = get_dataframes(data_args.testing_dataframes)
        df_test = process_data(
            test_dataframe,
            non_train_args.transform_method,
            non_train_args.move_entities_to_start,
            non_train_args.label_column_name,
            mode='test',
            use_prompt_format=non_train_args.use_prompt_format)
    else:
        df_test = None

    # Model Parameters
    PRE_TRAINED_MODEL_NAME = non_train_args.model_name_or_path
    MAX_LEN = 512

    # Validate MXFP4 — this script is exclusively for MXFP4 models
    if not is_mxfp4_model(PRE_TRAINED_MODEL_NAME):
        raise ValueError(
            f"This script is for MXFP4 models only. "
            f"Model '{PRE_TRAINED_MODEL_NAME}' is not an MXFP4 model. "
            f"Use modeling_accelerate_qlora.py or modeling_accelerate.py instead."
        )

    # https://huggingface.co/blog/zero-deepspeed-fairscale
    tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    # GPT models need pad token set (they don't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Critical for causal LMs with sequence classification
    tokenizer.padding_side = "right"
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
    if df_test is not None:
        test_data = SentenceDataset(
            sentences=df_test.text.to_numpy(),
            labels=df_test.label.to_numpy(),
            tokenizer=tokenizer,
            max_len=MAX_LEN
        )
    else:
        test_data = None

    from transformers import AutoModelForSequenceClassification
    n_classes = len(label_dict)

    # For DeepSpeed ZeRO Stage 3, must create HfDeepSpeedConfig before model loading
    dschf = None
    if hasattr(training_args, 'deepspeed') and training_args.deepspeed:
        dschf = HfDeepSpeedConfig(training_args.deepspeed)
        logger.info(f"Initialized DeepSpeed config from: {training_args.deepspeed}")

    # Determine dtype for model loading
    if training_args.bf16:
        model_dtype = torch.bfloat16
    elif training_args.fp16:
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    # Dequantize MXFP4 → model_dtype for full fine-tuning
    logger.info(f"Detected MXFP4 model: {PRE_TRAINED_MODEL_NAME}")
    logger.info(f"Dequantizing from MXFP4 to {model_dtype} for fine-tuning")

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

    # Set pad_token_id in model config for GPT models
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Resize token embeddings to accommodate new special tokens
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

    # Ensure uniform dtype across all parameters (newly initialized layers like classification head)
    if model_dtype is not None:
        model = model.to(model_dtype)
    # Note: Trainer handles device placement when using Accelerate for distributed training

    from transformers import Trainer

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

    # Set trainer reference for DeepSpeed-aware saving
    save_best_callback.set_trainer(trainer)

    if training_args.do_train:
            train_result = trainer.train()

            metrics = train_result.metrics
            metrics["train_samples"] = len(train_data)
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)

    # Disable checkpoint saving after training completes
    save_best_callback.enable_saving = False

    # Wait for all processes to ensure checkpoint is saved before loading
    distributed_state.wait_for_everyone()

    # Best model reload commented out — use predict.py for evaluation
    # # Auto-detect checkpoint when not training (prediction-only mode)
    # # Checkpoints are always saved at output_dir/checkpoint-best and output_dir/checkpoint-last
    # if not training_args.do_train and save_best_callback.best_checkpoint_path is None:
    #     best_ckpt = os.path.join(training_args.output_dir, "checkpoint-best")
    #     if os.path.exists(best_ckpt):
    #         save_best_callback.best_checkpoint_path = best_ckpt
    #         logger.info(f"Prediction mode: Using checkpoint-best at {best_ckpt}")
    #     else:
    #         logger.warning(f"Prediction mode: checkpoint-best not found at {best_ckpt}")
    #
    # if save_best_callback.best_checkpoint_path is not None:
    #     logger.info(f"Loading best model from {save_best_callback.best_checkpoint_path}")
    #     n_classes = len(label_dict)
    #     model = AutoModelForSequenceClassification.from_pretrained(
    #         save_best_callback.best_checkpoint_path,
    #         num_labels=n_classes,
    #         torch_dtype=model_dtype,  # Use same dtype as initial model
    #     )
    #     tokenizer = AutoTokenizer.from_pretrained(save_best_callback.best_checkpoint_path)
    #     # Must resize embeddings to match the checkpoint (special tokens were added during training)
    #     model.resize_token_embeddings(len(tokenizer))
    #     # Ensure pad_token_id is set in model config
    #     if model.config.pad_token_id is None:
    #         model.config.pad_token_id = tokenizer.pad_token_id
    #     # Ensure uniform dtype and move to device
    #     if model_dtype is not None:
    #         model = model.to(model_dtype)
    #     model = model.to(device)
    #     trainer.model = model
    #     trainer.tokenizer = tokenizer
    # else:
    #     logger.info("No best checkpoint found, using current model (last checkpoint)")

    # Evaluation/prediction handled by predict.py
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #
    #     metrics = trainer.evaluate()
    #
    #     max_eval_samples = len(val_data)
    #     metrics["eval_samples"] = min(max_eval_samples, len(val_data))
    #
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)
    #
    #     # Generate predictions on validation dataset
    #     logger.info("*** Predict on validation dataset ***")
    #     ret = trainer.predict(val_data)
    #
    #     predictions = ret.predictions
    #     output_dir = training_args.output_dir
    #     with open(os.path.join(output_dir, 'predictions_prob_eval.txt'), 'w') as f:
    #         f.writelines('\n'.join(['\t'.join(list(map(str, x))) for x in predictions])+'\n')
    #     predicted_class = np.argmax(predictions, axis=1)
    #     predicted_class = [label_dict[cl] for cl in predicted_class]
    #     expanded_predicted_class = []
    #     for idx in range(len(predicted_class)):
    #         expanded_predicted_class.append(predicted_class[idx])
    #     df_val["predicted_class"] = expanded_predicted_class
    #     df_val.to_csv(os.path.join(output_dir, "eval_predictions.csv"))

    # if training_args.do_predict:
    #     ret = trainer.predict(test_data)
    #
    #     predictions = ret.predictions
    #     output_dir = training_args.output_dir
    #     with open(os.path.join(output_dir, 'predictions_prob.txt'), 'w') as f:
    #         f.writelines('\n'.join(['\t'.join(list(map(str, x))) for x in predictions])+'\n')
    #     predicted_class = np.argmax(predictions, axis=1)
    #     predicted_class = [label_dict[cl] for cl in predicted_class]
    #     expanded_predicted_class = []
    #     for idx in range(len(predicted_class)):
    #         expanded_predicted_class.append(predicted_class[idx])
    #     df_test["predicted_class"] = expanded_predicted_class
    #     df_test.to_csv(os.path.join(output_dir, "predictions.csv"))


if __name__ == "__main__":
    main()
