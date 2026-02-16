#!/usr/bin/env python
# coding: utf-8

from concurrent.futures import process
from random import randrange

import pandas as pd
import json
import torch
from torch import nn
import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers import TrainerCallback
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from collections import defaultdict
import numpy as np

import logging

from torch.utils.data import Dataset, DataLoader

import sys
import os
import argparse
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser, set_seed
from transformers import TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)

label_list = ["NOT", "Association", "Positive_Correlation", "Negative_Correlation", "Bind", "Cotreatment", "Comparison", "Drug_Interaction", "Conversion"]
label_dict = {idx: val for idx, val in enumerate(label_list)}

class SaveBestModelCallback(TrainerCallback):
    """
    Custom callback that saves both the best model and the last model.
    - checkpoint-best/: Contains the model with the best evaluation metric
    - checkpoint-last/: Contains the most recent model (updated every evaluation)
    """
    def __init__(self, enable_saving=True):
        self.enable_saving = enable_saving
        self.best_metric = None
        self.best_checkpoint_path = None

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

        # Always save the last checkpoint
        last_checkpoint_path = os.path.join(args.output_dir, "checkpoint-last")

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

        # Save best checkpoint if improved
        if is_better:
            self.best_metric = current_metric
            best_checkpoint_path = os.path.join(args.output_dir, "checkpoint-best")

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

    # Simple device detection for single-GPU training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # Configure 4-bit quantization using BitsAndBytes (enables QLoRA training)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        PRE_TRAINED_MODEL_NAME,
        num_labels=n_classes,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
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

    # Prepare quantized model for k-bit training
    # This freezes base model weights and enables gradient checkpointing
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA for sequence classification
    # This enables fine-tuning of quantized models by adding small trainable adapter layers
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,                              # LoRA rank
        lora_alpha=32,                     # Scaling factor
        lora_dropout=0.1,
        bias="none",
        target_modules="all-linear",       # Auto-detect linear layers
        modules_to_save=["score"],         # Keep classification head trainable
    )

    # Apply LoRA adapters
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    from transformers import Trainer

    # Example JSON config for TrainingArguments (to use with --config.json):
    # {
    #   "output_dir": "BERT_train",
    #   "num_train_epochs": 5,
    #   "evaluation_strategy": "epoch",
    #   "learning_rate": 3e-5,
    #   "save_strategy": "no",
    #   "metric_for_best_model": "f1",
    #   "greater_is_better": true,
    #   "warmup_ratio": 0.2,
    #   "per_device_train_batch_size": 4,
    #   "per_device_eval_batch_size": 4,
    #   "load_best_model_at_end": false,
    #   "fp16": false
    # }
    # Note: save_strategy='no' because we use SaveBestModelCallback to save only best/last models

    # https://stackoverflow.com/questions/69087044/early-stopping-in-bert-trainer-instances
    from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
    # def compute_f1(evalprediction_instance):
    #     predictions, label_ids = evalprediction_instance.predictions, evalprediction_instance.label_ids
    #     predictions = np.argmax(predictions, axis=1)
    #     accuracy = accuracy_score(label_ids, predictions)
    #     f1 = f1_score(label_ids, predictions, labels=range(1, len(label_dict)), average='micro')
    #     return {"accuracy": accuracy, "f1": f1}

    def compute_metrics(evalprediction_instance):
        predictions, label_ids = evalprediction_instance.predictions, evalprediction_instance.label_ids
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(label_ids, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(label_ids, predictions,  labels=range(1, len(label_dict)), average='micro', zero_division=0)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    # Initialize custom callback for saving only the best model
    save_best_callback = SaveBestModelCallback(enable_saving=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        # compute_metrics=compute_f1,
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
    # With LoRA, checkpoints only contain adapter weights, so we must:
    # 1. Load the original base model
    # 2. Load the LoRA adapter weights on top
    if save_best_callback.best_checkpoint_path is not None:
        logger.info(f"Loading best model from {save_best_callback.best_checkpoint_path}")
        n_classes = len(label_dict)
        # Load the base model first (with same quantization config)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            PRE_TRAINED_MODEL_NAME,  # Original base model path
            num_labels=n_classes,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=bnb_config,
        )
        # Load LoRA adapter weights on top of base model
        model = PeftModel.from_pretrained(
            base_model,
            save_best_callback.best_checkpoint_path,  # LoRA adapter weights
        )
        tokenizer = AutoTokenizer.from_pretrained(save_best_callback.best_checkpoint_path)
        # Ensure pad_token_id is set in model config
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
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


if __name__ == "__main__":
    main()
