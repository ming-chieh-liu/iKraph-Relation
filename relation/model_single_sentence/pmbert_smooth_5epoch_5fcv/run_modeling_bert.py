# from importlib.metadata import metadata
import sys
import os
import logging
import copy
from dataclasses import dataclass, field
from typing import Optional

import transformers
import datasets
import pandas as pd
import numpy as np
import torch 
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments
from transformers import HfArgumentParser
from transformers import TrainerCallback
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from SentenceDataset import SentenceDataset
from modeling_bert import MyBertForSequenceClassification
from modeling_bert_triplet import BertTripletForSequenceClassification

logger = logging.getLogger(__name__)

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
        kwargs['tokenizer'].save_pretrained(last_checkpoint_path)

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
            kwargs['tokenizer'].save_pretrained(best_checkpoint_path)

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
    remove_cellline_organismtaxon: bool = field(
        metadata={"help": "Whether or not remove all entries with either organism taxon or cell line appearing."}
    )
    no_relation_file: Optional[str] = field(
        default="",
        metadata={"help": "Optional csv file indicating a pair of entity types not having relations"}
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


def main(config_dict=None):
    parser = HfArgumentParser((NonTrainingArguments, DataArguments, TrainingArguments))
    if config_dict is not None:
        non_train_args, data_args, training_args = parser.parse_dict(config_dict)
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        non_train_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        non_train_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    set_seed(42)

    def get_dataframes(cmd_str):
        str_list = cmd_str.split(";")
        dataframes = [pd.read_json(path, orient="table") for path in str_list]
        return dataframes

    train_config = {
        "max_len": non_train_args.max_len,
        "transform_method": non_train_args.transform_method,
        "model_type": non_train_args.model_type,
        "label_column_name": non_train_args.label_column_name,
        "move_entities_to_start": non_train_args.move_entities_to_start,
        "no_relation_file": non_train_args.no_relation_file,
        "remove_cellline_organismtaxon": non_train_args.remove_cellline_organismtaxon,
        "is_train": True
    }

    test_config = copy.deepcopy(train_config)
    test_config["is_train"] = False

    tokenizer = AutoTokenizer.from_pretrained(non_train_args.tokenizer_path)

    any_dataset = None
    if training_args.do_train:
        training_dataframes = get_dataframes(data_args.training_dataframes)
        train_dataset = SentenceDataset(training_dataframes, tokenizer, train_config)
        any_dataset = train_dataset
    else:
        train_dataset = None
    if training_args.do_eval:
        validation_dataframes = get_dataframes(data_args.validation_dataframes)
        eval_dataset = SentenceDataset(validation_dataframes, tokenizer, test_config)
        any_dataset = eval_dataset
    else:
        eval_dataset = None
    if training_args.do_predict:
        test_dataframes = get_dataframes(data_args.testing_dataframes)
        predict_dataset = SentenceDataset(test_dataframes, tokenizer, test_config)
        any_dataset = predict_dataset
    else:
        predict_dataset = None

    if any_dataset is None:
        raise ValueError("At least one of --do_train, --do_eval, --do_predict has to be true!")    
    n_classes = len(any_dataset.LABEL_DICT)

    model_configs = {
        "num_labels": n_classes,
        "id2label": any_dataset.LABEL_DICT,
        "label2id": {val: key for key, val in any_dataset.LABEL_DICT.items()}
    }

    if train_config["model_type"] == "cls":
        model = MyBertForSequenceClassification.from_pretrained(non_train_args.model_name_or_path, **model_configs)
    elif train_config["model_type"] == "triplet":
        model = BertTripletForSequenceClassification.from_pretrained(non_train_args.model_name_or_path, **model_configs)
    else:
        raise NotImplementedError

    if model.base_model_prefix == "roberta":
        model.roberta.resize_token_embeddings(len(any_dataset.tokenizer))
    elif model.base_model_prefix == "bert":
        model.bert.resize_token_embeddings(len(any_dataset.tokenizer))
    else:
        raise NotImplementedError

    model.config.id2label = any_dataset.LABEL_DICT
    model.config.label2id = {val: key for key, val in any_dataset.LABEL_DICT.items()}

    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    def compute_metrics(evalprediction_instance):
        predictions, label_ids = evalprediction_instance.predictions, evalprediction_instance.label_ids
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(label_ids, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(label_ids, predictions, labels=any_dataset.get_f1_true_labels(), average='micro', zero_division=0)
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    # Initialize custom callback for saving only the best model
    # Only enable saving during training to prevent nested checkpoints during prediction
    save_best_callback = SaveBestModelCallback(enable_saving=training_args.do_train)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        callbacks=[save_best_callback],
        tokenizer=tokenizer
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        # Disable checkpoint saving after training completes
        save_best_callback.enable_saving = False

        # Load best model if available
        if save_best_callback.best_checkpoint_path is not None:
            logger.info(f"Loading best model from {save_best_callback.best_checkpoint_path}")
            model = model.__class__.from_pretrained(save_best_callback.best_checkpoint_path, **model_configs)
            tokenizer = AutoTokenizer.from_pretrained(save_best_callback.best_checkpoint_path)

            model = model.to(trainer.args.device)

            trainer.model = model
            trainer.tokenizer = tokenizer
        else:
            logger.info("No best checkpoint found, using current model (last checkpoint)")

    print(f"Callback enable_saving status: {save_best_callback.enable_saving}")

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(eval_dataset, metric_key_prefix="predict_eval")
        predictions = predict_results.predictions
        metrics = predict_results.metrics

        trainer.log_metrics("predict_eval", metrics)
        trainer.save_metrics("predict_eval", metrics)

        # Save predictions
        if trainer.is_world_process_zero():
            predicted_class = np.argmax(predictions, axis=1)
            test_dataframe = eval_dataset.get_processed_dataframe()
            id2label = eval_dataset.LABEL_DICT
            predicted_class = [id2label[elem] for elem in predicted_class]
            test_dataframe["predicted_class"] = predicted_class

            logits_dataframe = pd.DataFrame(predictions, index=test_dataframe.index, columns=list(model.config.label2id))
            ret_dataframe = pd.concat([test_dataframe, logits_dataframe], axis=1)

            ret_dataframe.to_csv(os.path.join(training_args.output_dir, "predictions_eval.csv"))

        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = predict_results.predictions
        metrics = predict_results.metrics

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # Save predictions
        if trainer.is_world_process_zero():
            predicted_class = np.argmax(predictions, axis=1)
            test_dataframe = predict_dataset.get_processed_dataframe()
            test_dataframe["predicted_class"] = predicted_class

            logits_dataframe = pd.DataFrame(predictions, index=test_dataframe.index, columns=list(model.config.label2id))
            ret_dataframe = pd.concat([test_dataframe, logits_dataframe], axis=1)

            ret_dataframe.to_csv(os.path.join(training_args.output_dir, "predictions.csv"))


if __name__ == "__main__":
    main()
    # import os
    # os.environ["WANDB_PROJECT"] = "litcoin_sentence_model"
    # os.environ["WANDB_DISABLED"] = "true"
    # MAX_SPLITS = 5
    # import sys
    # # FOLD_IDX = int(sys.argv[1])
    # FOLD_IDX = 0
    # dataframes = [f"annotation_data/new_train_splits/split_{split_id}/data.json" for split_id in range(0, MAX_SPLITS)]
    # folds = {
    #     0: [[0, 1, 2], [3], [4]],
    #     1: [[1, 2, 3], [4], [0]],
    #     2: [[2, 3, 4], [0], [1]],
    #     3: [[3, 4, 0], [1], [2]],
    #     4: [[4, 0, 1], [2], [3]]
    # }
    # # TRAIN_SPLIT_INDEXES, VAL_SPLIT_INDEXES, TEST_SPLIT_INDEXES = folds[FOLD_IDX]
    # TRAIN_SPLIT_INDEXES, VAL_SPLIT_INDEXES, TEST_SPLIT_INDEXES = folds[FOLD_IDX]
    # TRAIN_SPLIT_INDEXES = TEST_SPLIT_INDEXES
# 
    # PRE_TRAINED_MODEL_NAME = "/data/xin.sui/litcoin_phase2/pre_trained_models/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf"
    # MAX_LEN = 384
    # BATCH_SIZE = 16
    # EPOCHS = 1
    # LEARNING_RATE = 3e-5
    # SAVE_PATH = "TEST_NEWDATA_split{}".format(0)
    # SAVE_PATH = "fine_tuned_models/{}".format(SAVE_PATH)
# 
    # train_dict = dict(
    #     output_dir=SAVE_PATH,
    #     num_train_epochs=EPOCHS,
    #     evaluation_strategy='steps',
    #     learning_rate=LEARNING_RATE,
    #     eval_steps=200,
    #     save_total_limit = 5,
    #     save_strategy='steps',
    #     save_steps=200,
    #     metric_for_best_model = 'f1',
    #     warmup_ratio=0.1,
    #     per_device_train_batch_size=BATCH_SIZE,
    #     per_device_eval_batch_size=BATCH_SIZE,
    #     # gradient_accumulation_steps=4,
    #     # eval_accumulation_steps=4,
    #     load_best_model_at_end=True,
    #     # deepspeed="dsconfig.json",
    #     # gradient_checkpointing=True,
    #     # fp16=True,  # 6903 -> 6231,
    #     # report_to="wandb"
    # )

    # config_dict = {
    #     "model_type": "cls",
    #     "max_len": 384,
    #     "transform_method": "entity_mask",
    #     "label_column_name": "annotated_type",
    #     "move_entities_to_start": True,
    #     "training_dataframes": ";".join([dataframes[idx] for idx in TEST_SPLIT_INDEXES]),
    #     "validation_dataframes": ";".join([dataframes[idx] for idx in TEST_SPLIT_INDEXES]),
    #     "testing_dataframes": ";".join([dataframes[idx] for idx in TEST_SPLIT_INDEXES]),
    #     "no_relation_file": "no_rel.csv",
    #     "overwrite_output_dir": True,
    #     "dataloader_num_workers": 4,
    #     "remove_cellline_organismtaxon": True,
    #     "fp16": True,  # 6903 -> 6231,
    #     "model_name_or_path": "/data/xin.sui/litcoin_phase2/pre_trained_models/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf",
    #     "do_train": True,
    #     "deepspeed": "dsconfig.json",
    #     **train_dict
    # }
    # main(config_dict)