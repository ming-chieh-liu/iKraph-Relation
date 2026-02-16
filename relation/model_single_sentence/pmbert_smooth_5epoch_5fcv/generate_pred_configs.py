# generate prediction config files
import os
import sys
import json
import argparse 
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Generate training configurations')
    parser.add_argument('--models', type=str, help='Model paths')
    parser.add_argument("--test_data", type=str, default="./new_annotated_test_pd.json", help="Test data path")

    args = parser.parse_args()

    model_dir = Path(args.models).resolve()
    posfix = model_dir.name.replace("fine_tuned_models", "")
    test_data = Path(args.test_data).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    folds = {
        0: [[0, 1, 2], [3], [4]],
        1: [[1, 2, 3], [4], [0]],
        2: [[2, 3, 4], [0], [1]],
        3: [[3, 4, 0], [1], [2]],
        4: [[4, 0, 1], [2], [3]]
    }

    ckpt_name = "checkpoint-best"

    pred_config_path = Path(f"./pred_configs{posfix}_{test_data.stem}").resolve()

    for checkpoint_dir in model_dir.iterdir():
        if not checkpoint_dir.is_dir():
            continue 
        
        items = checkpoint_dir.name.split("_")
        FIDX = int(items[8])
        BS = int(items[9])
        LR = float(items[10])
        TOKENIZER_TYPE = items[11]
        SF = float(items[12][2:])
        CKPT = checkpoint_dir/ckpt_name
        FILE_NAME = f"{checkpoint_dir.name}-{CKPT.name}"
        # print(FIDX, BS, LR, MODEL_TYPE, SF)
        # print(checkpoint)
        # print(items)
        # raise KeyboardInterrupt()
        
        MAX_SPLITS = 5
        dataframes = [f"./new_train_splits{posfix}/split_{split_id}/data.json" for split_id in range(0, MAX_SPLITS)]
        TEST_DATA = test_data.as_posix()
        folds = {
            0: [[0, 1, 2], [3], [4]],
            1: [[1, 2, 3], [4], [0]],
            2: [[2, 3, 4], [0], [1]],
            3: [[3, 4, 0], [1], [2]],
            4: [[4, 0, 1], [2], [3]]
        }

        MODEL_TYPE = "triplet"
        TRANSFORM_METHOD = "typed_entity_marker_punct"
        LEARNING_RATE = LR
        MOVE_TO_START = False
        FOLD_IDX = FIDX

        TRAIN_SPLIT_INDEXES, VAL_SPLIT_INDEXES, TEST_SPLIT_INDEXES = folds[FOLD_IDX]

        if TOKENIZER_TYPE == "pmbert":
            PRE_TRAINED_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        elif TOKENIZER_TYPE == "roberta":
            PRE_TRAINED_MODEL_NAME = "../../pre_trained_models/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf"
        else:
            raise ValueError(f"Unknown model type: {MODEL_TYPE}")
        MAX_LEN = 384
        BATCH_SIZE = BS
        EPOCHS = 5
        SAVE_PATH = f"./results{posfix}_{test_data.stem}/{FILE_NAME}"

        train_dict = dict(
            output_dir=SAVE_PATH,
            num_train_epochs=EPOCHS,
            eval_strategy='no',
            learning_rate=LEARNING_RATE,
            # eval_steps=200,
            # save_total_limit = 10,
            save_strategy='no',
            # save_steps=200,
            metric_for_best_model = 'f1',
            warmup_ratio=0.1,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
    #                     load_best_model_at_end=True,
            label_smoothing_factor=SF,
        )
        config_dict = {
            "model_name_or_path": CKPT.as_posix(),
            "tokenizer_path": PRE_TRAINED_MODEL_NAME,
            "model_type": MODEL_TYPE,
            "max_len": MAX_LEN,
            "transform_method": TRANSFORM_METHOD,
            "label_column_name": "annotated_type",
            "move_entities_to_start": MOVE_TO_START,
            "training_dataframes": ";".join([dataframes[idx] for idx in TRAIN_SPLIT_INDEXES+VAL_SPLIT_INDEXES]),
            "validation_dataframes": ";".join([dataframes[idx] for idx in TEST_SPLIT_INDEXES]),
            "testing_dataframes": TEST_DATA,
            "no_relation_file": "./no_rel.csv",
            "overwrite_output_dir": True,
            "dataloader_num_workers": 4,
            "remove_cellline_organismtaxon": True,
            "do_train": False,
            "do_eval": True,
            "do_predict": True,
            **train_dict
        }

        FILE_PATH = pred_config_path/f"{FILE_NAME}.json"
        FILE_PATH.parent.mkdir(parents=True, exist_ok=True)

        json.dump(config_dict, open(f"{FILE_PATH.as_posix()}", "w", encoding="utf-8"))

if __name__ == "__main__":
    main()