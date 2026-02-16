# generate prediction config files
import os
import sys
import json

pred_config_path = f"./pred_configs_with_extra"
if not os.path.exists(pred_config_path): os.mkdir(pred_config_path) 
checkpoints = json.load(open(f"./selected_checkpoints.json", "r", encoding="utf-8"))

for checkpoint in [ckp for stn in checkpoints for ckp in stn]:
    CKPOINT = f"./fine_tuned_models_with_extra{checkpoint[1:]}"
    items = checkpoint.split("_")
    fn_items = checkpoint.split("/")
    FIDX = int(items[8])
    BS = int(items[9])
    LR = float(items[10])
    SF = float(items[12].split("/")[0][2:])
    CKP = items[12].split("/")[1]
    FILE_NAME = f"{fn_items[1]}_{fn_items[2]}"
#     print(FIDX, BS, LR, LS, CKP, FILE_NAME)
    
    os.environ["WANDB_PROJECT"] = "litcoin_sentence_model"
    MAX_SPLITS = 5
    dataframes = [f"./new_train_splits/split_{split_id}/data.json" for split_id in range(0, MAX_SPLITS)]
    test_data = f"../../pubmed_data/annotated_train_pubmed_pd.json"
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

    PRE_TRAINED_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

    MAX_LEN = 384
    BATCH_SIZE = BS
    EPOCHS = 5
    SAVE_PATH = CKPOINT

    train_dict = dict(
        output_dir=SAVE_PATH,
        num_train_epochs=EPOCHS,
        eval_strategy='steps', # steps
        learning_rate=LEARNING_RATE,
        eval_steps=200,
        save_total_limit = 10,
        save_strategy='steps',
        save_steps=200,
        metric_for_best_model = 'f1',
        warmup_ratio=0.1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
#                     load_best_model_at_end=True,
        label_smoothing_factor=SF,
    )
    config_dict = {
        "model_name_or_path": CKPOINT,
        "tokenizer_path": PRE_TRAINED_MODEL_NAME,
        "model_type": MODEL_TYPE,
        "max_len": MAX_LEN,
        "transform_method": TRANSFORM_METHOD,
        "label_column_name": "annotated_type",
        "move_entities_to_start": MOVE_TO_START,
        "training_dataframes": ";".join([dataframes[idx] for idx in TRAIN_SPLIT_INDEXES+VAL_SPLIT_INDEXES+TEST_SPLIT_INDEXES]),
        "validation_dataframes": ";".join([dataframes[idx] for idx in TRAIN_SPLIT_INDEXES+VAL_SPLIT_INDEXES+TEST_SPLIT_INDEXES]),
        "testing_dataframes": test_data,
        "no_relation_file": "./no_rel.csv",
        "overwrite_output_dir": True,
        "dataloader_num_workers": 4,
        "remove_cellline_organismtaxon": True,
        "do_train": False,
        "do_eval": True,
        "do_predict": True,
        **train_dict
    }

    json.dump(config_dict, open(f"{pred_config_path}/{FILE_NAME}.json", "w", encoding="utf-8"))