# aggregate prediction files
import os
import json

fine_tuned_models = f"./fine_tuned_models_with_extra"
target_preds_folder = f"./predictions_with_extra"
if not os.path.exists(target_preds_folder): os.mkdir(target_preds_folder)
checkpoints = json.load(open(f"./selected_checkpoints.json", "r", encoding="utf-8"))

for setting in checkpoints:
    for i, checkpoint in enumerate(setting):
        fn_items = checkpoint.split("/")
        FILE_NAME = f"{fn_items[1]}_{i+1}"[48:]
        CKPOINT = f"{fine_tuned_models}{checkpoint[1:]}"
        tmpComm = f"cp {CKPOINT}/predictions_eval.csv {target_preds_folder}/{FILE_NAME}_predictions_eval.csv"
        print(tmpComm)
        os.system(tmpComm)
        tmpComm = f"cp {CKPOINT}/predictions.csv {target_preds_folder}/{FILE_NAME}_predictions.csv"
        print(tmpComm)
        os.system(tmpComm)
        print()