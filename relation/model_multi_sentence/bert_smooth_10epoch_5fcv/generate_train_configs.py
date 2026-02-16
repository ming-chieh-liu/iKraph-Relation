import argparse 
from pathlib import Path
import json 

def main():
    parser = argparse.ArgumentParser(description='Generate training configurations')
    parser.add_argument('--posfix', type=str, help='Postfix for configuration files')
    parser.add_argument('--transform_method', type=str, default='typed_entity_marker_punct', help='Data transformation method')
    # entity_mask
    args = parser.parse_args()

    posfix = args.posfix
    transform_method = args.transform_method 

    config_dir = Path(f'./configs{posfix}_{transform_method}')
    config_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(f'./fine_tuned_models{posfix}_{transform_method}')

    training_dir = Path(f'./multi_sentence_split{posfix}')

    models = ["roberta"]

    for model in models: 
        
        for fold in range(5):
            config_dict = {}
            config_dict["model_name_or_path"] =  "../../pre_trained_models/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf"
            config_dict["tokenizer_path"] = "../../pre_trained_models/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf"
            config_dict["model_type"] = "triplet"
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

            training_paths = ";".join([f"./{training_dir}/split_{i}.json" for i in training_splits])
            validation_path = f"./{training_dir}/split_{validation_split}.json"

            config_dict["training_dataframes"] = training_paths
            config_dict["validation_dataframes"] = validation_path
            config_dict["testing_dataframes"] = "./multi_sentence_test.json"
            # config_dict["no_relation_file"] = f"./{training_dir}/no_rel.csv"
            config_dict["overwrite_output_dir"] = True
            config_dict["dataloader_num_workers"] = 4
            # config_dict["remove_cellline_organismtaxon"] = True
            config_dict["do_train"] = True
            config_dict["do_eval"] = True
            config_dict["do_predict"] = True
            config_dict["num_train_epochs"] = 10
            config_dict["eval_strategy"] = "steps"
            lr = 3e-05
            config_dict["learning_rate"] = lr
            config_dict["eval_steps"] = 200
            # config_dict["save_total_limit"] = 10
            config_dict["save_strategy"] = "no" # "steps"
            # config_dict["save_steps"] = 200
            config_dict["metric_for_best_model"] = "f1"
            config_dict["warmup_ratio"] = 0.1
            bs = 16
            config_dict["per_device_train_batch_size"] = bs
            config_dict["per_device_eval_batch_size"] = bs
            config_dict["load_best_model_at_end"] = False # True 
            sf = 0.02
            config_dict["label_smoothing_factor"] = sf
            config_dict["output_dir"] = f"{output_dir}/NEWDATA_triplet_{transform_method}_False_split_{fold}_{bs}_{lr}_{model}_ls{sf}"

            config_path = config_dir / f"config_{model}_ls{sf}_split_{fold}_{bs}_{lr}.json"

            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=4)

if __name__ == "__main__":
    main()