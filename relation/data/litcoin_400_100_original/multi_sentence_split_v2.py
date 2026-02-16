import os
import json
import argparse

parser =  argparse.ArgumentParser(description="Generate actual training data (multi-sentence) for LitCoin Phase 2.")
# parser.add_argument("--data_path", action="store", dest="data_path", help="Directory of processed LitCoin phase 2 data.")
parser.add_argument("--abstract_file", action="store", dest="abstract_file", default="../../data_processing/all_abstracts.json", help="Path to all LitCoin phase 2 abstract ids, use none to not use this file.")
args = parser.parse_args()

train_data_path = "./split/multi_sentence_train.json"
val_data_path = "./split/multi_sentence_val.json"
output_dir = "./multi_sentence_split_v2"
os.makedirs(output_dir, exist_ok=True)

all_data = json.load(open(train_data_path)) + json.load(open(val_data_path))
splits = json.load(open(args.abstract_file))

def flatten_list(l_of_l):
    ret = []
    for l in l_of_l:
        for elem in l:
            ret.append(elem)
    return ret

all_splits = []
for split_id in range(5):
    all_splits.append(splits[str(split_id)])

splits = [0, 1, 2, 3, 4]

all_this_train = []

for split in splits:
    all_train_ids = all_splits[split]
    # all_train_ids = flatten_list(all_train_ids)
    this_train = []
    for elem in all_data:
        if elem["abstract_id"] in all_train_ids:
            this_train.append(elem)
            all_this_train.append(elem)

    json.dump(this_train, open(os.path.join(output_dir, f"split_{split}.json"), "w"), indent=4)\


json.dump(all_this_train, open(os.path.join(output_dir, f"train_all.json"), "w"), indent=4)