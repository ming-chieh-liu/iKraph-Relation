import os
import json

# Path to the updated data (with corrected annotations)
all_data_path = "./multi_sentence_split_litcoin_600_80updated/train_all.json"

# Path to litcoin_600 split information
litcoin_600_splits_path = "../litcoin_600/multi_sentence_split_litcoin_600/all_abstracts_600.json"

# Output directory
output_dir = "./multi_sentence_split_litcoin_600_80updated"
os.makedirs(output_dir, exist_ok=True)

# Load all data
all_data = json.load(open(all_data_path))

# Load the predefined splits from litcoin_600
split_info = json.load(open(litcoin_600_splits_path))

# Create a mapping from abstract_id to split index
abstract_id_to_split = {}
for split_idx in range(5):
    for abstract_id in split_info[str(split_idx)]:
        abstract_id_to_split[abstract_id] = split_idx

# Create splits based on predefined mappings
splits = {i: [] for i in range(5)}
all_this_train = []

for elem in all_data:
    abstract_id = elem["abstract_id"]
    if abstract_id in abstract_id_to_split:
        split_idx = abstract_id_to_split[abstract_id]
        splits[split_idx].append(elem)
        all_this_train.append(elem)
    else:
        print(f"Warning: abstract_id {abstract_id} not found in litcoin_600 splits")

# Save each split
for split_idx in range(5):
    this_train = splits[split_idx]
    unique_abstracts = len(set(elem["abstract_id"] for elem in this_train))
    print(f"Split {split_idx}: {unique_abstracts} abstract_ids, {len(this_train)} records")
    json.dump(this_train, open(os.path.join(output_dir, f"split_{split_idx}.json"), "w"), indent=4)

# Save all training data
unique_total_abstracts = len(set(elem["abstract_id"] for elem in all_this_train))
json.dump(all_this_train, open(os.path.join(output_dir, f"train_all.json"), "w"), indent=4)
print(f"\nTotal: {unique_total_abstracts} abstract_ids, {len(all_this_train)} records")
