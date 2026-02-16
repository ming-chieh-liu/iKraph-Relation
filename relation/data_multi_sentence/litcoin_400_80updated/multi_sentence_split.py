import os
import json
import random

all_data_path = "./updated_train_all.json"
output_dir = "./multi_sentence_split_litcoin_400_80updated"
os.makedirs(output_dir, exist_ok=True)

# Load all data
all_data = json.load(open(all_data_path))

# Extract unique abstract_ids
unique_abstract_ids = list(set(elem["abstract_id"] for elem in all_data))

# Set seed for reproducibility
random.seed(42)
random.shuffle(unique_abstract_ids)

# Split into 5 folds
n_folds = 5
fold_size = len(unique_abstract_ids) // n_folds
folds = []

for i in range(n_folds):
    if i < n_folds - 1:
        fold_ids = unique_abstract_ids[i * fold_size:(i + 1) * fold_size]
    else:
        # Last fold gets remaining IDs
        fold_ids = unique_abstract_ids[i * fold_size:]
    folds.append(set(fold_ids))

# Create splits and save
all_this_train = []

for split_idx in range(n_folds):
    fold_abstract_ids = folds[split_idx]
    this_train = []

    for elem in all_data:
        if elem["abstract_id"] in fold_abstract_ids:
            this_train.append(elem)
            all_this_train.append(elem)

    print(f"Split {split_idx}: {len(fold_abstract_ids)} abstract_ids, {len(this_train)} records")
    json.dump(this_train, open(os.path.join(output_dir, f"split_{split_idx}.json"), "w"), indent=4)

# Save all training data
json.dump(all_this_train, open(os.path.join(output_dir, f"train_all.json"), "w"), indent=4)
print(f"\nTotal: {len(unique_abstract_ids)} abstract_ids, {len(all_this_train)} records")
