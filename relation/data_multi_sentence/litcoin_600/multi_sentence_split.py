import os
import json
import random

# Paths
all_data_path = "./processed/multi_sentence_all.json"
existing_split_path = "/data/mliu/iKraph/relation/data_processing/all_abstracts.json"
all_json_path = "./All.json"
output_dir = "./multi_sentence_split_litcoin_600"
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("LITCOIN 600 PMID SPLITTING")
print("=" * 80)

# Step 1: Load existing 400 PMID split
print("\n[Step 1] Loading existing 400 PMID split...")
existing_split = json.load(open(existing_split_path))
existing_pmids_400 = set(existing_split['all_pmids'])
print(f"  Loaded {len(existing_pmids_400)} PMIDs from existing split")

# Verify fold structure
existing_folds = {}
for i in range(5):
    existing_folds[i] = set(existing_split[str(i)])
    print(f"  Fold {i}: {len(existing_folds[i])} PMIDs")

# Step 2: Load all 600 PMIDs from All.json
print("\n[Step 2] Loading all 600 PMIDs from All.json...")
all_json = json.load(open(all_json_path))
all_pmids_600 = set([int(doc['document_id']) for doc in all_json])
print(f"  Loaded {len(all_pmids_600)} PMIDs from All.json")

# Verify all 400 are in 600
assert existing_pmids_400.issubset(all_pmids_600), "Not all 400 PMIDs are in the 600 dataset!"
print(f"  ✓ All 400 PMIDs are present in the 600 dataset")

# Step 3: Identify new 200 PMIDs
print("\n[Step 3] Identifying new PMIDs...")
new_pmids_200 = sorted(list(all_pmids_600 - existing_pmids_400))
print(f"  Found {len(new_pmids_200)} new PMIDs")

# Step 4: Distribute 200 new PMIDs evenly across 5 folds (40 each)
print("\n[Step 4] Distributing new PMIDs evenly across folds...")
random.seed(42)
random.shuffle(new_pmids_200)

# Calculate how many to add to each fold (40 per fold)
pmids_per_fold = 40
extended_folds = {}

for i in range(5):
    # Start with existing PMIDs in this fold
    extended_folds[i] = list(existing_folds[i])

    # Add new PMIDs
    start_idx = i * pmids_per_fold
    end_idx = start_idx + pmids_per_fold
    new_for_fold = new_pmids_200[start_idx:end_idx]
    extended_folds[i].extend(new_for_fold)

    print(f"  Fold {i}: {len(existing_folds[i])} → {len(extended_folds[i])} PMIDs (+{len(new_for_fold)})")

# Step 5: Create updated all_abstracts.json
print("\n[Step 5] Creating updated all_abstracts_600.json...")
all_abstracts_600 = {
    "all_pmids": sorted(list(all_pmids_600))
}
for i in range(5):
    all_abstracts_600[str(i)] = extended_folds[i]

output_split_file = os.path.join(output_dir, "all_abstracts_600.json")
json.dump(all_abstracts_600, open(output_split_file, "w"), indent=4)
print(f"  Saved to {output_split_file}")

# Step 6: Load multi-sentence data and split by fold
print("\n[Step 6] Splitting multi-sentence data by folds...")
all_data = json.load(open(all_data_path))
print(f"  Loaded {len(all_data)} relation records")

# Convert folds to sets for faster lookup
fold_sets = {i: set(extended_folds[i]) for i in range(5)}

# Split data by fold
fold_data = {i: [] for i in range(5)}
all_train_data = []

for record in all_data:
    abstract_id = record["abstract_id"]
    for fold_idx in range(5):
        if abstract_id in fold_sets[fold_idx]:
            fold_data[fold_idx].append(record)
            all_train_data.append(record)
            break

# Step 7: Save split files
print("\n[Step 7] Saving split files...")
for fold_idx in range(5):
    fold_file = os.path.join(output_dir, f"split_{fold_idx}.json")
    json.dump(fold_data[fold_idx], open(fold_file, "w"), indent=4)

    unique_abstracts = len(set(r["abstract_id"] for r in fold_data[fold_idx]))
    print(f"  split_{fold_idx}.json: {unique_abstracts} abstracts, {len(fold_data[fold_idx])} records")

# Save all training data
train_all_file = os.path.join(output_dir, "train_all.json")
json.dump(all_train_data, open(train_all_file, "w"), indent=4)
unique_abstracts_total = len(set(r["abstract_id"] for r in all_train_data))
print(f"\n  train_all.json: {unique_abstracts_total} abstracts, {len(all_train_data)} records")

print("\n" + "=" * 80)
print("✓ Splitting completed successfully!")
print("=" * 80)
