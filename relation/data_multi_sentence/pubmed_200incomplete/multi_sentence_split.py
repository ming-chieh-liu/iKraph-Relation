import os
import json
import random

# Paths
all_data_path = "./processed/multi_sentence_all.json"
output_dir = "./multi_sentence_split_pubmed_200incomplete"
os.makedirs(output_dir, exist_ok=True)

print("=" * 80)
print("PUBMED 200 INCOMPLETE PMID SPLITTING")
print("=" * 80)

# Step 1: Load multi-sentence data
print("\n[Step 1] Loading multi-sentence data...")
all_data = json.load(open(all_data_path))
print(f"  Loaded {len(all_data)} relation records")

# Step 2: Extract all unique PMIDs
print("\n[Step 2] Extracting unique PMIDs...")
all_pmids = sorted(list(set([record["abstract_id"] for record in all_data])))
print(f"  Found {len(all_pmids)} unique PMIDs")

# Step 3: Shuffle PMIDs with fixed seed for reproducibility
print("\n[Step 3] Shuffling PMIDs with fixed seed (42)...")
random.seed(42)
shuffled_pmids = all_pmids.copy()
random.shuffle(shuffled_pmids)

# Step 4: Distribute PMIDs evenly across 5 folds
print("\n[Step 4] Distributing PMIDs across 5 folds...")
num_folds = 5
pmids_per_fold = len(shuffled_pmids) // num_folds
remainder = len(shuffled_pmids) % num_folds

folds = {}
start_idx = 0
for i in range(num_folds):
    # Distribute remainder PMIDs to first folds
    fold_size = pmids_per_fold + (1 if i < remainder else 0)
    folds[i] = shuffled_pmids[start_idx:start_idx + fold_size]
    start_idx += fold_size
    print(f"  Fold {i}: {len(folds[i])} PMIDs")

# Step 5: Create all_abstracts.json for reproducibility
print("\n[Step 5] Creating all_abstracts.json for reproducibility...")
all_abstracts = {
    "all_pmids": all_pmids
}
for i in range(num_folds):
    all_abstracts[str(i)] = folds[i]

output_abstracts_file = os.path.join(output_dir, "all_abstracts.json")
json.dump(all_abstracts, open(output_abstracts_file, "w"), indent=4)
print(f"  Saved to {output_abstracts_file}")

# Step 6: Convert folds to sets for faster lookup
fold_sets = {i: set(folds[i]) for i in range(num_folds)}

# Step 7: Split data by fold
print("\n[Step 6] Splitting relation data by folds...")
fold_data = {i: [] for i in range(num_folds)}
all_train_data = []

for record in all_data:
    abstract_id = record["abstract_id"]
    for fold_idx in range(num_folds):
        if abstract_id in fold_sets[fold_idx]:
            fold_data[fold_idx].append(record)
            all_train_data.append(record)
            break

# Step 8: Save split files
print("\n[Step 7] Saving split files...")
for fold_idx in range(num_folds):
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
print("Splitting completed successfully!")
print("=" * 80)
