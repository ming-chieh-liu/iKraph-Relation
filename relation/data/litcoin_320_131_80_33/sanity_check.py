import pandas as pd

# Load filter PMIDs
filter_data = pd.read_json("../litcoin_80updated_pubmed_200/new_annotated_train_litcoin_80updated_pubmed_200_pd.json", orient='table')
filter_pmids = set(filter_data["abstract_id"])

# Load test data
test_data = pd.read_json("./new_annotated_test_litcoin_80_33_pd.json", orient='table')
test_pmids = set(test_data["abstract_id"])

# Check for intersection
intersection = filter_pmids.intersection(test_pmids)

print(f"Filter PMIDs count: {len(filter_pmids)}")
print(f"Test PMIDs count: {len(test_pmids)}")
print(f"Intersection count: {len(intersection)}")

if len(intersection) == 0:
    print("\n✓ PASS: No filter PMIDs found in test set")
else:
    print(f"\n✗ FAIL: Found {len(intersection)} filter PMIDs in test set:")
    print(f"Overlapping PMIDs: {intersection}")
