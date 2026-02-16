import pandas as pd
from pathlib import Path

# Define file paths
base_dir = Path("/data/mliu/iKraph/relation/data")
file1 = base_dir / "litcoin_400_164_80updated_pubmed_160_40/new_annotated_train_litcoin_400_164_80updated_pubmed_160_pd.json"
file2 = base_dir / "litcoin_400_164_80updated_pubmed_160_40/new_annotated_test_pubmed_40_pd.json"
file3 = base_dir / "litcoin_400_164_80updated_pubmed_100_100/new_annotated_train_litcoin_400_164_80updated_pubmed_100_pd.json"
file4 = base_dir / "litcoin_400_164_80updated_pubmed_100_100/new_annotated_test_pubmed_100_pd.json"

# Load abstract_ids from each file
def load_abstract_ids(filepath):
    df = pd.read_json(filepath, orient="table")
    return set(df['abstract_id'].unique())

print("Loading abstract IDs from files...")
ids1 = load_abstract_ids(file1)
ids2 = load_abstract_ids(file2)
ids3 = load_abstract_ids(file3)
ids4 = load_abstract_ids(file4)

print(f"\nFile 1 (train_160_40): {len(ids1)} unique abstract_ids")
print(f"File 2 (test_160_40):  {len(ids2)} unique abstract_ids")
print(f"File 3 (train_100_100): {len(ids3)} unique abstract_ids")
print(f"File 4 (test_100_100):  {len(ids4)} unique abstract_ids")

print("\n" + "="*80)
print("VERIFICATION RESULTS")
print("="*80)

# Check 1: file 1 and 2 don't have overlapped abstract_id
overlap_1_2 = ids1 & ids2
print(f"\n1. File 1 and File 2 overlap check:")
if len(overlap_1_2) == 0:
    print(f"   ✓ PASS: No overlap (0 common abstract_ids)")
else:
    print(f"   ✗ FAIL: {len(overlap_1_2)} overlapping abstract_ids")
    print(f"   Overlapping IDs: {sorted(list(overlap_1_2))[:10]}..." if len(overlap_1_2) > 10 else f"   Overlapping IDs: {sorted(list(overlap_1_2))}")

# Check 2: file 3 and 4 don't have overlapped abstract_id
overlap_3_4 = ids3 & ids4
print(f"\n2. File 3 and File 4 overlap check:")
if len(overlap_3_4) == 0:
    print(f"   ✓ PASS: No overlap (0 common abstract_ids)")
else:
    print(f"   ✗ FAIL: {len(overlap_3_4)} overlapping abstract_ids")
    print(f"   Overlapping IDs: {sorted(list(overlap_3_4))[:10]}..." if len(overlap_3_4) > 10 else f"   Overlapping IDs: {sorted(list(overlap_3_4))}")

# Check 3: file 2 and 3 don't have overlapped abstract_id
overlap_2_3 = ids2 & ids3
print(f"\n3. File 2 and File 3 overlap check:")
if len(overlap_2_3) == 0:
    print(f"   ✓ PASS: No overlap (0 common abstract_ids)")
else:
    print(f"   ✗ FAIL: {len(overlap_2_3)} overlapping abstract_ids")
    print(f"   Overlapping IDs: {sorted(list(overlap_2_3))[:10]}..." if len(overlap_2_3) > 10 else f"   Overlapping IDs: {sorted(list(overlap_2_3))}")

# Check 4: file 2's abstract_id should all be in file 4
ids2_in_ids4 = ids2.issubset(ids4)
diff_2_4 = ids2 - ids4
print(f"\n4. File 2 abstract_ids all in File 4 check:")
if ids2_in_ids4:
    print(f"   ✓ PASS: All {len(ids2)} abstract_ids from File 2 are in File 4")
else:
    print(f"   ✗ FAIL: {len(diff_2_4)} abstract_ids from File 2 are NOT in File 4")
    print(f"   Missing IDs: {sorted(list(diff_2_4))[:10]}..." if len(diff_2_4) > 10 else f"   Missing IDs: {sorted(list(diff_2_4))}")

# Check 5: file 3's abstract_id should all be in file 1
ids3_in_ids1 = ids3.issubset(ids1)
diff_3_1 = ids3 - ids1
print(f"\n5. File 3 abstract_ids all in File 1 check:")
if ids3_in_ids1:
    print(f"   ✓ PASS: All {len(ids3)} abstract_ids from File 3 are in File 1")
else:
    print(f"   ✗ FAIL: {len(diff_3_1)} abstract_ids from File 3 are NOT in File 1")
    print(f"   Missing IDs: {sorted(list(diff_3_1))[:10]}..." if len(diff_3_1) > 10 else f"   Missing IDs: {sorted(list(diff_3_1))}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
all_pass = (len(overlap_1_2) == 0 and len(overlap_3_4) == 0 and
            len(overlap_2_3) == 0 and ids2_in_ids4 and ids3_in_ids1)
if all_pass:
    print("✓ All checks PASSED")
else:
    print("✗ Some checks FAILED - see details above")
