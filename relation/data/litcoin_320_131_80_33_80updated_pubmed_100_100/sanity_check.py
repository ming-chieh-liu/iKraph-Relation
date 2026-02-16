#!/usr/bin/env python3
"""
Sanity check script to validate data integrity between annotation and train/test split files.

Validates:
1. No test PMIDs exist in the annotated training data
2. All annotated abstract_ids are present in the training PMID list
"""

import json
from pathlib import Path


def load_json(filepath: Path) -> dict:
    """Load JSON file and return parsed data."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_abstract_ids(annotated_data: dict) -> set:
    """Extract unique abstract_id values from annotated data."""
    abstract_ids = set()
    for record in annotated_data['data']:
        abstract_ids.add(record['abstract_id'])
    return abstract_ids


def main():
    # Define file paths
    base_dir = Path(__file__).parent
    file1_path = base_dir / "new_annotated_train_litcoin_320_131_80updated_pubmed_100_pd.json"
    file2_path = base_dir.parent / "litcoin_400_164_80updated_pubmed_100_100" / "train_test_pmids.json"

    print("Loading files...")
    print(f"File 1: {file1_path}")
    print(f"File 2: {file2_path}")
    print()

    # Load data
    annotated_data = load_json(file1_path)
    train_test_data = load_json(file2_path)

    # Extract data
    abstract_ids = extract_abstract_ids(annotated_data)
    train_pmids = set(train_test_data['train_pmid'])
    test_pmids = set(train_test_data['test_pmid'])

    # Print statistics
    print("=" * 70)
    print("DATA STATISTICS")
    print("=" * 70)
    print(f"Unique abstract_ids in file 1: {len(abstract_ids)}")
    print(f"Train PMIDs in file 2: {len(train_pmids)}")
    print(f"Test PMIDs in file 2: {len(test_pmids)}")
    print()

    # Check 1: No test PMIDs should exist in abstract_ids
    print("=" * 70)
    print("CHECK 1: Test PMIDs should NOT exist in annotated abstract_ids")
    print("=" * 70)
    test_in_annotated = abstract_ids & test_pmids

    if test_in_annotated:
        print(f"❌ FAILED: Found {len(test_in_annotated)} test PMIDs in annotated data!")
        print(f"Violating PMIDs: {sorted(test_in_annotated)}")
        check1_passed = False
    else:
        print("✓ PASSED: No test PMIDs found in annotated data")
        check1_passed = True
    print()

    # Check 2: All abstract_ids should be in train_pmids
    print("=" * 70)
    print("CHECK 2: All abstract_ids should exist in train PMIDs")
    print("=" * 70)
    not_in_train = abstract_ids - train_pmids

    if not_in_train:
        print(f"❌ FAILED: Found {len(not_in_train)} abstract_ids NOT in train PMIDs!")
        print(f"Missing PMIDs: {sorted(not_in_train)}")
        check2_passed = False
    else:
        print("✓ PASSED: All abstract_ids exist in train PMIDs")
        check2_passed = True
    print()

    # Final summary
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    if check1_passed and check2_passed:
        print("✓ ALL CHECKS PASSED")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        if not check1_passed:
            print("  - Test/train data leakage detected")
        if not check2_passed:
            print("  - Incomplete training data coverage")
        return 1


if __name__ == "__main__":
    exit(main())
