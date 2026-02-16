"""
Sanity check script for relation annotations in litcoin_200 dataset.

Validates that relations do not violate the following rules:
1. NO_REL_TYPES: Relations cannot involve OrganismTaxon or CellLine entity types
2. NO_REL_PAIRS: Specific entity type pairs cannot have relations
"""

import json
from typing import List, Dict, Tuple, Set
from pathlib import Path


NO_REL_TYPES = ["OrganismTaxon", "CellLine"]
NO_REL_PAIRS = [
    ["DiseaseOrPhenotypicFeature", "DiseaseOrPhenotypicFeature"],
    ["SequenceVariant", "GeneOrGeneProduct"],
    ["GeneOrGeneProduct", "SequenceVariant"],
    ["SequenceVariant", "SequenceVariant"]
]


def check_violation(type1: str, type2: str) -> Tuple[bool, str]:
    """
    Check if a relation between two entity types violates the rules.

    Returns:
        Tuple of (is_violation, violation_reason)
    """
    if type1 in NO_REL_TYPES:
        return True, f"{type1} cannot have relations with any entity"

    if type2 in NO_REL_TYPES:
        return True, f"{type2} cannot have relations with any entity"

    for pair in NO_REL_PAIRS:
        if [type1, type2] == pair:
            return True, f"Forbidden pair: {type1} - {type2}"

    return False, ""


def check_doc_level_file(filepath: str) -> List[Dict]:
    """
    Check violations in document-level file (All_with_empty_relations.json).

    Returns:
        List of violation records
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    violations = []
    total_relations = 0

    for doc in data:
        doc_id = doc.get('document_id', 'unknown')
        relations = doc.get('relation', [])

        for rel in relations:
            total_relations += 1
            ent1 = rel.get('ent1', {})
            ent2 = rel.get('ent2', {})
            type1 = ent1.get('type', '')
            type2 = ent2.get('type', '')

            is_violation, reason = check_violation(type1, type2)

            if is_violation:
                violations.append({
                    'file': 'All_with_empty_relations.json',
                    'document_id': doc_id,
                    'entity_1': ent1.get('Name', [''])[0] if ent1.get('Name') else '',
                    'type_1': type1,
                    'entity_2': ent2.get('Name', [''])[0] if ent2.get('Name') else '',
                    'type_2': type2,
                    'reason': reason
                })

    return violations, total_relations


def check_sent_level_file(filepath: str) -> List[Dict]:
    """
    Check violations in sentence-level file (litcoin_200_sent_for_annotation.json).

    Returns:
        List of violation records
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    violations = []
    total_relations = 0

    for sent in data:
        sent_name = sent.get('Name', 'unknown')
        pmid = sent.get('PMID', 'unknown')
        relations = sent.get('Relation_Annotation', [])

        for rel in relations:
            total_relations += 1
            type1 = rel.get('type_1', '')
            type2 = rel.get('type_2', '')

            is_violation, reason = check_violation(type1, type2)

            if is_violation:
                violations.append({
                    'file': 'litcoin_200_sent_for_annotation.json',
                    'sentence_name': sent_name,
                    'PMID': pmid,
                    'entity_1': rel.get('entity_1', ''),
                    'type_1': type1,
                    'entity_2': rel.get('entity_2', ''),
                    'type_2': type2,
                    'reason': reason
                })

    return violations, total_relations


def print_violations(violations: List[Dict], file_name: str) -> None:
    """Print violations in a readable format."""
    if not violations:
        print(f"✓ No violations found in {file_name}")
        return

    print(f"\n✗ Found {len(violations)} violation(s) in {file_name}:")
    print("-" * 100)

    for i, v in enumerate(violations, 1):
        print(f"\nViolation #{i}:")
        if 'document_id' in v:
            print(f"  Document ID: {v['document_id']}")
        else:
            print(f"  Sentence: {v['sentence_name']}")
            print(f"  PMID: {v['PMID']}")

        print(f"  Entity 1: {v['entity_1']} (Type: {v['type_1']})")
        print(f"  Entity 2: {v['entity_2']} (Type: {v['type_2']})")
        print(f"  Reason: {v['reason']}")


def main():
    """Main function to run sanity checks."""
    base_dir = Path(__file__).parent
    file1 = base_dir / "All_with_empty_relations.json"
    file2 = base_dir / "litcoin_200_sent_for_annotation.json"

    print("=" * 100)
    print("RELATION SANITY CHECK")
    print("=" * 100)

    print("\nValidation Rules:")
    print(f"1. NO_REL_TYPES: {NO_REL_TYPES}")
    print(f"2. NO_REL_PAIRS: {NO_REL_PAIRS}")

    # Check document-level file
    print(f"\n{'=' * 100}")
    print(f"Checking File 1: {file1.name}")
    print(f"{'=' * 100}")
    doc_violations, doc_total = check_doc_level_file(file1)
    print(f"\nTotal relations checked: {doc_total}")
    print_violations(doc_violations, file1.name)

    # Check sentence-level file
    print(f"\n{'=' * 100}")
    print(f"Checking File 2: {file2.name}")
    print(f"{'=' * 100}")
    sent_violations, sent_total = check_sent_level_file(file2)
    print(f"\nTotal relations checked: {sent_total}")
    print_violations(sent_violations, file2.name)

    # Summary
    print(f"\n{'=' * 100}")
    print("SUMMARY")
    print(f"{'=' * 100}")
    print(f"File 1 ({file1.name}):")
    print(f"  - Total relations: {doc_total}")
    print(f"  - Violations: {len(doc_violations)}")
    print(f"\nFile 2 ({file2.name}):")
    print(f"  - Total relations: {sent_total}")
    print(f"  - Violations: {len(sent_violations)}")
    print(f"\nTotal violations across both files: {len(doc_violations) + len(sent_violations)}")

    if len(doc_violations) + len(sent_violations) == 0:
        print("\n✓ All checks passed! No violations found.")
    else:
        print("\n✗ Violations detected. Please review the output above.")


if __name__ == "__main__":
    main()
