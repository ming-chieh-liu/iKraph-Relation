#!/usr/bin/env python3
"""
Script to update annotations from new file to old file.

Matches entries based on pmid, text, entity IDs, and entity spans,
then tracks annotation changes.
"""

import json
import re
from typing import Dict, List, Tuple, Any
from collections import defaultdict


def extract_batch_number(name_or_relation_id: str) -> int:
    """
    Extract batch number from Name field or relation_id.

    Examples:
        - "NER_batch2_76_sent_9" -> 2
        - "True.11683992.6401.101404.2.1.combination0" -> 2 (5th component)

    Args:
        name_or_relation_id: The Name field or relation_id string

    Returns:
        Batch number, or -1 if not found
    """
    # Try Name field format first
    match = re.search(r'batch(\d+)', name_or_relation_id)
    if match:
        return int(match.group(1))

    # Try relation_id format (assuming batch is the 5th component)
    parts = name_or_relation_id.split('.')
    if len(parts) >= 5:
        try:
            return int(parts[4])
        except ValueError:
            pass

    return -1


def create_match_keys(entry: Dict[str, Any]) -> Tuple[Tuple, Tuple]:
    """
    Create match keys for both entity orderings.

    Args:
        entry: Dictionary containing entry data

    Returns:
        Two tuples: (key1, key2) where key2 has entities swapped
    """
    entity_a_span = tuple(entry['entity_a'][:2])  # (start, end)
    entity_b_span = tuple(entry['entity_b'][:2])  # (start, end)

    key1 = (
        entry['abstract_id'],
        entry['text'],
        entry['entity_a_id'],
        entry['entity_b_id'],
        entity_a_span,
        entity_b_span
    )

    key2 = (
        entry['abstract_id'],
        entry['text'],
        entry['entity_b_id'],
        entry['entity_a_id'],
        entity_b_span,
        entity_a_span
    )

    return key1, key2


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['data']


def main():
    # File paths
    old_file = '../litcoin_400_164/new_annotated_train_litcoin_400_164_pd.json'
    new_file = '../litcoin_80updated_pubmed_200/new_annotated_train_litcoin_80updated_pubmed_200_pd.json'

    print("Loading files...")
    old_data = load_json_data(old_file)
    new_data = load_json_data(new_file)

    print(f"Loaded {len(old_data)} entries from old file")
    print(f"Loaded {len(new_data)} entries from new file")

    # Step 1: Filter new file to keep only batch 1 samples
    print("\nFiltering new file for batch 1 samples...")
    batch1_new_data = []

    for entry in new_data:
        # Check if there's a Name field
        if 'Name' in entry:
            batch_num = extract_batch_number(entry['Name'])
        else:
            # Try using relation_id
            batch_num = extract_batch_number(entry['relation_id'])

        if batch_num == 1:
            batch1_new_data.append(entry)

    print(f"Found {len(batch1_new_data)} batch 1 samples in new file")

    # Step 2: Create index for old file entries
    print("\nIndexing old file entries...")
    old_index: Dict[Tuple, Dict[str, Any]] = {}

    for entry in old_data:
        key1, key2 = create_match_keys(entry)
        old_index[key1] = entry
        old_index[key2] = entry

    print(f"Indexed {len(old_index)} unique keys from old file")

    # Step 3: Match entries and track changes
    print("\nMatching entries and tracking changes...")

    matched_mappings = []  # List of (new_rel_id, old_rel_id, old_annot, new_annot)
    unmatched_new = []     # List of new relation_ids that couldn't be matched

    for new_entry in batch1_new_data:
        key1, key2 = create_match_keys(new_entry)

        # Try both key orderings
        if key1 in old_index:
            old_entry = old_index[key1]
        elif key2 in old_index:
            old_entry = old_index[key2]
        else:
            old_entry = None

        if old_entry:
            new_rel_id = new_entry['relation_id']
            old_rel_id = old_entry['relation_id']
            old_annotation = old_entry['annotated_type']
            new_annotation = new_entry['annotated_type']

            matched_mappings.append({
                'new_relation_id': new_rel_id,
                'old_relation_id': old_rel_id,
                'old_annotation': old_annotation,
                'new_annotation': new_annotation,
                'changed': old_annotation != new_annotation
            })
        else:
            unmatched_new.append(new_entry['relation_id'])

    # Step 4: Generate statistics and summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    total_matched = len(matched_mappings)
    total_unmatched = len(unmatched_new)

    # Count annotation changes
    changed_annotations = [m for m in matched_mappings if m['changed']]
    unchanged_annotations = [m for m in matched_mappings if not m['changed']]

    print(f"\nTotal batch 1 samples in new file: {len(batch1_new_data)}")
    print(f"Matched entries: {total_matched}")
    print(f"Unmatched entries (new): {total_unmatched}")
    print(f"\nAnnotation changes: {len(changed_annotations)}")
    print(f"Annotations unchanged: {len(unchanged_annotations)}")

    # Show annotation change details
    if changed_annotations:
        print("\n" + "-"*80)
        print("ANNOTATION CHANGES DETAILS:")
        print("-"*80)

        # Group by annotation type change
        change_types = defaultdict(list)
        for mapping in changed_annotations:
            change_key = (mapping['old_annotation'], mapping['new_annotation'])
            change_types[change_key].append(mapping)

        for (old_annot, new_annot), mappings in sorted(change_types.items()):
            print(f"\n'{old_annot}' -> '{new_annot}': {len(mappings)} changes")
            for i, m in enumerate(mappings[:5]):  # Show first 5 examples
                print(f"  {i+1}. {m['new_relation_id']}")
            if len(mappings) > 5:
                print(f"  ... and {len(mappings) - 5} more")

    # Show some unmatched examples
    if unmatched_new:
        print("\n" + "-"*80)
        print("UNMATCHED NEW ENTRIES (first 10):")
        print("-"*80)
        for rel_id in unmatched_new[:10]:
            print(f"  - {rel_id}")
        if len(unmatched_new) > 10:
            print(f"  ... and {len(unmatched_new) - 10} more")

    # Update old data with new annotations
    print("\nUpdating old file annotations...")
    updated_old_data = old_data.copy()

    # Create mapping from old_relation_id to new annotation
    update_map = {}
    for mapping in matched_mappings:
        if mapping['changed']:
            update_map[mapping['old_relation_id']] = mapping['new_annotation']

    # Apply updates
    for entry in updated_old_data:
        if entry['relation_id'] in update_map:
            entry['annotated_type'] = update_map[entry['relation_id']]

    # Add unmatched new entries to the data
    print(f"Adding {len(unmatched_new)} unmatched new entries to the data...")
    unmatched_entries = [entry for entry in batch1_new_data if entry['relation_id'] in unmatched_new]
    updated_old_data.extend(unmatched_entries)

    print(f"Total entries after update: {len(updated_old_data)} (original: {len(old_data)}, added: {len(unmatched_entries)})")

    # Save updated annotation file
    updated_file = './new_annotated_train_litcoin_400_164_80updated_pd.json'

    # Load original JSON structure to preserve schema
    with open(old_file, 'r') as f:
        original_json = json.load(f)

    original_json['data'] = updated_old_data

    with open(updated_file, 'w') as f:
        json.dump(original_json, f, indent=2)

    print(f"Updated annotation file saved to: {updated_file}")

    # Save detailed results to JSON
    output_file = './mapping_results.json'

    results = {
        'summary': {
            'total_batch1_samples': len(batch1_new_data),
            'matched_entries': total_matched,
            'unmatched_entries': total_unmatched,
            'changed_annotations': len(changed_annotations),
            'unchanged_annotations': len(unchanged_annotations)
        },
        'matched_mappings': matched_mappings,
        'unmatched_new_relation_ids': unmatched_new
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Detailed mapping results saved to: {output_file}")
    print("="*80)


if __name__ == '__main__':
    main()