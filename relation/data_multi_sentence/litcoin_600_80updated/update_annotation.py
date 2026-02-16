import json 
from pathlib import Path 
from typing import Any, Dict, Tuple

TYPE_MAPPING = {
    "Disease": "DiseaseOrPhenotypicFeature",
    "Gene": "GeneOrGeneProduct",
    "Chemical": "ChemicalEntity",
}

def create_match_keys(entry: Dict[str, Any]) -> Tuple[Tuple, Tuple]:
    """
    Create match keys for both entity orderings.

    Args:
        entry: Dictionary containing entry data

    Returns:
        Two tuples: (key1, key2) where key2 has entities swapped
    """
    key1 = (
        str(entry['abstract_id']),
        entry["text"],
        entry['entity_a_id'],
        entry['entity_b_id'],
    )

    key2 = (
        str(entry['abstract_id']),
        entry["text"],
        entry['entity_b_id'],
        entry['entity_a_id'],
    )

    return key1, key2

with open("../litcoin_600/multi_sentence_split_litcoin_600/train_all.json") as f:
    old_data = json.load(f)

with open("../litcoin_80updated/processed/multi_sentence_all.json") as f:
    new_data = json.load(f)
lookup = {}
idx_lookup = {}

old_pmids = set()
for entry in old_data: 
    relation_id = entry["relation_id"]
    _, abstract_id, idx = relation_id.split(".")
    idx = int(idx)
    if abstract_id not in idx_lookup:
        idx_lookup[abstract_id] = 0
    idx_lookup[abstract_id] = max(idx_lookup[abstract_id], idx)
    old_pmids.add(str(entry["abstract_id"]))
    key1, key2 = create_match_keys(entry)
    lookup[key1] = entry
    lookup[key2] = entry

directory = Path("./DS_with_id")
to_add = []
modified_count = 0
modified_details = []
total_searched = 0
found_in_lookup = 0

new_pmids = set()
for entry in new_data:
    new_pmids.add(entry["abstract_id"])
    pmid = entry["abstract_id"]
    text = entry["text"]

    total_searched += 1
    relation_type =  entry["type"]
    id_1 = entry["entity_a_id"]
    id_2 = entry["entity_b_id"]
    key = (pmid, text, id_1, id_2)
    if key not in lookup:
        to_add.append(entry)
    else:
        found_in_lookup += 1
        original_entry = lookup[key]
        if original_entry["type"] != relation_type:
            modified_count += 1
            modified_details.append({
                "pmid": pmid,
                "relation_id": original_entry["relation_id"],
                "old_type": original_entry["type"],
                "new_type": relation_type
            })
            original_entry["type"] = relation_type

print(f"\n=== Statistics ===")
print(f"Total original annotations: {len(old_data)}")
print(f"Total relations searched: {total_searched}")
print(f"Found in lookup (existing): {found_in_lookup}")
print(f"New annotations added: {len(to_add)}")
print(f"Existing annotations modified: {modified_count}")
print(f"Total annotations after update: {len(old_data) + len(to_add)}")

if modified_details:
    print(f"\n=== Modified Annotations ===")
    for detail in modified_details:
        print(f"  {detail['relation_id']}: {detail['old_type']} -> {detail['new_type']}")

old_data.extend(to_add)

f_path = Path("./multi_sentence_split_litcoin_600_80updated/train_all.json").resolve()
f_path.parent.mkdir(parents=True, exist_ok=True)

with open(f_path, "w") as f:
    json.dump(old_data, f, indent=4)

