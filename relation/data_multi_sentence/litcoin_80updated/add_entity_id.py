import json 
import pandas as pd 
from pathlib import Path 

def main():
    with open("./filtered_All_for_annotation.json", "r") as f:
        data = json.load(f)

    type_mapping = {
        "Disease": "DiseaseOrPhenotypicFeature",
    }

    lookup = {}
    for entry in data:
        lookup[entry["Name"]] = entry
    
    directory = Path("./DS")
    new_directory = Path("./DS_with_id")
    new_directory.mkdir(exist_ok=True)
    for file in directory.iterdir():
        with open(file, "r") as f:
            annotated_data = json.load(f) 

        for entry in annotated_data:
            name = entry["Name"]

            if name not in lookup: 
                print(f"Warning: {name} not found in lookup.")
                continue

            original_entry = lookup[name]

            assert entry["PMID"] == original_entry["PMID"]
            
            if len(entry["Relation_Annotation"]) != len(original_entry["Relation_Annotation"]):
                print(f"Warning: Relation_Annotation length mismatch for {name}.")
                print(f"  New file has {len(entry['Relation_Annotation'])} relations, original has {len(original_entry['Relation_Annotation'])} relations.")

                # Create lookup dictionary for original relations using entity names, types, and span lists
                def make_hashable(spans_list):
                    return tuple(tuple(span) if isinstance(span, list) else span for span in spans_list)

                original_lookup = {}
                for orig_rel in original_entry["Relation_Annotation"]:
                    key = (
                        orig_rel["entity_1"],
                        orig_rel["entity_2"],
                        type_mapping.get(orig_rel["type_1"], orig_rel["type_1"]),
                        type_mapping.get(orig_rel["type_2"], orig_rel["type_2"]),
                        make_hashable(orig_rel["entity_1_spans_list"]),
                        make_hashable(orig_rel["entity_2_spans_list"])
                    )
                    original_lookup[key] = orig_rel

                # Match and add IDs
                matched_count = 0
                for relation in entry["Relation_Annotation"]:
                    key = (
                        relation["entity_1"],
                        relation["entity_2"],
                        type_mapping.get(relation["type_1"], relation["type_1"]),
                        type_mapping.get(relation["type_2"], relation["type_2"]),
                        make_hashable(relation["entity_1_spans_list"]),
                        make_hashable(relation["entity_2_spans_list"])
                    )

                    if key in original_lookup:
                        relation["id_1"] = original_lookup[key]["id_1"]
                        relation["id_2"] = original_lookup[key]["id_2"]
                        matched_count += 1
                    else:
                        print(f"  No match found for relation: ({relation['entity_1']}, {relation['entity_2']})")

                print(f"  Matched {matched_count}/{len(entry['Relation_Annotation'])} relations.")

            else:
                # Length matches, use original index-based matching
                for i, relation in enumerate(entry["Relation_Annotation"]):
                    assert relation["entity_1"] == original_entry["Relation_Annotation"][i]["entity_1"]
                    assert relation["entity_2"] == original_entry["Relation_Annotation"][i]["entity_2"]
                    assert type_mapping.get(relation["type_1"], relation["type_1"]) == original_entry["Relation_Annotation"][i]["type_1"]
                    assert type_mapping.get(relation["type_2"], relation["type_2"]) == original_entry["Relation_Annotation"][i]["type_2"]
                    assert relation["entity_1_spans_list"] == original_entry["Relation_Annotation"][i]["entity_1_spans_list"]
                    assert relation["entity_2_spans_list"] == original_entry["Relation_Annotation"][i]["entity_2_spans_list"]
                    relation["id_1"] = original_entry["Relation_Annotation"][i]["id_1"]
                    relation["id_2"] = original_entry["Relation_Annotation"][i]["id_2"]

        output_file = new_directory / file.name
        with open(output_file, "w") as f:
            json.dump(annotated_data, f, indent=4)

if __name__ == "__main__":
    main()