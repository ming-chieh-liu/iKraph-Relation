import json 
import pandas as pd 
from pathlib import Path 

def main():
    with open("./combined_for_annotation_sentence_level.json", "r") as f:
        data = json.load(f)

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
                continue

            for i, relation in enumerate(entry["Relation_Annotation"]):
                assert relation["entity_1"] == original_entry["Relation_Annotation"][i]["entity_1"]
                assert relation["entity_2"] == original_entry["Relation_Annotation"][i]["entity_2"]
                assert relation["type_1"] == original_entry["Relation_Annotation"][i]["type_1"]
                assert relation["type_2"] == original_entry["Relation_Annotation"][i]["type_2"]
                assert relation["entity_1_spans_list"] == original_entry["Relation_Annotation"][i]["entity_1_spans_list"]
                assert relation["entity_2_spans_list"] == original_entry["Relation_Annotation"][i]["entity_2_spans_list"]
                relation["id_1"] = original_entry["Relation_Annotation"][i]["id_1"]
                relation["id_2"] = original_entry["Relation_Annotation"][i]["id_2"]

        output_file = new_directory / file.name
        with open(output_file, "w") as f:
            json.dump(annotated_data, f, indent=4)

if __name__ == "__main__":
    main()