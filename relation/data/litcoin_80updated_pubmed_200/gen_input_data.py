import json 
import pandas as pd 
from pathlib import Path 
import itertools 

def main():
    keys = {}
    new_data = []
    directory = Path("./DS_with_id")
    for file in directory.iterdir():
        with open(file, "r") as f:
            data = json.load(f) 

        for entry in data:
            pmid = entry['PMID']

            name_split = entry["Name"].split('_')
            batch_id = int(name_split[1].replace('batch', ''))
            sent_id = int(name_split[-1])

            for relation in entry["Relation_Annotation"]:
            
                key = (pmid, relation["id_1"], relation["id_2"], batch_id, sent_id)
                relation_id = ".".join([str(k) for k in key])

                entity_1_spans = relation["entity_1_spans_list"]
                entity_2_spans = relation["entity_2_spans_list"]

                type_1 = relation["type_1"] 
                if type_1 == "Gene":
                    type_1 = "GeneOrGeneProduct"
                elif type_1 == "Chemical":
                    type_1 = "ChemicalSubstance"
                elif type_1 == "Disease":
                    type_1 = "DiseaseOrPhenotypicFeature"
                
                type_2 = relation["type_2"]
                if type_2 == "Gene":
                    type_2 = "GeneOrGeneProduct"
                elif type_2 == "Chemical":
                    type_2 = "ChemicalSubstance"
                elif type_2 == "Disease":
                    type_2 = "DiseaseOrPhenotypicFeature"

                for span_1, span_2 in itertools.product(entity_1_spans, entity_2_spans):

                    if key not in keys:
                        keys[key] = 0
                    else:
                        keys[key] += 1

                    combination_id = keys[key]

                    new_entry = {
                        "abstract_id": int(entry["PMID"]),
                        "relation_id": f"True.{relation_id}.combination{combination_id}",
                        "entity_a_id": relation["id_1"],
                        "entity_b_id": relation["id_2"],
                        "text": entry["text"],
                        "document_level_type": "",
                        "document_level_novel": "",
                        "annotated_type": relation["gold_standard"]["relation_type"],
                        "annotated_relation_words": [],
                        "entity_a": [span_1[0], span_1[1], type_1],
                        "entity_b": [span_2[0], span_2[1], type_2],
                        "duplicated_flag": False
                    }

                    new_data.append(new_entry)

    print(f"Number of new data entries added: {len(new_data)}")

    df = pd.DataFrame(new_data)
    output_file = "./new_annotated_train_litcoin_80updated_pubmed_200_pd.json"
    df.to_json(output_file, orient='table', indent=4)



if __name__ == "__main__":
    main()