import json 
import pandas as pd 

def main():
    with open("./Annotated_Tina_RD1_with_pmids.json") as f:
        tina_data = json.load(f)

    # tina_pmid = set([item["pmid"] for item in tina_data if item['pmid'] is not None])

    with open("../litcoin_600/All.json") as f: 
        all_data = json.load(f)

    all_ids = set([item['document_id'] for item in all_data])

    train_data = pd.read_csv("../original_data/abstracts_train.csv", sep='\t')
    train_ids = set(train_data['abstract_id'].astype(str).tolist())

    valid_ids = all_ids - train_ids

    print(f"Valid pmid to add extra data: {len(valid_ids)}")

    keys = {}
    new_data = []

    for entry in tina_data:
        pmid = entry['pmid']
        if pmid not in valid_ids:
            continue
        
        key = (pmid, entry["entity_a_id"], entry["entity_b_id"], entry["sentence_id"])

        if key not in keys:
            keys[key] = 0
        else:
            keys[key] += 1

        combination_id = keys[key]

        new_entry = {
            "abstract_id": int(entry["pmid"]),
            "relation_id": f"True.{entry['pmid']}.{entry['relation_id']}.sentence{entry["sentence_id"]}.combination{combination_id}",
            "entity_a_id": entry["entity_a_id"],
            "entity_b_id": entry["entity_b_id"],
            "text": entry["sentence"],
            "document_level_type": "",
            "document_level_novel": "",
            "annotated_type": entry["anno_type"],
            "annotated_relation_words": [],
            "entity_a": [entry["entity_a_span"][0], entry["entity_a_span"][1], entry["entity_a_type"]],
            "entity_b": [entry["entity_b_span"][0], entry["entity_b_span"][1], entry["entity_b_type"]],
            "duplicated_flag": False
        }

        new_data.append(new_entry)

    print(f"Number of new data entries added: {len(new_data)}")

    df = pd.DataFrame(new_data)
    output_file = "./new_annotated_train_litcoin_164_pd.json"
    df.to_json(output_file, orient='table', indent=4)



if __name__ == "__main__":
    main()