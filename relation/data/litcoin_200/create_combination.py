import json

NO_REL_TYPES = ["OrganismTaxon", "CellLine"]
NO_REL_PAIRS = [["DiseaseOrPhenotypicFeature", "DiseaseOrPhenotypicFeature"],
                ["SequenceVariant", "GeneOrGeneProduct"], ["GeneOrGeneProduct", "SequenceVariant"],
                ["SequenceVariant", "SequenceVariant"]
                ]

def remove_invalid_relations(relations):
    valid_relations = []
    for rel in relations:
        type1 = rel['ent1']['type']
        type2 = rel['ent2']['type']
        if type1 in NO_REL_TYPES or type2 in NO_REL_TYPES:
            continue
        if [type1, type2] in NO_REL_PAIRS or [type2, type1] in NO_REL_PAIRS:
            continue
        valid_relations.append(rel)
    return valid_relations

def main():
    # Parse command-line arguments
    input_file = "../litcoin_600/All.json"
    output_file = "./All_with_empty_relations.json"

    with open(input_file, 'r') as f:
        data = json.load(f)


    for entry in data:
        entry['relation'] = remove_invalid_relations(entry['relation'])
        local_entities = {}
        new_relations = []

        for entity in entry['entities']:
            if entity["type"] in NO_REL_TYPES:
                continue 
            local_entities[entity['id']] = entity
        
        existing_relations = set()
        for rel in entry['relation']:
            key = sorted([rel["ent1"]["id"], rel["ent2"]["id"]])
            existing_relations.add(tuple(key))

        entity_ids = sorted(list(local_entities.keys()))
        n = len(entity_ids)
        for i in range(n):
            id_1 = entity_ids[i]
            type_1 = local_entities[id_1]["type"]
            for j in range(i + 1, n):
                id_2 = entity_ids[j]
                type_2 = local_entities[id_2]["type"]

                if id_1 == id_2:
                    continue
                    
                if (id_1, id_2) in existing_relations or (id_2, id_1) in existing_relations:
                    continue

                if [type_1, type_2] in NO_REL_PAIRS or [type_2, type_1] in NO_REL_PAIRS:
                    continue 
                
                new_relation = {}
                ent1 = local_entities[entity_ids[i]]
                ent2 = local_entities[entity_ids[j]]

                key = tuple(sorted([ent1['id'], ent2['id']]))
                new_relation = {
                    "ent1": ent1,
                    "ent2": ent2,
                    "reltyp": "NOT",
                    "novelty": None
                }
                new_relations.append(new_relation)
                existing_relations.add(key)

        entry['relation'].extend(new_relations)


    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()