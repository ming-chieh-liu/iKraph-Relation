import json
from collections import OrderedDict
import pandas as pd 

def lookup_entities(entities:list[dict]) -> dict[dict]:
    entities = {entity["id"]: entity for entity in entities}
    return entities

def lookup_relations(relations:list):
    lookup = {}
    if not relations:
        return {}
    else:
        for relation in relations:
            key = (relation["ent1"]["id"], relation["ent2"]["id"])
            lookup[key] = relation["reltyp"]

    return lookup


def normalize_entity_pair(ent1_id, ent2_id, direction):
    """
    Normalize entity pair by sorting IDs and adjusting direction.
    If IDs are swapped, reverse direction ('12' <-> '21').

    Returns: (min_id, max_id, adjusted_direction)
    """
    if ent1_id <= ent2_id:
        return (ent1_id, ent2_id, direction)
    else:
        # IDs are swapped, so reverse direction if it's '12' or '21'
        if direction == '12':
            adjusted_direction = '21'
        elif direction == '21':
            adjusted_direction = '12'
        else:
            adjusted_direction = direction
        return (ent2_id, ent1_id, adjusted_direction)


def sample_documents(data, sample_size):
    """
    Sample documents with the following rules:
    1. Rank papers by relation count (higher is better)
    2. Track sampled entity pairs to avoid duplicates
    3. Skip entire document if any entity pair was already sampled
    4. Continue until exactly N documents with unique pairs are collected

    Returns: List of sampled documents
    """
    # Rank documents by relation count (descending)
    if sample_size >= len(data):
        return data
    ranked_docs = sorted(data, key=lambda doc: len(doc.get("relation", [])), reverse=True)

    sampled_docs = []
    sampled_pairs = set()

    for doc in ranked_docs:
        if len(sampled_docs) >= sample_size:
            break

        # Extract all entity pairs from this document
        doc_pairs = set()
        # gene_entities, disease_entities = lookup_entities(doc["entities"])
        relations = doc.get("relation", [])

        for relation in relations:
            ent1 = relation["ent1"]
            ent2 = relation["ent2"]
            
            ent1_id = ent1["id"]
            ent2_id = ent2["id"]
            # ent1_type = ent1["type"]
            # ent2_type = ent2["type"]
            direction = relation.get("direction", "NA")

            pair = normalize_entity_pair(ent1_id, ent2_id, direction)

            doc_pairs.add(pair)

        # Check if any pair in this document was already sampled
        if doc_pairs.intersection(sampled_pairs):
            # Skip this document - it has duplicate pairs
            continue

        # No duplicates - add this document and record its pairs
        sampled_docs.append(doc)
        sampled_pairs.update(doc_pairs)

    return sampled_docs


def main():
    # Parse command-line arguments
    
    input_file = "../litcoin_600/All.json"
    train_file = "../original_data/abstracts_train.csv"
    output_file = "./litcoin_200_doc_for_annotation.json"

    # Load input data
    print(f"Loading data from {input_file}...")
    with open(input_file, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} documents")

    # Load train data
    train_pmids = set()
    train_data = pd.read_csv(train_file, sep='\t')
    for pmid in train_data["abstract_id"].tolist():
        train_pmids.add(str(pmid))
    
    # Keep only documents not in train set
    data = [doc for doc in data if doc["document_id"] not in train_pmids]
    unique_pmids = {doc["document_id"] for doc in data}
    print(f"{len(data)} documents remain after excluding training PMIDs")
    print(f"{len(unique_pmids)} unique PMIDs in remaining documents")

    # Convert sampled documents
    new_data = []
    for doc_id, doc in enumerate(data, start=1):
        new_doc = OrderedDict()
        new_doc["Name"] = f"NER_batch1_{doc_id}"
        new_doc["PMID"] = doc["document_id"]
        new_doc["text"] = doc["title"] + " " + doc["abstract"]

        entities_lookup = lookup_entities(doc["entities"])

        # gene_entities, disease_entities = lookup_entities(doc["entities"])
        # relation_lookup = lookup_relations(doc.get("relation", []))

        relation_annotations = []
        seq_id = 1

        for relation in doc.get("relation", []):
            ent1 = relation["ent1"]
            ent2 = relation["ent2"]

            relation_annotation = OrderedDict()
            relation_annotation["entity_1"] = ent1["Name"][0]
            relation_annotation["type_1"] = ent1["type"]
            relation_annotation["id_1"] = ent1["id"]
            relation_annotation["entity_1_spans_list"] = entities_lookup[ent1["id"]]["span"]
            relation_annotation["entity_2"] = ent2["Name"][0]
            relation_annotation["type_2"] = ent2["type"]
            relation_annotation["id_2"] = ent2["id"]
            relation_annotation["entity_2_spans_list"] = entities_lookup[ent2["id"]]["span"]
            relation_annotation["seq_id"] = seq_id
            seq_id += 1
            # relation_annotation["direction"] = relation["direction"]
            relation_annotation["relation_types_Claude"] = relation["reltyp"]
            relation_annotation["relation_types_GPT40"] = ""
            relation_annotation["relation_types_GPT40-mini"] = ""
            relation_annotation["matching score (out of 100%)"] = ""

            relation_annotations.append(relation_annotation)

        new_doc["Relation_Annotation"] = relation_annotations

        new_data.append(new_doc)

    # Write output
    print(f"Writing output to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(new_data, f, indent=4)
    print(f"Done! Converted {len(new_data)} documents")

if __name__ == "__main__":
    main()