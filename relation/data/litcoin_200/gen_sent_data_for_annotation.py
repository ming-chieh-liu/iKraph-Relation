import json
from collections import OrderedDict
import argparse
import os
from SentenceTokenizer import LitcoinSentenceTokenizer
import pandas as pd 


def lookup_entities(entities: list[dict]) -> dict:
    """Create a lookup dictionary for entities by their ID."""
    return {entity["id"]: entity for entity in entities}


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
    if len(data) <= sample_size:
        return data
    # Rank documents by relation count (descending)
    ranked_docs = sorted(data, key=lambda doc: len(doc.get("relation", [])), reverse=True)

    sampled_docs = []
    sampled_pairs = set()

    for doc in ranked_docs:
        if len(sampled_docs) >= sample_size:
            break

        # Extract all entity pairs from this document
        doc_pairs = set()
        relations = doc.get("relation", [])

        for relation in relations:
            ent1 = relation["ent1"]
            ent2 = relation["ent2"]

            ent1_id = ent1["id"]
            ent2_id = ent2["id"]
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


def is_entity_in_sentence(entity_spans, sentence_start, sentence_end):
    """
    Check if any of the entity's spans fall within the sentence bounds.

    Args:
        entity_spans: list of [start, end] pairs for entity mentions
        sentence_start: starting offset of sentence
        sentence_end: ending offset of sentence

    Returns:
        bool: True if any span overlaps with the sentence
    """
    for span_start, span_end in entity_spans:
        # Check if this span overlaps with the sentence
        if span_start >= sentence_start and span_end <= sentence_end:
            return True
    return False


def recalculate_spans(entity_spans, sentence_start, sentence_end):
    """
    Recalculate entity spans relative to the sentence start.

    Args:
        entity_spans: list of [start, end] pairs for entity mentions
        sentence_start: starting offset of sentence
        sentence_end: ending offset of sentence

    Returns:
        list: new spans relative to sentence start (0-indexed for the sentence)
    """
    new_spans = []
    for span_start, span_end in entity_spans:
        # Only include spans that are within the sentence
        if span_start >= sentence_start and span_end <= sentence_end:
            new_span_start = span_start - sentence_start
            new_span_end = span_end - sentence_start
            new_spans.append([new_span_start, new_span_end])
    return new_spans


def get_first_entity_name_in_sentence(entity, sentence_start, sentence_end):
    """
    Get the entity name corresponding to the first mention within the sentence.

    Args:
        entity: entity dict with 'Name' and 'span' lists
        sentence_start: starting offset of sentence
        sentence_end: ending offset of sentence

    Returns:
        str: name of the first mention within the sentence, or first name if none found
    """
    # Find all spans within the sentence and their corresponding names
    mentions_in_sentence = []
    for i, (span_start, span_end) in enumerate(entity["span"]):
        if span_start >= sentence_start and span_end <= sentence_end:
            mentions_in_sentence.append((span_start, entity["Name"][i]))

    # Sort by offset and return the first one
    if mentions_in_sentence:
        mentions_in_sentence.sort(key=lambda x: x[0])
        return mentions_in_sentence[0][1]

    # Fallback to first name if no spans found in sentence
    return entity["Name"][0]


def main():
    # Parse command-line arguments
    input_file = "./All_with_empty_relations.json"
    train_file = "../original_data/abstracts_train.csv"
    prelabel_file = "../litcoin_164/Annotated_Tina_RD1_with_pmids.json"
    output_file = "./litcoin_200_sent_for_annotation.json"

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

    # load data that may be already pre-labeled
    with open(prelabel_file, "r") as f:
        prelabel_data = json.load(f)
    
    prelabel_lookup = {}
    for entry in prelabel_data:
        pmid = entry["pmid"]
        entity_a_id = entry["entity_a_id"]
        entity_b_id = entry["entity_b_id"]
        entity_a_span = tuple(entry['entity_a_span'])
        entity_b_span = tuple(entry["entity_b_span"])
        text = entry["text"]
        annotation = entry["anno_type"]
        key1 = (
            pmid,
            text,
            entity_a_id,
            entity_b_id,
            entity_a_span,
            entity_b_span
        )

        key2 = (
            pmid,
            text,
            entity_b_id,
            entity_a_id,
            entity_b_span,
            entity_a_span
        )

        prelabel_lookup[key1] = annotation
        prelabel_lookup[key2] = annotation
        
    # Initialize sentence tokenizer
    tokenizer = LitcoinSentenceTokenizer()

    # Process documents
    new_data = []
    item_counter = 1
    warning_count = 0

    for doc_id, doc in enumerate(data, start=1):
        pmid = doc["document_id"]
        text = doc["title"] + " " + doc["abstract"]

        relations = doc.get("relation", [])
        entities_lookup = lookup_entities(doc["entities"])

        # If no relations, skip
        if not relations:
            continue

        # Tokenize text into sentences
        sentences = tokenizer.sentence_tokenize_with_offsets(text)

        # Group relations by the sentence they belong to
        # sentence_relations: dict mapping sentence_idx -> list of relations
        sentence_relations = {}

        for relation in relations:
            ent1 = relation["ent1"]
            ent2 = relation["ent2"]

            # Get entity spans from lookup
            ent1_entity = entities_lookup[ent1["id"]]
            ent2_entity = entities_lookup[ent2["id"]]
            ent1_spans = ent1_entity["span"]
            ent2_spans = ent2_entity["span"]

            # Find which sentence(s) contain entity 1 and entity 2
            ent1_sentences = []
            ent2_sentences = []

            for sent_idx, sentence in enumerate(sentences):
                sent_start = sentence["offset_start"]
                sent_end = sentence["offset_end"]

                if is_entity_in_sentence(ent1_spans, sent_start, sent_end):
                    ent1_sentences.append(sent_idx)

                if is_entity_in_sentence(ent2_spans, sent_start, sent_end):
                    ent2_sentences.append(sent_idx)

            # Find common sentences where both entities appear
            common_sentences = set(ent1_sentences) & set(ent2_sentences)

            if not common_sentences:
                # Warning: entities appear in different sentences
                warning_count += 1
                print(f"WARNING: PMID={pmid}, Relation Type={relation['reltyp']}, "
                      f"Entity 1={ent1['Name'][0]} (type={ent1['type']}), "
                      f"Entity 2={ent2['Name'][0]} (type={ent2['type']}) - "
                      f"Entities appear in different sentences!")
                continue

            # Add this relation to all sentences where both entities appear
            for sent_idx in common_sentences:
                if sent_idx not in sentence_relations:
                    sentence_relations[sent_idx] = []
                sentence_relations[sent_idx].append(relation)

        # Create output items for each sentence with relations
        for sent_idx, sent_relations in sorted(sentence_relations.items()):
            sentence = sentences[sent_idx]
            sent_text = sentence["text"]
            sent_start = sentence["offset_start"]
            sent_end = sentence["offset_end"]

            # Create new document item
            new_doc = OrderedDict()
            new_doc["Name"] = f"NER_batch1_{doc_id}_sent_{sent_idx + 1}"
            new_doc["PMID"] = pmid
            new_doc["text"] = sent_text

            # Recalculate relation annotations with new spans
            new_relation_annotations = []
            for rel_idx, relation in enumerate(sent_relations, start=1):
                ent1 = relation["ent1"]
                ent2 = relation["ent2"]

                # Get full entity info from lookup
                ent1_entity = entities_lookup[ent1["id"]]
                ent2_entity = entities_lookup[ent2["id"]]

                new_relation = OrderedDict()
                # Get first entity name by offset in sentence
                new_relation["entity_1"] = get_first_entity_name_in_sentence(ent1_entity, sent_start, sent_end)
                new_relation["type_1"] = ent1["type"]
                new_relation["id_1"] = ent1["id"]

                # Recalculate entity 1 spans
                new_ent1_spans = recalculate_spans(
                    ent1_entity["span"],
                    sent_start,
                    sent_end
                )
                new_relation["entity_1_spans_list"] = new_ent1_spans

                # Get first entity name by offset in sentence
                new_relation["entity_2"] = get_first_entity_name_in_sentence(ent2_entity, sent_start, sent_end)
                new_relation["type_2"] = ent2["type"]
                new_relation["id_2"] = ent2["id"]

                # Recalculate entity 2 spans
                new_ent2_spans = recalculate_spans(
                    ent2_entity["span"],
                    sent_start,
                    sent_end
                )
                new_relation["entity_2_spans_list"] = new_ent2_spans

                new_relation["seq_id"] = rel_idx
                new_relation["direction"] = relation.get("direction", "NA")
                annotation = relation["reltyp"]

                found = False
                for ent1_span in new_ent1_spans:
                    if found:
                        break
                    for ent2_span in new_ent2_spans:
                        key = (
                            pmid,
                            sent_text,
                            new_relation["id_1"],
                            new_relation["id_2"],
                            tuple(ent1_span),  # single span as tuple
                            tuple(ent2_span)   # single span as tuple
                        )
                        if key in prelabel_lookup:
                            annotation = prelabel_lookup[key]
                            found = True
                            break
                new_relation["relation_types_Claude"] = annotation
                new_relation["relation_types_GPT40"] = ""
                new_relation["relation_types_GPT40-mini"] = ""
                new_relation["matching score (out of 100%)"] = ""

                new_relation_annotations.append(new_relation)

            new_doc["Relation_Annotation"] = new_relation_annotations
            new_data.append(new_doc)
            item_counter += 1

    # Write output
    print(f"Writing output to {output_file}...")
    with open(output_file, "w") as f:
        json.dump(new_data, f, indent=4)

    print(f"Done! Created {len(new_data)} sentence-level items from {len(data)} sampled documents")
    if warning_count > 0:
        print(f"Total warnings (cross-sentence relations): {warning_count}")


if __name__ == "__main__":
    main()
