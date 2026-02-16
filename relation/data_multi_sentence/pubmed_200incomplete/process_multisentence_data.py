import re
import os
import json
import copy
import argparse

import nltk
from nltk.corpus import words


# Map entity types to consistent format
TYPE_MAPPING = {
    "Disease": "DiseaseOrPhenotypicFeature",
    "Gene": "GeneOrGeneProduct",
    "Chemical": "ChemicalEntity",
}


def post_nltk_processing(sentence_list, VERBOSE=False):
    """
    Post processing.
    Sentences_list: a list of [sent, start, end] that [start, end] is the span of the sentence. All the sentences must be in the same paragraph.
    Returns: a list of [sent, start, end] after post processing.
    """
    ret = []
    prev_line = ''
    prev_span1 = 0
    prev_span2 = 0
    MERGE = 0
    for sent, span1, span2 in sentence_list:
        if MERGE == 1:
            ret.append([prev_line + ' '*(span1-prev_span2) + sent, prev_span1, span2])
            if VERBOSE:
                print('Fixed an error by concatenating two sentences below:')
                print('SENTENCE 1: ', prev_line+'\t'+str(prev_span1)+'\t'+str(prev_span2))
                print('SENTENCE 2: ', sent+'\t'+str(span1)+'\t'+str(span2))
                print('MERGED: ', prev_line + ' '*(span1-prev_span2) + sent + '\t' + str(prev_span1) + '\t' + str(span2))
            MERGE = 0
            continue

        p = re.compile(r"[\.\?]\s([A-Z][a-z]*)\s")
        result = p.search(sent)
        if result != None:
            tmpW = result.group(1)
            if tmpW != '' and tmpW.lower() in words.words():
                # we should split this case
                pos = re.search(r"[\.\?]\s([A-Z][a-z]*)\s", sent)
                pos = pos.span()[0]
                sent1 = sent[0:pos+1]
                sent2 = sent[pos+1:]
                ret.append([sent1, span1, span1 + pos + 1])
                tmp = sent2.lstrip()
                diff = len(sent2) - len(tmp)
                ret.append([tmp, span1+pos+1+diff, span2])
                if VERBOSE == 1:
                    print('Fixed an error by splitting a sentence as below:')
                    print(sent+'\t'+str(span1)+'\t'+str(span2))
                    print('Split sentence 1: ', sent1+ '\t' + str(span1) + '\t' + str(span1 + pos))
                    print('Split sentence 2: ', sent2+ '\t' + str(span1 + pos + diff) +'\t' + str(span2))
            else:
                ret.append([sent, span1, span2])
        else:
            if sent.endswith(')'):
                MERGE = 1
                # the newline character needs to be removed
                prev_line = sent
                prev_span1 = span1
                prev_span2 = span2
            else:
                ret.append([sent, span1, span2])
    return ret


def sentence_tokenize(paragraph):
    """
    Given a paragraph which can be title + ' ' + abstract, sentence tokenize it into:
    Returns: a list of sentences [sent, start, end], such that paragraph[start:end] == sent.
    """
    extra_abbreviations = ['e.g', 'i.e', 'i.m', 'a.u', 'p.o', 'i.v', 'i.p', 'vivo', 'p.o', 'i.p', 'Vmax', 'i.c.v', ')(', 'E.C', 'sp', 'al']
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentence_tokenizer._params.abbrev_types.update(extra_abbreviations)
    sents = sentence_tokenizer.tokenize(paragraph)

    sentence_list = []
    for idx, sent in enumerate(sents):
        if idx == 0:
            start = paragraph.find(sent)
        else:
            start = 1+paragraph[1:].find(sent)
        end = start+len(sent)
        sentence_list.append([sent, start, end])

    return post_nltk_processing(sentence_list)


def flatten_list(list_of_list):
    ret = []
    for sublist in list_of_list:
        for elem in sublist:
            ret.append(elem)
    return ret


def get_entity_poses(start, end, entity_infos):
    """
    entity_infos: list of [offset_start, offset_finish, entity_type]
    Returns positions of entities that fall within [start, end) sentence boundary
    """
    ret = []
    for offset_start, offset_finish, entity_type in entity_infos:
        if start <= offset_start and offset_finish <= end:
            ret.append([offset_start-start, offset_finish-start, entity_type])
        elif start <= offset_start and offset_start < end and end < offset_finish:
            raise ValueError("Entity spans sentence boundary")
    return ret


def convert_new_format_to_old(doc):
    """
    Convert pubmed_200incomplete format to old format for processing.
    Flattens document with multiple relations into separate relation objects.

    Key differences from litcoin_80updated DS format:
    - Uses relation_types_Claude instead of gold_standard.relation_type
    - Uses direction field directly instead of gold_standard.direction
    - Entity IDs already present (id_1, id_2)
    """
    relations = []
    for rel in doc["Relation_Annotation"]:
        # Map entity types to consistent format
        type_1 = TYPE_MAPPING.get(rel["type_1"], rel["type_1"])
        type_2 = TYPE_MAPPING.get(rel["type_2"], rel["type_2"])

        # Create entity_a_info format from entity_1_spans_list
        entity_a_info = []
        for span in rel["entity_1_spans_list"]:
            entity_a_info.append([span[0], span[1], type_1])

        # Create entity_b_info format from entity_2_spans_list
        entity_b_info = []
        for span in rel["entity_2_spans_list"]:
            entity_b_info.append([span[0], span[1], type_2])

        # Use relation_types_Claude for relation type
        relation_type = rel["relation_types_Claude"]
        relation_indicator = False if relation_type == "NOT" else True

        # Create relation object in old format
        relation = {
            "abstract_id": doc["PMID"],
            "relation_id": f"{relation_indicator}.{doc['PMID']}.{rel['seq_id']}",
            "text": doc["text"],
            "entity_a_id": rel["id_1"],
            "entity_b_id": rel["id_2"],
            "type": relation_type,
            "novel": False,  # Not specified in new format
            "entity_a_info": entity_a_info,
            "entity_b_info": entity_b_info,
            "overlapping": "Unknown"  # Not specified in new format
        }
        relations.append(relation)

    return relations


def sentence_split(data):
    """
    Split text into sentences and track entity positions
    Adapted for new format with single 'text' field (no title/abstract separation)
    """
    text = data["text"]
    entity_a = copy.deepcopy(data["entity_a_info"])
    entity_b = copy.deepcopy(data["entity_b_info"])

    # Tokenize the text into sentences
    tokenized_text = sentence_tokenize(text)
    text_sents = [sent for sent, _, _ in tokenized_text]

    # Get entity positions for each sentence
    entity_a_poses = []
    entity_b_poses = []
    for _, start, end in tokenized_text:
        entity_a_poses.append(get_entity_poses(start, end, entity_a))
        entity_b_poses.append(get_entity_poses(start, end, entity_b))

    assert len(text_sents) == len(entity_a_poses)

    sents = []
    for text, entity_a_poses_in_sent, entity_b_poses_in_sent in zip(text_sents, entity_a_poses, entity_b_poses):
        sents.append({
            "text": text,
            "entity_a": entity_a_poses_in_sent,
            "entity_b": entity_b_poses_in_sent,
        })

    return {
        "abstract_id": data["abstract_id"],
        "relation_id": data["relation_id"],
        "entity_a_id": data["entity_a_id"],
        "entity_b_id": data["entity_b_id"],
        "type": data["type"],
        "novel": data["novel"],
        "overlapping": data["overlapping"],
        "sents": sents
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-sentence data for pubmed_200incomplete format.")
    parser.add_argument("--data_path", action="store", dest="data_path", help="Directory containing combined_already_labeled_document_level.json.")
    args = parser.parse_args()

    # Read from single JSON file (not DS_with_id directory)
    input_file = os.path.join(args.data_path, "combined_already_labeled_document_level.json")
    all_docs = json.load(open(input_file, "r"))

    print(f"Total documents loaded: {len(all_docs)}")

    # Convert new format to old format and flatten relations
    all_relations = []
    for doc in all_docs:
        relations = convert_new_format_to_old(doc)
        all_relations.extend(relations)

    print(f"Total relations: {len(all_relations)}")

    # Process all relations
    processed = list(map(sentence_split, all_relations))

    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.data_path, "processed")
    os.makedirs(output_dir, exist_ok=True)

    # Save processed data
    output_file = os.path.join(output_dir, "processed_all.json")
    json.dump(processed, open(output_file, "w"), indent=4)
    print(f"Saved processed data to {output_file}")
