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


def get_entity_poses(start, end, entity_infos, position):
    """
    entity_infos: list of dicts with position, offset_start, offset_finish, type
    Returns positions of entities that fall within [start, end) sentence boundary
    """
    ret = []
    for entity_info in entity_infos:
        if entity_info["position"] != position:
            continue
        offset_start = entity_info["offset_start"]
        offset_finish = entity_info["offset_finish"]
        entity_type = entity_info["type"]
        if start <= offset_start and offset_finish <= end:
            ret.append([offset_start-start, offset_finish-start, entity_type])
        elif start <= offset_start and offset_start < end and end < offset_finish:
            raise ValueError("Entity spans sentence boundary")
    return ret


def convert_all_json_to_old(doc):
    """
    Convert All.json format to old format for processing
    Flattens document with multiple relations into separate relation objects
    """
    relations = []
    pmid = doc["document_id"]
    title = doc["title"]
    abstract = doc["abstract"]
    combined_text = title + " " + abstract

    for rel_idx, rel in enumerate(doc["relation"]):
        # Extract entity information from relation
        ent1 = rel["ent1"]
        ent2 = rel["ent2"]

        # Create entity_a_info format from ent1 spans
        entity_a_info = []
        for span in ent1["span"]:
            entity_a_info.append([span[0], span[1], ent1["type"]])

        # Create entity_b_info format from ent2 spans
        entity_b_info = []
        for span in ent2["span"]:
            entity_b_info.append([span[0], span[1], ent2["type"]])

        relation_type = rel["reltyp"]
        novelty = rel["novelty"]
        novel = True if novelty == 1 else False
        relation_indicator = False if relation_type == "NOT" else True

        # Create relation object in old format
        relation = {
            "abstract_id": int(pmid),
            "relation_id": f"{relation_indicator}.{pmid}.{rel_idx}",
            "text": combined_text,
            "entity_a_id": ent1["id"],
            "entity_b_id": ent2["id"],
            "type": relation_type,
            "novel": novel,
            "entity_a_info": entity_a_info,
            "entity_b_info": entity_b_info,
            "overlapping": "Unknown"
        }
        relations.append(relation)

    return relations


def sentence_split(data):
    """
    Split text into sentences and track entity positions
    Works with data format from gen_data_all.py (title/abstract separated, position field)
    """
    title = data["title"]
    abstract = data["abstract"]
    entity_a = copy.deepcopy(data["entity_a_info"])
    entity_b = copy.deepcopy(data["entity_b_info"])

    # Tokenize abstract into sentences
    tokenized_abstract = sentence_tokenize(abstract)
    abstract_sents = [sent for sent, _, _ in tokenized_abstract]

    # Get entity positions: first for title, then for each abstract sentence
    entity_a_poses = [get_entity_poses(0, len(title), entity_a, position="title")]
    entity_b_poses = [get_entity_poses(0, len(title), entity_b, position="title")]

    for _, start, end in tokenized_abstract:
        entity_a_poses.append(get_entity_poses(start, end, entity_a, position="abstract"))
        entity_b_poses.append(get_entity_poses(start, end, entity_b, position="abstract"))

    assert len([title] + abstract_sents) == len(entity_a_poses)

    sents = []
    for text, entity_a_poses_in_sent, entity_b_poses_in_sent in zip([title] + abstract_sents, entity_a_poses, entity_b_poses):
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
        "novel": (data["novel"]=="Novel") if isinstance(data["novel"], str) else data["novel"],
        "overlapping": data["overlapping"],
        "sents": sents
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multi-sentence data from gen_data_all.py output.")
    parser.add_argument("--data_path", action="store", dest="data_path", help="Directory containing processed/processed_all.json file.")
    args = parser.parse_args()

    # Read processed_all.json file from gen_data_all.py
    processed_input_file = os.path.join(args.data_path, "processed", "processed_all.json")
    print(f"Reading {processed_input_file}...")
    all_relations = json.load(open(processed_input_file, "r"))

    print(f"Total relations: {len(all_relations)}")

    # Process all relations with sentence splitting
    processed = list(map(sentence_split, all_relations))

    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.data_path, "processed")
    os.makedirs(output_dir, exist_ok=True)

    # Save sentence-split processed data
    output_file = os.path.join(output_dir, "sentence_split_all.json")
    json.dump(processed, open(output_file, "w"), indent=4)
    print(f"Saved sentence-split data to {output_file}")
    print(f"Total relations processed: {len(processed)}")
