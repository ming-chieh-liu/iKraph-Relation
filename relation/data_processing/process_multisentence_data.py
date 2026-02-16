import re
import os
import json
import copy
import argparse

import nltk
from nltk.corpus import words


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
    ret = [[] for _ in range(len(entity_infos))]
    for idx, entity_info in enumerate(entity_infos):
        if entity_info["position"] != position: continue
        offset_start, offset_finish = entity_info["offset_start"], entity_info["offset_finish"]
        entity_type = entity_info["type"]
        if start <= offset_start and offset_finish <= end:
            ret[idx].append([offset_start-start, offset_finish-start, entity_type])
        elif start <= offset_start and offset_start < end and end < offset_finish:
            raise ValueError
    return flatten_list(ret)


def sentence_split(data):
    title = data["title"]
    abstract = data["abstract"]
    entity_a = copy.deepcopy(data["entity_a_info"])
    entity_b = copy.deepcopy(data["entity_b_info"])
    # Returns: a list of sentences [sent, start, end], such that paragraph[start:end] == sent.
    tokenized_abstract = sentence_tokenize(abstract)
    abstract_sents = [sent for sent, _, _ in tokenized_abstract]

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
        "novel": (data["novel"]=="Novel"),
        "overlapping": data["overlapping"],
        "sents": sents
    }


if __name__ == "__main__":
    parser =  argparse.ArgumentParser(description="Generate multi-sentence data for LitCoin Phase 2.")
    parser.add_argument("--data_path", action="store", dest="data_path", help="Directory of processed LitCoin phase 2 data.")
    args = parser.parse_args()

    for sub_file in ["train", "val", "test"]:
        this_data = json.load(open(os.path.join(args.data_path, "split", f"{sub_file}.json"), "r"))
        processed = list(map(sentence_split, this_data))
        json.dump(processed, open(os.path.join(args.data_path, "split", f"processed_{sub_file}.json"), "w"), indent=4)
