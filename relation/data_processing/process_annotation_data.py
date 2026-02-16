import re
import os
import json
import copy
import nltk
import argparse
import itertools
from collections import defaultdict

import pandas
from nltk.corpus import words

TRUE_TYPES = ["Association", "Positive_Correlation", "Negative_Correlation", "Bind", "Cotreatment", "Comparison", "Drug_Interaction", "Conversion"]

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


def process_annotated_type(doc_type, true_type, true_relation):
    if true_type != "":
        if true_type not in TRUE_TYPES:
            raise ValueError(f"Annotated true relation: {true_type} is not a valid interaction type!")
        return true_type
    if true_relation not in ["T", "F"]:
        raise ValueError(f"True relation must be T or F, but received {true_relation}!")
    if true_relation == "T":
        return doc_type
    else:
        return "NOT"


def process_relation_words(relation_word1, relation_word2):
    if relation_word1 == "" and relation_word2 == "":
        return []
    elif relation_word1 == "" and relation_word2 != "":
        raise ValueError(f"Relation word 1 is empty, but relation word 2 is {relation_word2}!")
    elif relation_word1 != "" and relation_word2 == "":
        return [relation_word1]
    else:
        return [relation_word1, relation_word2]


def find_entity_prune_sent(processed_sentence, start_token, end_token, other_start_token, other_end_token):
    start_tag_pos = processed_sentence.find(start_token)
    end_tag_pos = processed_sentence.find(end_token)
    entity_start_pos = start_tag_pos
    entity_end_pos = end_tag_pos - len(start_token)
    entity_name = processed_sentence[start_tag_pos+len(start_token): end_tag_pos]

    pruned_sentence = processed_sentence[0:start_tag_pos] + entity_name + processed_sentence[end_tag_pos+len(end_token):]

    if other_start_token not in entity_name:
        return entity_start_pos, entity_end_pos, entity_name, pruned_sentence

    # Nested case, like <ENT1><ENT2> .... </ENT2></ENT1>
    if entity_name.find(other_start_token) == 0 and entity_name.find(other_end_token) == len(entity_name) - len(other_end_token):
        entity_start_pos = start_tag_pos
        entity_end_pos = end_tag_pos - len(start_token) - len(other_start_token) - len(other_end_token)
        entity_name = processed_sentence[start_tag_pos+len(start_token)+len(other_start_token):end_tag_pos-len(other_end_token)]
    else:
        # No other type of nested entities in litcoin data
        raise ValueError(f"Nested Entities found in sentence {processed_sentence}.")

    return entity_start_pos, entity_end_pos, entity_name, pruned_sentence


def find_entity_positions(processed_sentence: str):
    first_ent1_start = processed_sentence.find("<ENT1>")
    first_ent2_start = processed_sentence.find("<ENT2>")
    next_to_find = None
    if first_ent1_start == -1 and first_ent2_start == -1:  # returns -1 if can't find
        return [], []
    elif first_ent1_start == -1 and first_ent2_start != -1:
        next_to_find = "entity2"
    elif first_ent1_start != -1 and first_ent2_start == -1:
        next_to_find = "entity1"
    elif first_ent1_start > first_ent2_start:
        next_to_find = "entity2"
    else:
        next_to_find = "entity1"
    if next_to_find == "entity1":
        ent_start, ent_end, ent_name, pruned_sentence = find_entity_prune_sent(processed_sentence, "<ENT1>", "</ENT1>", "<ENT2>", "</ENT2>")
        remaining_ent1, remaining_ent2 = find_entity_positions(pruned_sentence)
        return [[ent_start, ent_end, ent_name]] + remaining_ent1, remaining_ent2
    else:
        ent_start, ent_end, ent_name, pruned_sentence = find_entity_prune_sent(processed_sentence, "<ENT2>", "</ENT2>", "<ENT1>", "</ENT1>")
        remaining_ent1, remaining_ent2 = find_entity_positions(pruned_sentence)
        return remaining_ent1, [[ent_start, ent_end, ent_name]] + remaining_ent2


def process_interaction_positions(relation_words, num_entity_a, ent1_select, num_entity_b, ent2_select, anno_notes:str):
    if anno_notes.startswith("keep"):  # not dealing with this right now
        anno_notes = ""

    interacting_pairs = []
    if anno_notes != "":
        pairs = anno_notes.split(",")
        interacting_pairs = [[int(pair[0])-1, int(pair[1])-1] for pair in pairs]
    else:
        assert "," not in ent1_select or "," not in ent2_select
        interacting_pos1 = list(range(num_entity_a))
        interacting_pos2 = list(range(num_entity_b))
        if "," in ent1_select and ent2_select != "":
            interacting_pos1 = [int(pos)-1 for pos in ent1_select.split(",")]
            interacting_pos2 = [int(ent2_select) - 1]
        if "," in ent1_select and ent2_select == "":
            interacting_pos1 = [int(pos)-1 for pos in ent1_select.split(",")]
            interacting_pos2 = list(range(num_entity_b))
        if ent1_select != "" and "," in ent2_select:
            interacting_pos1 = [int(ent1_select) - 1]
            interacting_pos2 = [int(pos)-1 for pos in ent2_select.split(",")]
        if ent1_select == "" and "," in ent2_select:
            interacting_pos1 = list(range(num_entity_a))
            interacting_pos2 = [int(pos)-1 for pos in ent2_select.split(",")]
        for pos_a, pos_b in itertools.product(interacting_pos1, interacting_pos2):
            interacting_pairs.append([pos_a, pos_b])

    if len(relation_words) == 0:
        relation_words = [[""] for _ in range(len(interacting_pairs))]
    elif len(relation_words) == 1:  # only the first cell has inputs
        split_words = relation_words[0].split(";")
        if len(split_words) == 1:
            relation_words = [[split_words[0]] for _ in range(len(interacting_pairs))]
        elif len(split_words) == len(interacting_pairs):
            relation_words = [[word] for word in split_words]
        else:
            raise ValueError(f"Fix relation words {relation_words} in ent1_select {ent1_select} ent2_select {ent2_select} anno_notes {anno_notes}")
    else:
        split_words1 = relation_words[0].split(";")
        split_words2 = relation_words[1].split(";")
        assert len(split_words2) == 1
        if len(split_words1) == 1:
            relation_words = [[split_words1[0], split_words2[0]] for _ in range(len(interacting_pairs))]
        elif len(split_words1) == len(interacting_pairs):
            relation_words = [[word, split_words2[0]] for word in split_words1]
        else:
            raise ValueError(f"Fix relation words {relation_words} in ent1_select {ent1_select} ent2_select {ent2_select} anno_notes {anno_notes}")
    return [[pos_a, pos_b, relation_word] for (pos_a, pos_b), relation_word in zip(interacting_pairs, relation_words)]


def get_annotations(data_df):
    annotations = {}
    for _, entry in data_df.iterrows():

        abstract_id, relation_id = entry["abstract_id"].split(";"), entry["relation_id"].split(";")
        entity_a_id, entity_b_id = entry["entity_a_id"].split(";"), entry["entity_b_id"].split(";")
        document_level_type, document_level_novelty = entry["type"].split(";"), entry["novel"].split(";")

        entity_a_positions, entity_b_positions = find_entity_positions(entry["sentence"])
        num_entity_a, num_entity_b = len(entity_a_positions), len(entity_b_positions)
        relation_words = process_relation_words(entry["Relation word"], entry["Relation word 2"])
        relation = process_interaction_positions(relation_words, num_entity_a, entry["ent1 select"], num_entity_b, entry["ent2 select"], entry["Note"])

        for idx in range(len(abstract_id)):
            this_abstract_id, this_relation_id = int(abstract_id[idx]), int(relation_id[idx])
            this_entity_a_id, this_entity_b_id = entity_a_id[idx], entity_b_id[idx]
            this_document_level_type, this_document_level_novelty = document_level_type[idx], (document_level_novelty[idx] == "TRUE")
            # all types should be the same, but just put this here...
            annotated_type = process_annotated_type(this_document_level_type, entry["True type"], entry["True relation"])
            pruned_sentence = entry["sentence"].replace("<ENT1>", "").replace("<ENT2>", "").replace("</ENT1>", "").replace("</ENT2>", "")

            key1 = (this_abstract_id, this_entity_a_id, this_entity_b_id, pruned_sentence)
            key2 = (this_abstract_id, this_entity_b_id, this_entity_a_id, pruned_sentence)
            value = dict(
                annotated_type=annotated_type,
                relation=relation
            )
            assert key1 not in annotations
            assert key2 not in annotations
            annotations[key1] = value
            annotations[key2] = value


    return annotations


def flatten_list(list_of_list):
    ret = []
    for sublist in list_of_list:
        for elem in sublist:
            ret.append(elem)
    return ret


def get_entity(start, end, entity_infos, position):
    ret = [[] for _ in range(len(entity_infos))]
    for idx, entity_info in enumerate(entity_infos):
        if entity_info["position"] != position: continue
        offset_start, offset_finish = entity_info["offset_start"], entity_info["offset_finish"]
        this_entity_info = copy.deepcopy(entity_info)
        this_entity_info["offset_start"], this_entity_info["offset_finish"] = offset_start-start, offset_finish-start
        if start <= offset_start and offset_finish <= end:
            ret[idx].append(this_entity_info)
        elif start <= offset_start and offset_start < end and end < offset_finish:
            raise ValueError
    return flatten_list(ret)


def is_fully_overlapping(entity_a, entity_b):
    if len(entity_a) != len(entity_b): return False
    a_list = []
    for occurance in entity_a:
        a_list.append(occurance["offset_start"])
        a_list.append(occurance["offset_finish"])
    b_list = []
    for occurance in entity_b:
        b_list.append(occurance["offset_start"])
        b_list.append(occurance["offset_finish"])
    if any([elem_a != elem_b for elem_a, elem_b in zip(a_list, b_list)]):
        return False
    return True


def check_increasing_order(entity):
    a_list = []
    for occurance in entity:
        a_list.append(occurance["offset_start"])
        a_list.append(occurance["offset_finish"])
    b_list = sorted(a_list)
    if any([elem_a != elem_b for elem_a, elem_b in zip(a_list, b_list)]):
        raise ValueError


def get_data(DATA_DIR, annotations, copy_list_dict, is_test=False, out_file=None):
    sent_reps = {}
    sents = {}
    for data in json.load(open(DATA_DIR, "r")):
        has_relation = True
        if data["type"] == "NOT": has_relation = False
        if is_test: has_relation = False
        title = data["title"]
        abstract = data["abstract"]
        entity_a = copy.deepcopy(data["entity_a_info"])
        entity_b = copy.deepcopy(data["entity_b_info"])
        # Returns: a list of sentences [sent, start, end], such that paragraph[start:end] == sent.
        tokenized_abstract = sentence_tokenize(abstract)
        abstract_sents = [sent for sent, _, _ in tokenized_abstract]

        entity_as = [get_entity(0, len(title), entity_a, position="title")]
        entity_bs = [get_entity(0, len(title), entity_b, position="title")]
        for _, start, end in tokenized_abstract:
            entity_as.append(get_entity(start, end, entity_a, position="abstract"))
            entity_bs.append(get_entity(start, end, entity_b, position="abstract"))

        assert len([title] + abstract_sents) == len(entity_as)

        for sent_id, (text, entity_a, entity_b) in enumerate(zip([title] + abstract_sents, entity_as, entity_bs)):
            if len(entity_a) < 1 or len(entity_b) < 1: continue
            if is_fully_overlapping(entity_a, entity_b): continue
            check_increasing_order(entity_a)
            check_increasing_order(entity_b)


            if has_relation:
                copied_entity_a = copy_list_dict.get((data["abstract_id"], data["entity_a_id"]), [data["entity_a_id"]])
                copied_entity_b = copy_list_dict.get((data["abstract_id"], data["entity_b_id"]), [data["entity_b_id"]])

                results = []
                for this_entity_a, this_entity_b in itertools.product(copied_entity_a, copied_entity_b):
                    query = (
                        data["abstract_id"],
                        this_entity_a,
                        this_entity_b,
                        text
                    )
                    query_result = annotations.get(query, None)
                    if query_result is not None:
                        results.append(query_result)
                assert len(results) <= 1
                if len(results) == 1:
                    query_result = results[0]
                    annotated_type = query_result["annotated_type"]
                    relation = query_result["relation"]
                    relation = {(a[0], a[1]): a[2] for a in relation}
                else:
                    print(f"abstract_id {data['abstract_id']} entity_a_id {data['entity_a_id']} entity_b_id {data['entity_b_id']} sentence begins with {text[0:10]} can't be found.")
                    continue
            else:
                relation = {}
            if is_test:
                annotated_type = "TBD"
            for combination_idx, ((ent1_idx, ent1), (ent2_idx, ent2)) in enumerate(itertools.product(enumerate(entity_a), enumerate(entity_b))):
                this_relation_id = f"{data['relation_id']}.sentence{sent_id}.combination{combination_idx}"
                if ent1['offset_start'] == ent2['offset_start'] and ent1['offset_finish'] == ent2['offset_finish']:
                    continue
                relation_word = relation.get((ent1_idx, ent2_idx), None)
                this_annotated_type = "NOT" if relation_word is None else annotated_type
                relation_word = [] if relation_word is None else relation_word

                # sent_rep is a unique identifier for each instance in our sentence model. Use this to detect any duplicated sentences and process them.
                # if the annotated types agree, we set duplicated_flag=True to remove them in training.
                # For conflicts, set duplicated_flag=True and do:
                #   case a non-NOT vs. non-NOT: print out and manually examine;
                #   case b NOT vs. non-NOT: change the NOT case annotation to the non-NOT annotation.
                sent_rep = (text, ent1["offset_start"], ent1["offset_finish"], ent2["offset_start"], ent2["offset_finish"])
                duplicated_flag = False
                if sent_rep in sent_reps:
                    duplicated_flag = True
                    previous_id, (previous_annotated_type, previous_relation_word) = sent_reps[sent_rep]
                    if previous_annotated_type == this_annotated_type:
                        pass  # do nothing
                    else:
                        if previous_annotated_type == "NOT":  # NOT vs. non-NOT case b situation
                            sents[previous_id]["annotated_type"] = this_annotated_type
                            sents[previous_id]["annotated_relation_words"] = relation_word
                            sent_reps[sent_rep][1] = (this_annotated_type, relation_word)
                        elif this_annotated_type == "NOT":  # NOT vs. non-NOT case b situation
                            this_annotated_type = previous_annotated_type
                            relation_word = previous_relation_word
                        else: # Neither is "NOT", manually examine them.
                            # There is one case of Positive_Correlation vs. Association. Change to Association
                            if this_annotated_type == "Association" and previous_annotated_type == "Positive_Correlation":
                                sents[previous_id]["annotated_type"] = this_annotated_type
                                sents[previous_id]["annotated_relation_words"] = relation_word
                                sent_reps[sent_rep][1] = (this_annotated_type, relation_word)
                                print(f"{previous_id} and {this_relation_id} conflict: {previous_annotated_type} vs {this_annotated_type}, change to Association. Beginning: {text[0:10]}")
                            elif this_annotated_type != "Positive_Correlation" and previous_annotated_type != "Association":
                                print(f"{previous_id} and {this_relation_id} conflict: {previous_annotated_type} vs {this_annotated_type}. Beginning: {text[0:10]}")
                else:
                    sent_reps[sent_rep] = [this_relation_id, (this_annotated_type, relation_word)]

                sents[this_relation_id] = dict(
                    abstract_id=data["abstract_id"],
                    relation_id=this_relation_id,
                    entity_a_id=data["entity_a_id"],
                    entity_b_id=data["entity_b_id"],
                    text=text,
                    document_level_type=data["type"],
                    document_level_novel=data["novel"],
                    annotated_type=this_annotated_type,
                    annotated_relation_words=relation_word,
                    entity_a=[ent1["offset_start"], ent1["offset_finish"], ent1["type"]],
                    entity_b=[ent2["offset_start"], ent2["offset_finish"], ent2["type"]],
                    duplicated_flag=duplicated_flag
                )
    return sents

if __name__ == "__main__":

    parser =  argparse.ArgumentParser(description="Generate multi-sentence data for LitCoin Phase 2.")
    parser.add_argument("--data_path", action="store", dest="data_path", help="Directory of processed LitCoin phase 2 data.")
    args = parser.parse_args()

    data_df = pandas.read_csv("annotated_data.csv", na_filter=False)
    data_df["sentence"] = [sent.replace("</ENT2></ENT1>", "</ENT1></ENT2>") for sent in data_df["sentence"]]

    new_data_df = pandas.DataFrame()
    for column in ["abstract_id", "relation_id", "entity_a_id", "entity_b_id", "type", "novel"]:
        new_data_df[column] = [text.split(";") for text in data_df[column]]
    new_data_df["sentence"] = data_df["sentence"]

    new_data_df["annotated_type"] = [process_annotated_type(doc_type, true_type, true_relation) for doc_type, true_type, true_relation in zip(data_df["type"], data_df["True type"], data_df["True relation"])]

    copy_list_fp = os.path.join(args.data_path, "copy_list_train.csv")
    copy_list_dict = {}
    copy_list_df = pandas.read_csv(copy_list_fp)
    for pmid in set(copy_list_df["abstract_id"]):
        sub_df = copy_list_df[copy_list_df["abstract_id"] == pmid]
        for copy_from in set(sub_df["copy_from"]):
            sub_sub_df = sub_df[sub_df["copy_from"] == copy_from]
            all_copy_tos = list(sub_sub_df["copy_to"])
            this_group = [copy_from] + all_copy_tos
            for elem in this_group:
                this_to = copy.deepcopy(this_group)
                copy_list_dict[(pmid, elem)] = this_to


    annotations = get_annotations(data_df)

    sents_train = get_data(DATA_DIR=os.path.join(args.data_path, "processed_data_train.json"), annotations=annotations, copy_list_dict=copy_list_dict, is_test=False)
    train_out_file = os.path.join(args.data_path, "processed_data_train_split_annotation.json")
    train_outs = list(sents_train.values())
    json.dump(train_outs, open(train_out_file, "w"), indent=4)
    df = pandas.DataFrame(train_outs)
    train_out_pd_file = os.path.join(args.data_path, "processed_data_train_split_annotation_pd.json")
    df.to_json(train_out_pd_file, orient="table", indent=4)

    sents_test = get_data(DATA_DIR=os.path.join(args.data_path, "processed_data_test.json"), annotations=None, copy_list_dict=copy_list_dict, is_test=True)
    test_out_file=os.path.join(args.data_path, "processed_data_test_split.json")
    test_outs = list(sents_test.values())
    json.dump(test_outs, open(test_out_file, "w"), indent=4)
    df = pandas.DataFrame(test_outs)
    test_out_pd_file=os.path.join(args.data_path, "processed_data_test_split_pd.json")
    df.to_json(test_out_pd_file, orient="table", indent=4)