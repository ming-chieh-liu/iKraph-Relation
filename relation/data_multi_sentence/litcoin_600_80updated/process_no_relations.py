import os
import json
import itertools
import argparse

import pandas
import numpy

parser =  argparse.ArgumentParser(description="Generate no relation file for LitCoin Phase 2.")
parser.add_argument("--data_path", action="store", dest="data_path", help="Directory of processed LitCoin phase 2 data.")
args = parser.parse_args()

train_data_path = os.path.join(".", "multi_sentence_split_litcoin_600_80updated", "train_all.json")
no_rel_path = os.path.join(".", "multi_sentence_split_litcoin_600_80updated", "no_rel.csv")

RELATIONS_LIST = ["Association", "Positive_Correlation", "Negative_Correlation", "Bind", "Cotreatment", "Comparison", "Drug_Interaction", "Conversion"]
RELATIONS_DICT = {elem: idx for idx, elem in enumerate(RELATIONS_LIST)}
l_relations = len(RELATIONS_LIST)

ENTITY_LIST = ["CellLine", "ChemicalEntity", "DiseaseOrPhenotypicFeature", "GeneOrGeneProduct", "OrganismTaxon", "SequenceVariant"]
ENTITY_DICT = {elem: idx for idx, elem in enumerate(ENTITY_LIST)}
l_entities = len(ENTITY_LIST)


ret = {}
ret["average"] = numpy.zeros((l_entities, l_entities), dtype=float)

for rel_type in RELATIONS_LIST:
    ret[rel_type] = numpy.zeros((l_entities, l_entities))

train_data = json.load(open(train_data_path))

for elem in train_data:
    entity_a_type = elem["entity_a"][0][-1]
    entity_b_type = elem["entity_b"][0][-1]

    entity_a_idx = ENTITY_DICT[entity_a_type]
    entity_b_idx = ENTITY_DICT[entity_b_type]

    relation_type = elem["type"]
    if relation_type != "NOT":
        ret[relation_type][entity_a_idx, entity_b_idx] += 1
        ret[relation_type][entity_b_idx, entity_a_idx] += 1
        ret["average"][entity_a_idx, entity_b_idx] += 1
        ret["average"][entity_b_idx, entity_a_idx] += 1

with open(no_rel_path, "w") as no_rel_file:
    no_rel_file.write("type_a,type_b,relation\n")
    for key, val in ret.items():
        total = numpy.sum(val)
        val = val / total
        import pandas
        df = pandas.DataFrame(val, index=ENTITY_LIST, columns=ENTITY_LIST)
        if key == "average":
            continue
        for ent1, ent2 in itertools.combinations(ENTITY_LIST, r=2):
            idx1 = ENTITY_DICT[ent1]
            idx2 = ENTITY_DICT[ent2]
            line = f"{ent1},{ent2},{key}\n"
            if val[idx1, idx2] == 0.0:
                no_rel_file.write(line)