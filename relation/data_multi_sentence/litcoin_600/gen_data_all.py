import os
import itertools
import copy
import json
import argparse


def flatten_list(list_of_list):
    ret = []
    for elem in list_of_list:
        for sub_elem in elem:
            ret.append(sub_elem)
    return ret


def get_appearance(entity_id, list_of_entity_ids):
    """Get list of booleans indicating which entity mentions contain this entity_id"""
    return [entity_id in entity_ids for entity_ids in list_of_entity_ids]


def check_overlapping(appearance_a, appearance_b):
    """Check overlapping pattern between two entity appearances"""
    assert len(appearance_a) == len(appearance_b)
    has_a_true_b_false = False
    has_a_false_b_true = False
    has_a_true_b_true = False

    for bool_a, bool_b in zip(appearance_a, appearance_b):
        if bool_a == True and bool_b == False: has_a_true_b_false = True
        if bool_b == True and bool_a == False: has_a_false_b_true = True
        if bool_a == True and bool_b == True: has_a_true_b_true = True

    if not has_a_true_b_true: return "No Overlapping"
    if has_a_true_b_true and has_a_false_b_true and not has_a_true_b_false: return "A subset of B"
    if has_a_true_b_true and has_a_true_b_false and not has_a_false_b_true: return "A superset of B"
    if has_a_true_b_true and has_a_true_b_false and has_a_false_b_true: return "Partial Overlapping"
    raise ValueError


def convert_all_json_format(doc):
    """
    Convert All.json format to entity-centric format for processing
    Returns: pmid, title, abstract, entity_list, relation_dict
    """
    pmid = str(doc["document_id"])
    title = doc["title"]
    abstract = doc["abstract"]
    combined_text = title + " " + abstract
    len_title = len(title)
    abstract_pos_start = len_title + 1

    # Build entity list: each entity mention with its entity_ids and position info
    entity_list = []
    for entity in doc["entities"]:
        entity_id = entity["id"]
        entity_type = entity["type"]
        for i, span in enumerate(entity["span"]):
            offset_start = span[0]
            offset_finish = span[1]

            # Determine if entity is in title or abstract
            if offset_start < abstract_pos_start:
                position = "title"
                # offset remains the same for title
            else:
                position = "abstract"
                # Adjust offset to be relative to abstract start
                offset_start = offset_start - abstract_pos_start
                offset_finish = offset_finish - abstract_pos_start

            mention = entity["Name"][i] if i < len(entity["Name"]) else entity["Name"][0]

            # Verify mention matches text
            if position == "title":
                assert title[offset_start:offset_finish] == mention, f"Mismatch: '{title[offset_start:offset_finish]}' != '{mention}'"
            else:
                assert abstract[offset_start:offset_finish] == mention, f"Mismatch: '{abstract[offset_start:offset_finish]}' != '{mention}'"

            entity_list.append({
                "entity_ids": [entity_id],  # Store as list for consistency with original format
                "offset_start": offset_start,
                "offset_finish": offset_finish,
                "type": entity_type,
                "mention": mention,
                "position": position
            })

    # Build relation dictionary: key=(entity_a_id, entity_b_id), value=(relation_type, novel, relation_id)
    relation_dict = {}
    for rel_idx, rel in enumerate(doc["relation"]):
        ent1_id = rel["ent1"]["id"]
        ent2_id = rel["ent2"]["id"]
        rel_type = rel["reltyp"]
        novelty = rel["novelty"]
        novel = "Novel" if novelty == 1 else "Known"

        # Store both directions
        relation_dict[(ent1_id, ent2_id)] = (rel_type, novel, rel_idx)
        relation_dict[(ent2_id, ent1_id)] = (rel_type, novel, rel_idx)

    return pmid, title, abstract, entity_list, relation_dict


def entity_to_info_format(entities, entity_id):
    """
    Convert entity mentions matching entity_id to info dict format
    Returns list of dicts with id, position, offset_start, offset_finish, type, mention, entity_ids
    """
    info_list = []
    for entity in entities:
        if entity_id in entity["entity_ids"]:
            info_list.append({
                "position": entity["position"],
                "offset_start": entity["offset_start"],
                "offset_finish": entity["offset_finish"],
                "type": entity["type"],
                "mention": entity["mention"],
                "entity_ids": entity["entity_ids"]
            })
    return info_list


def gen_data_from_all_json(all_json_path, output_path):
    """
    Generate training data with synthetic negatives from All.json format
    Mimics the behavior of original gen_data.py but reads from All.json
    """
    print(f"Reading {all_json_path}...")
    all_docs = json.load(open(all_json_path, "r"))
    print(f"Total documents loaded: {len(all_docs)}")

    false_case_id = 0
    all_cases = []
    copy_list = []  # abstract_id, copy_from, copy_to

    for doc in all_docs:
        pmid, title, abstract, entity_list, relation_dict = convert_all_json_format(doc)

        # Get all entity_ids as list of lists (each mention's entity_ids)
        all_entity_ids = [entity["entity_ids"] for entity in entity_list]

        # Get unique entity_ids
        all_entity_ids_set = set(flatten_list(all_entity_ids))

        # Filter out entities with identical appearances (keep only one representative)
        filtered_entity_ids_set = copy.deepcopy(all_entity_ids_set)

        for entity_a_id, entity_b_id in itertools.combinations(all_entity_ids_set, r=2):
            appearance_a = get_appearance(entity_a_id, all_entity_ids)
            appearance_b = get_appearance(entity_b_id, all_entity_ids)
            if appearance_a == appearance_b and entity_b_id in filtered_entity_ids_set:
                filtered_entity_ids_set.remove(entity_b_id)
                copy_list.append([pmid, entity_a_id, entity_b_id])

        # Generate all entity pair combinations
        for entity_a_id, entity_b_id in itertools.combinations(filtered_entity_ids_set, r=2):
            appearance_a = get_appearance(entity_a_id, all_entity_ids)
            appearance_b = get_appearance(entity_b_id, all_entity_ids)
            assert appearance_a != appearance_b
            overlapping_flag = check_overlapping(appearance_a, appearance_b)

            # Get entity info in required format
            entity_a_info = entity_to_info_format(entity_list, entity_a_id)
            entity_b_info = entity_to_info_format(entity_list, entity_b_id)

            # Check if this entity pair has an explicit relation
            if (entity_a_id, entity_b_id) in relation_dict:
                relation_type, novel, rel_idx = relation_dict[(entity_a_id, entity_b_id)]
                relation_id = f"True.{pmid}.{rel_idx}"
            else:
                # Create synthetic "NOT" relation
                relation_type = "NOT"
                novel = "N/A"
                relation_id = f"False.{pmid}.{false_case_id}"
                false_case_id += 1

            this_case = {
                "relation_id": relation_id,
                "abstract_id": int(pmid),
                "title": title,
                "abstract": abstract,
                "entity_a_id": entity_a_id,
                "entity_b_id": entity_b_id,
                "type": relation_type,
                "novel": novel,
                "entity_a_info": entity_a_info,
                "entity_b_info": entity_b_info,
                "overlapping": overlapping_flag
            }
            all_cases.append(this_case)

    # Save processed data
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "processed_all.json")
    json.dump(all_cases, open(output_file, "w"), indent=4)
    print(f"Saved {len(all_cases)} relations to {output_file}")

    # Save copy list
    copy_list_file = os.path.join(output_path, "copy_list_all.csv")
    with open(copy_list_file, "w") as copy_fp:
        copy_fp.write("abstract_id,copy_from,copy_to\n")
        for abstract_id, copy_from, copy_to in copy_list:
            copy_fp.write(f"{abstract_id},{copy_from},{copy_to}\n")
    print(f"Saved copy list to {copy_list_file}")

    # Print statistics
    positive_relations = sum(1 for case in all_cases if case["type"] != "NOT")
    negative_relations = sum(1 for case in all_cases if case["type"] == "NOT")
    print(f"\nStatistics:")
    print(f"  Total relations: {len(all_cases)}")
    print(f"  Positive relations: {positive_relations} ({100*positive_relations/len(all_cases):.1f}%)")
    print(f"  Synthetic NOT relations: {negative_relations} ({100*negative_relations/len(all_cases):.1f}%)")

    return all_cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training data with synthetic negatives from All.json format.")
    parser.add_argument("--data_path", action="store", dest="data_path",
                        help="Directory containing All.json file.")
    args = parser.parse_args()

    all_json_path = os.path.join(args.data_path, "All.json")
    output_path = os.path.join(args.data_path, "processed")

    gen_data_from_all_json(all_json_path, output_path)
