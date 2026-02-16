# Generate training json data for multi-sentence format from All.json data
import os
import json
import argparse


def gen_multi_sentence_data(input_data):
    """
    Generate multi-sentence data without needing external entity mentions file
    Entity information is already in the processed data from process_multisentence_data_all.py
    """
    output_data = []
    for data in input_data:
        doc_id = str(data["abstract_id"])
        ent1_id = data["entity_a_id"]
        ent2_id = data["entity_b_id"]
        rel_id = data["relation_id"]
        tag = data["type"]
        novel = data["novel"]
        overlap = data["overlapping"]
        sents = data["sents"]

        output_data.append({})
        output_data[-1]["abstract_id"] = data["abstract_id"]
        output_data[-1]["relation_id"] = rel_id
        output_data[-1]["entity_a_id"] = ent1_id
        output_data[-1]["entity_b_id"] = ent2_id
        output_data[-1]["type"] = tag
        output_data[-1]["novel"] = novel
        output_data[-1]["text"] = ""
        output_data[-1]["entity_a"] = []
        output_data[-1]["entity_b"] = []

        # Concatenate sentences containing entities
        for st in sents:
            tmptxt = st["text"]
            if len(st["entity_a"]) > 0 or len(st["entity_b"]) > 0:
                ent_exist = True
            else:
                ent_exist = False

            # Add entity_a positions
            for e in st["entity_a"]:
                s = e[0]
                f = e[1]
                entity_type = e[2]
                clen = len(output_data[-1]["text"])
                output_data[-1]["entity_a"].append([clen+s, clen+f, entity_type])

            # Add entity_b positions
            for e in st["entity_b"]:
                s = e[0]
                f = e[1]
                entity_type = e[2]
                clen = len(output_data[-1]["text"])
                output_data[-1]["entity_b"].append([clen+s, clen+f, entity_type])

            # Only include sentences that contain entities
            if ent_exist:
                output_data[-1]["text"] = output_data[-1]["text"] + tmptxt + ' '

        # Remove trailing space
        output_data[-1]["text"] = output_data[-1]["text"].rstrip()

    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate actual training data (multi-sentence) from sentence-split data.")
    parser.add_argument("--data_path", action="store", dest="data_path", help="Directory containing processed data.")
    args = parser.parse_args()

    # Read sentence-split data from process_multisentence_data_all.py
    processed_file = os.path.join(args.data_path, "processed", "sentence_split_all.json")
    all_data = json.load(open(processed_file))

    print(f"Processing {len(all_data)} relations...")

    # Generate multi-sentence data
    outputs = gen_multi_sentence_data(all_data)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.data_path, "processed")
    os.makedirs(output_dir, exist_ok=True)

    # Save output
    output_file = os.path.join(output_dir, "multi_sentence_all.json")
    json.dump(outputs, open(output_file, "w"), indent=4)
    print(f"Saved multi-sentence data to {output_file}")
    print(f"Total relations processed: {len(outputs)}")
