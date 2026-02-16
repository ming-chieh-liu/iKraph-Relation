import json 
import pandas as pd
import random

random.seed(318)

def main():
    litcoin_data = pd.read_json("../litcoin_320_131_80_33_80updated/new_annotated_train_litcoin_320_131_80updated_pd.json", orient='table')
    
    pubmed_data = pd.read_json("../litcoin_80updated_pubmed_200/new_annotated_train_litcoin_80updated_pubmed_200_pd.json", orient='table')

    with open("../litcoin_400_164_80updated_pubmed_100_100/train_test_pmids.json", "r") as f:
        pubmed_train_id = json.load(f)["train_pmid"]

    pubmed_train = pubmed_data[pubmed_data['abstract_id'].isin(pubmed_train_id)].reset_index(drop=True)

    # Combine litcoin with sampled pubmed for training
    combined_data = pd.concat([litcoin_data, pubmed_train], ignore_index=True)

    output_file = "./new_annotated_train_litcoin_320_131_80updated_pubmed_100_pd.json"
    combined_data.to_json(output_file, orient='table', indent=4)
    print(f"Saved combined data to {output_file}")

    pmids = list(set(combined_data["abstract_id"]))
    random.shuffle(pmids)

    total_ids = len(pmids)
    num_splits = 5
    per_split_len = total_ids // num_splits
    split_poses = [[idx*per_split_len, idx*per_split_len+per_split_len] for idx in range(0, num_splits-1)] # first n-1 splits
    split_poses.append([per_split_len*(num_splits-1), total_ids])

    pmids_per_split = []
    for split_start, split_end in split_poses:
        pmids_per_split.append(pmids[split_start: split_end])

    copy_list_df = pd.read_csv("../litcoin_400_curated/copy_list_litcoin_400_curated.csv")

    import os
    for split_id, this_pmids in enumerate(pmids_per_split):
        os.makedirs(os.path.join(f"new_train_splits_litcoin_320_131_80updated_pubmed_100/split_{split_id}"), exist_ok=True)
        this_copy_list = copy_list_df[copy_list_df["abstract_id"].isin(this_pmids)]
        this_data = combined_data[combined_data["abstract_id"].isin(this_pmids)]

        output_data_fp = os.path.join(f"new_train_splits_litcoin_320_131_80updated_pubmed_100/split_{split_id}/data.json")
        this_data.reset_index(drop=True).to_json(open(output_data_fp, "w"), indent=4, orient="table")
        output_copylist_fp = os.path.join(f"new_train_splits_litcoin_320_131_80updated_pubmed_100/split_{split_id}/copy_list.csv")
        this_copy_list.reset_index(drop=True).to_csv(open(output_copylist_fp, "w"), index=False)

if __name__ == "__main__":
    main()