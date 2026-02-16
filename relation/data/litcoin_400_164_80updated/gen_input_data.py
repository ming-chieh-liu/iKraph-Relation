import pandas as pd
import random 

def main():
    combined_data = pd.read_json("./new_annotated_train_litcoin_400_164_80updated_pd.json", orient='table')

    pmids = set(list(combined_data["abstract_id"]))
    pmids = list(pmids)
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
        os.makedirs(os.path.join(f"new_train_splits_litcoin_400_164_80updated/split_{split_id}"), exist_ok=True)
        this_copy_list = copy_list_df[copy_list_df["abstract_id"].isin(this_pmids)]
        this_data = combined_data[combined_data["abstract_id"].isin(this_pmids)]

        output_data_fp = os.path.join(f"new_train_splits_litcoin_400_164_80updated/split_{split_id}/data.json")
        this_data.reset_index(drop=True).to_json(open(output_data_fp, "w"), indent=4, orient="table")
        output_copylist_fp = os.path.join(f"new_train_splits_litcoin_400_164_80updated/split_{split_id}/copy_list.csv")
        this_copy_list.reset_index(drop=True).to_csv(open(output_copylist_fp, "w"), index=False)


if __name__ == "__main__":
    main()