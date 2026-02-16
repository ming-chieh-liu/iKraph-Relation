import pandas as pd
import random
import json

random.seed(318)

def split_data_by_abstract_id(data, train_ratio=0.8, filter_pmids=None):
    """Split data into train/test based on abstract_id

    Args:
        data: DataFrame to split
        train_ratio: Ratio for train split (default 0.8)
        filter_pmids: Set/list of PMIDs that must be in train set
    """
    if filter_pmids is None:
        filter_pmids = set()
    else:
        filter_pmids = set(filter_pmids)

    all_pmids = list(set(data["abstract_id"]))

    # Separate PMIDs into filter and non-filter
    filter_pmids_in_data = [p for p in all_pmids if p in filter_pmids]
    non_filter_pmids = [p for p in all_pmids if p not in filter_pmids]

    # Calculate how many PMIDs we need for train (total)
    total_train_needed = int(len(all_pmids) * train_ratio)

    # Filter PMIDs go to train
    train_pmids = filter_pmids_in_data.copy()

    # Calculate how many more PMIDs we need from non-filter set
    remaining_train_needed = total_train_needed - len(train_pmids)

    # Shuffle non-filter PMIDs and add to train/test
    random.shuffle(non_filter_pmids)
    train_pmids.extend(non_filter_pmids[:remaining_train_needed])
    test_pmids = non_filter_pmids[remaining_train_needed:]

    train_data = data[data["abstract_id"].isin(train_pmids)].reset_index(drop=True)
    test_data = data[data["abstract_id"].isin(test_pmids)].reset_index(drop=True)

    return train_data, test_data, len(train_pmids), len(test_pmids)

def main():
    old_train_data = pd.read_json("../litcoin_400_curated/new_annotated_train_litcoin_400_curated_pd.json", orient='table')
    new_train_data = pd.read_json("../litcoin_164/new_annotated_train_litcoin_164_pd.json", orient='table')
    filter_data = pd.read_json("../litcoin_80updated_pubmed_200/new_annotated_train_litcoin_80updated_pubmed_200_pd.json", orient='table')

    filter_pmids = set(filter_data["abstract_id"])

    # Split old_train_data into 80% train / 20% test
    old_train_80, old_test_20, old_train_n_pmids, old_test_n_pmids = split_data_by_abstract_id(old_train_data, train_ratio=0.8, filter_pmids=filter_pmids)

    # Split new_train_data into 80% train / 20% test
    new_train_80, new_test_20, new_train_n_pmids, new_test_n_pmids = split_data_by_abstract_id(new_train_data, train_ratio=0.8, filter_pmids=filter_pmids)

    # Combine train portions (80%)
    combined_train = pd.concat([old_train_80, new_train_80], ignore_index=True)
    combined_train_n_pmids = len(set(combined_train["abstract_id"]))

    # Combine test portions (20%)
    combined_test = pd.concat([old_test_20, new_test_20], ignore_index=True)
    combined_test_n_pmids = len(set(combined_test["abstract_id"]))

    # Save combined_train
    train_output_file = "./new_annotated_train_litcoin_320_131_pd.json"
    combined_train.to_json(train_output_file, orient='table', indent=4)
    print(f"Saved combined_train to {train_output_file} with {combined_train_n_pmids} PMIDs")

    # Save combined_test
    test_output_file = "./new_annotated_test_litcoin_80_33_pd.json"
    combined_test.to_json(test_output_file, orient='table', indent=4)
    print(f"Saved combined_test to {test_output_file} with {combined_test_n_pmids} PMIDs")

    # Save train and test PMIDs
    train_pmids = list(set(combined_train["abstract_id"]))
    test_pmids = list(set(combined_test["abstract_id"]))
    pmids_output = {
        "train_pmid": train_pmids,
        "test_pmid": test_pmids
    }
    pmids_output_file = "./train_test_pmids.json"
    with open(pmids_output_file, "w") as f:
        json.dump(pmids_output, f, indent=4)
    print(f"Saved train/test PMIDs to {pmids_output_file}")

    # Create 5-fold splits from combined_train
    pmids = list(set(combined_train["abstract_id"]))
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
        os.makedirs(os.path.join(f"new_train_splits_litcoin_320_131/split_{split_id}"), exist_ok=True)
        this_copy_list = copy_list_df[copy_list_df["abstract_id"].isin(this_pmids)]
        this_data = combined_train[combined_train["abstract_id"].isin(this_pmids)]

        output_data_fp = os.path.join(f"new_train_splits_litcoin_320_131/split_{split_id}/data.json")
        this_data.reset_index(drop=True).to_json(open(output_data_fp, "w"), indent=4, orient="table")
        output_copylist_fp = os.path.join(f"new_train_splits_litcoin_320_131/split_{split_id}/copy_list.csv")
        this_copy_list.reset_index(drop=True).to_csv(open(output_copylist_fp, "w"), index=False)

if __name__ == "__main__":
    main()