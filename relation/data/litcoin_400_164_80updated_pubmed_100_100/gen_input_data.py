import json 
import pandas as pd
import random

random.seed(318)

def main():
    litcoin_data = pd.read_json("../litcoin_400_164_80updated/new_annotated_train_litcoin_400_164_80updated_pd.json", orient='table')
    
    pubmed_data = pd.read_json("../litcoin_80updated_pubmed_200/new_annotated_train_litcoin_80updated_pubmed_200_pd.json", orient='table')

    # Filter to keep only batch 2 and 3
    pubmed_data['batch'] = pubmed_data['relation_id'].apply(lambda x: x.split('.')[-3])
    pubmed_data = pubmed_data[pubmed_data['batch'].isin(['2', '3'])]

    # Sample 50 unique PMIDs from batch 2 and 50 from batch 3
    batch_2_pmids = pubmed_data[pubmed_data['batch'] == '2']['abstract_id'].unique()
    batch_3_pmids = pubmed_data[pubmed_data['batch'] == '3']['abstract_id'].unique()

    n_batch_2 = len(batch_2_pmids)
    n_batch_3 = len(batch_3_pmids)

    sampled_batch_2 = random.sample(list(batch_2_pmids), n_batch_2//2)
    sampled_batch_3 = random.sample(list(batch_3_pmids), n_batch_3//2)
    sampled_pmids = sampled_batch_2 + sampled_batch_3

    # Split pubmed data into train (sampled) and test (rest)
    pubmed_train = pubmed_data[pubmed_data['abstract_id'].isin(sampled_pmids)].reset_index(drop=True)
    pubmed_test = pubmed_data[~pubmed_data['abstract_id'].isin(sampled_pmids)].reset_index(drop=True)

    # Save test data
    pubmed_test_output = "./new_annotated_test_pubmed_100_pd.json"
    pubmed_test.drop(columns=['batch']).to_json(pubmed_test_output, orient='table', indent=4)
    print(f"Saved {len(pubmed_test)} test entries with {len(pubmed_test['abstract_id'].unique())} unique PMIDs to {pubmed_test_output}")

    # Combine litcoin with sampled pubmed for training
    combined_data = pd.concat([litcoin_data, pubmed_train.drop(columns=['batch'])], ignore_index=True)

    output_file = "./new_annotated_train_litcoin_400_164_80updated_pubmed_100_pd.json"
    combined_data.to_json(output_file, orient='table', indent=4)
    print(f"Saved combined data to {output_file}")

    train_pmids = list(set(combined_data["abstract_id"]))
    test_pmids = list(set(pubmed_test["abstract_id"]))
    pmids_output = {
        "train_pmid": train_pmids,
        "test_pmid": test_pmids
    }
    pmids_output_file = "./train_test_pmids.json"
    with open(pmids_output_file, "w") as f:
        json.dump(pmids_output, f, indent=4)
    print(f"Saved train/test PMIDs to {pmids_output_file}")

    pmids = train_pmids.copy()
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
        os.makedirs(os.path.join(f"new_train_splits_litcoin_400_164_80updated_pubmed_100/split_{split_id}"), exist_ok=True)
        this_copy_list = copy_list_df[copy_list_df["abstract_id"].isin(this_pmids)]
        this_data = combined_data[combined_data["abstract_id"].isin(this_pmids)]

        output_data_fp = os.path.join(f"new_train_splits_litcoin_400_164_80updated_pubmed_100/split_{split_id}/data.json")
        this_data.reset_index(drop=True).to_json(open(output_data_fp, "w"), indent=4, orient="table")
        output_copylist_fp = os.path.join(f"new_train_splits_litcoin_400_164_80updated_pubmed_100/split_{split_id}/copy_list.csv")
        this_copy_list.reset_index(drop=True).to_csv(open(output_copylist_fp, "w"), index=False)

if __name__ == "__main__":
    main()