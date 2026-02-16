#!/usr/bin/env python3
"""Aggregate split training data from LitCoin-400 curated dataset.

This script consolidates data from multiple training splits (split_0 through split_4)
into single aggregated files. It processes both the main data files (data.json) and
copy list files (copy_list.csv) from each split directory.

Input:
    - ./new_train_splits_litcoin_400_curated/split_{0-4}/data.json: Training data splits
    - ./new_train_splits_litcoin_400_curated/split_{0-4}/copy_list.csv: Copy lists per split

Output:
    - new_annotated_train_litcoin_400_curated_pd.json: Aggregated training data
    - copy_list_litcoin_400_curated.csv: Aggregated copy list
"""
from pathlib import Path
import pandas as pd

def main():
    splits_dir = Path("./new_train_splits_litcoin_400_curated").resolve()

    sub_dirs = [splits_dir/f"split_{i}" for i in range(5)]

    aggregated_data = pd.DataFrame()
    aggregated_copy_list = pd.DataFrame()

    for sub_dir in sub_dirs:
        data = pd.read_json(sub_dir / "data.json", orient='table')

        aggregated_data = pd.concat([aggregated_data, data], ignore_index=True)

        copy_list = pd.read_csv(sub_dir / "copy_list.csv")
        aggregated_copy_list = pd.concat([aggregated_copy_list, copy_list], ignore_index=True)

    output_data_file = "new_annotated_train_litcoin_400_curated_pd.json"
    aggregated_data.to_json(output_data_file, orient='table', indent=4)
    print(f"Saved aggregated data to {output_data_file}")

    output_copy_list_file = "copy_list_litcoin_400_curated.csv"
    aggregated_copy_list.to_csv(output_copy_list_file, index=False)
    print(f"Saved aggregated copy list to {output_copy_list_file}")

if __name__ == "__main__":
    main()

