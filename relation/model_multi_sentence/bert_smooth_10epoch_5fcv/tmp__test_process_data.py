#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import json
import sys

# Import the functions from modeling.py
sys.path.insert(0, '/data/mliu/iKraph/relation/model_multi_sentence/bert_smooth_10epoch_5fcv')
from modeling import process_data, transform_sentence, label_dict

# Load data
with open('/data/mliu/iKraph/relation/model_multi_sentence/bert_smooth_10epoch_5fcv/multi_sentence_split_litcoin_600/split_0.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame and take first 10 rows
df = pd.DataFrame(data[:10])

# List of all transform methods to test
transform_methods = [
    # "entity_mask",
    # "entity_marker",
    # "entity_marker_punkt",
    # "typed_entity_marker",
    "typed_entity_marker_punct"
]

# Test each transform method
for method in transform_methods:
    print("\n\n")
    print("=" * 100)
    print(f"TRANSFORM METHOD: {method}")
    print("=" * 100)

    for flag in [True, False]: 

        # Process the data with current method
        df_processed = process_data(df.copy(), transform_method=method, move_entities_to_start=flag, mode='train')

        # Show first 3 rows for each method (to keep output manageable)
        for idx in range(min(3, len(df))):
            row = df.iloc[idx]

            # Extract entity mentions from original text
            entity_a_mentions = [row['text'][start:end] for start, end, _ in row['entity_a']]
            entity_b_mentions = [row['text'][start:end] for start, end, _ in row['entity_b']]

            print(f"\n{'-' * 100}")
            print(f"[Row {idx}] Type: {row['type']} -> Label: {df_processed.iloc[idx]['label']}")
            print(f"{'-' * 100}")
            print(f"Entity A (mentions): {entity_a_mentions}")
            print(f"Entity A (positions): {row['entity_a']}")
            print(f"Entity B (mentions): {entity_b_mentions}")
            print(f"Entity B (positions): {row['entity_b']}")
            print(f"\nBEFORE:")
            print(f"{row['text'][:300]}..." if len(row['text']) > 300 else row['text'])
            print(f"\nAFTER:")
            print(f"{df_processed.iloc[idx]['text'][:400]}..." if len(df_processed.iloc[idx]['text']) > 400 else df_processed.iloc[idx]['text'])
            print()
