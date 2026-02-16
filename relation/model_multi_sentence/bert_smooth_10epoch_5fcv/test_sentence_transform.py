#!/usr/bin/env python
"""
Test script to visualize sentence transformation with multiple entity mentions
"""

import json
import pandas as pd
from transformers import RobertaTokenizer
from SentenceDataset import SentenceDataset

# Read data
with open('multi_sentence_split_litcoin_600/split_0.json', 'r') as f:
    data = json.load(f)
    first_row = data[0]

print("=" * 100)
print("BEFORE PROCESSING")
print("=" * 100)

print(f"\nAbstract ID: {first_row['abstract_id']}")
print(f"Relation ID: {first_row['relation_id']}")
print(f"Type: {first_row['type']}")
print(f"\nOriginal Text:")
print(first_row['text'])

print(f"\n\nEntity A (entity_a_id={first_row['entity_a_id']}):")
print(f"  Total mentions: {len(first_row['entity_a'])}")
for i, mention in enumerate(first_row['entity_a']):
    start, end, ent_type = mention
    mention_text = first_row['text'][start:end]
    print(f"  [{i}] Position {start}-{end}, Type: {ent_type}")
    print(f"      Text: '{mention_text}'")

print(f"\nEntity B (entity_b_id={first_row['entity_b_id']}):")
print(f"  Total mentions: {len(first_row['entity_b'])}")
for i, mention in enumerate(first_row['entity_b']):
    start, end, ent_type = mention
    mention_text = first_row['text'][start:end]
    print(f"  [{i}] Position {start}-{end}, Type: {ent_type}")
    print(f"      Text: '{mention_text}'")

# Create actual dataset instance
print("\n" + "=" * 100)
print("AFTER PROCESSING (Using SentenceDataset)")
print("=" * 100)

# Create dataframe with just the first row
df = pd.DataFrame([first_row])

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained("../../pre_trained_models/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf")

# Config matching generate_train_configs.py
config = {
    "model_type": "triplet",
    "max_len": 512,
    "transform_method": "typed_entity_marker_punct",
    "label_column_name": "type",
    "move_entities_to_start": False,
    "no_relation_file": "",
    "is_train": False,
    "remove_cellline_organismtaxon": False
}

# Create dataset
dataset = SentenceDataset([df], tokenizer, config)

# Get processed dataframe
processed_df = dataset.get_processed_dataframe()
transformed_text = processed_df.loc[0, "processed_text"]
new_ent_a_pos = processed_df.loc[0, "processed_ent1"]
new_ent_b_pos = processed_df.loc[0, "processed_ent2"]

print(f"\nTransformed Text:")
print(transformed_text)

print(f"\n\nEntity A Positions After Processing:")
print(f"  Total mentions: {len(new_ent_a_pos)}")
for i, pos in enumerate(new_ent_a_pos):
    start, end, ent_type = pos
    extracted_text = transformed_text[start:end]
    print(f"  [{i}] Position {start}-{end}, Type: {ent_type}")
    print(f"      Extracted: '{extracted_text}'")

print(f"\nEntity B Positions After Processing:")
print(f"  Total mentions: {len(new_ent_b_pos)}")
for i, pos in enumerate(new_ent_b_pos):
    start, end, ent_type = pos
    extracted_text = transformed_text[start:end]
    print(f"  [{i}] Position {start}-{end}, Type: {ent_type}")
    print(f"      Extracted: '{extracted_text}'")

print("\n" + "=" * 100)
print("VERIFICATION")
print("=" * 100)

# Verify all extractions match expected entity types
all_correct = True
for i, pos in enumerate(new_ent_a_pos):
    start, end, ent_type = pos
    extracted = transformed_text[start:end]
    if extracted != ent_type:
        print(f"✗ Entity A[{i}]: Expected '{ent_type}', got '{extracted}'")
        all_correct = False

for i, pos in enumerate(new_ent_b_pos):
    start, end, ent_type = pos
    extracted = transformed_text[start:end]
    if extracted != ent_type:
        print(f"✗ Entity B[{i}]: Expected '{ent_type}', got '{extracted}'")
        all_correct = False

if all_correct:
    print("✓ All entity positions correctly extracted!")
else:
    print("✗ Some positions are incorrect")
