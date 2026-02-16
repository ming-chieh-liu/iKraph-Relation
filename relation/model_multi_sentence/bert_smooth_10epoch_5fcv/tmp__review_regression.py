#!/usr/bin/env python3
"""
Interactive review tool for regression analysis between two models.
Samples regression cases from fold-specific diff files and allows manual review.
"""

import argparse
import json
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ANSI color codes
BOLD = '\033[1m'
RESET = '\033[0m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
ENTITY1_COLOR = '\033[93m'  # Yellow for Entity A
ENTITY2_COLOR = '\033[92m'  # Green for Entity B


def load_original_data(data_dir: Path) -> Dict[Tuple, Dict]:
    """
    Load original untransformed data from train_all.json.

    Returns:
        Dictionary mapping (abstract_id, relation_id, entity_a_id, entity_b_id) to entry dict
    """
    train_all_path = data_dir / "train_all.json"

    if not train_all_path.exists():
        raise FileNotFoundError(f"train_all.json not found at {train_all_path}")

    with open(train_all_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Create lookup dictionary with composite key
    lookup = {}
    for entry in data:
        key = (
            entry['abstract_id'],
            entry['relation_id'],
            entry['entity_a_id'],
            entry['entity_b_id']
        )
        lookup[key] = entry

    return lookup


def clear_terminal():
    """Clear the terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def highlight_all_entity_mentions(text: str, entity_a_positions: List, entity_b_positions: List) -> str:
    """
    Highlight ALL entity mentions in the text using position lists from JSON.

    Args:
        text: Original text
        entity_a_positions: List of [start, end, type] for entity A mentions
        entity_b_positions: List of [start, end, type] for entity B mentions

    Returns:
        Text with ANSI color codes inserted at all entity positions
    """
    # Combine all positions with entity markers
    highlights = []
    for start, end, _ in entity_a_positions:
        highlights.append((start, end, 'A'))
    for start, end, _ in entity_b_positions:
        highlights.append((start, end, 'B'))

    # Sort by position in reverse order to maintain string offsets
    highlights.sort(reverse=True, key=lambda x: x[0])

    # Insert color codes
    result = text
    for start, end, entity_type in highlights:
        color = ENTITY1_COLOR if entity_type == 'A' else ENTITY2_COLOR
        result = result[:start] + f'{color}{BOLD}' + result[start:end] + f'{RESET}' + result[end:]

    return result


def convert_response(response: str) -> Optional[bool]:
    """Convert user response to boolean or special values."""
    response = response.strip().lower()
    if response in ['y', 'yes']:
        return True
    elif response in ['n', 'no']:
        return False
    elif response in ['na', 'n/a']:
        return None
    elif response in ['e', 'exit']:
        return "exit"
    else:
        raise ValueError("Invalid response. Please enter 'y', 'n', 'na', or 'e'.")


def sample_with_stride(df: pd.DataFrame, sample_size: int, stride: int = 5) -> pd.DataFrame:
    """
    Sample dataframe with strided indices to spread samples across the dataset.

    Strategy:
    - Pick indices: 0, 5, 10, 15, ... (step of stride)
    - If sample_size * stride > len(df): wrap around with offset
    - After first pass (0, 5, 10...), continue with (1, 6, 11...)
    - Continue until reaching sample_size samples

    This ensures deterministic and spread-out sampling.
    """
    indices = []
    offset = 0

    while len(indices) < sample_size and offset < stride:
        for i in range(offset, len(df), stride):
            if i not in indices:
                indices.append(i)
                if len(indices) >= sample_size:
                    break
        offset += 1

    # Cap at available data
    indices = indices[:sample_size]

    return df.iloc[indices]


def load_fold_samples(regression_dir: Path, fold_num: int, sample_size: int) -> pd.DataFrame:
    """Load and sample from a specific fold diff file using strided sampling."""
    fold_file = regression_dir / f"diff_split_{fold_num}.csv"

    if not fold_file.exists():
        raise FileNotFoundError(f"Fold file not found: {fold_file}")

    df = pd.read_csv(fold_file)

    # Use strided sampling instead of top N
    sampled_df = sample_with_stride(df, sample_size, stride=5)

    return sampled_df


def display_entry(
    entry: pd.Series,
    entry_index: int,
    total_entries: int,
    original_data_600: Dict,
    original_data_600_80: Dict
):
    """Display a single regression case with formatting using original untransformed text."""
    clear_terminal()

    print("=" * 80)
    print(f"{BOLD}{CYAN}REGRESSION REVIEW: Entry {entry_index + 1}/{total_entries}{RESET}")
    print("=" * 80)
    print()

    # Create composite key for lookup
    key = (
        entry['abstract_id_model1'],
        entry['relation_id_model1'],
        entry['entity_a_id_model1'],
        entry['entity_b_id_model1']
    )

    # Get original data for both models
    orig_600 = original_data_600.get(key)
    orig_600_80 = original_data_600_80.get(key)

    # Assert that both models have the entry
    assert orig_600 is not None, f"Entry not found in litcoin_600 data: {key}"
    assert orig_600_80 is not None, f"Entry not found in litcoin_600_80updated data: {key}"

    # Assert all key fields match
    assert orig_600['abstract_id'] == orig_600_80['abstract_id'], "abstract_id mismatch"
    assert orig_600['relation_id'] == orig_600_80['relation_id'], "relation_id mismatch"
    assert orig_600['entity_a_id'] == orig_600_80['entity_a_id'], "entity_a_id mismatch"
    assert orig_600['entity_b_id'] == orig_600_80['entity_b_id'], "entity_b_id mismatch"

    print(f"{BOLD}Fold:{RESET} {entry['fold']}")
    print(f"{BOLD}Relation ID:{RESET} {entry['relation_id_model1']}")
    print(f"{BOLD}Abstract ID:{RESET} {entry['abstract_id_model1']}")
    print()

    # Extract entity names from first mention in original text
    text = orig_600['text']
    entity_a_positions = orig_600['entity_a']
    entity_b_positions = orig_600['entity_b']

    entity_a_name = text[entity_a_positions[0][0]:entity_a_positions[0][1]]
    entity_b_name = text[entity_b_positions[0][0]:entity_b_positions[0][1]]

    print(f"{BOLD}Entity A:{RESET} {ENTITY1_COLOR}{BOLD}{entity_a_name}{RESET}")
    print(f"{BOLD}Entity B:{RESET} {ENTITY2_COLOR}{BOLD}{entity_b_name}{RESET}")
    print()

    # Display labels and predictions for both models
    label_600 = entry['label_class_model1']
    pred_600 = entry['predicted_class_model1']
    label_600_80 = entry['label_class_model2']
    pred_600_80 = entry['predicted_class_model2']

    print(f"{BOLD}litcoin_600:{RESET}")
    print(f"  Label:      {label_600}")
    print(f"  Prediction: {pred_600} {BOLD}✓ (was correct){RESET}")
    print()

    print(f"{BOLD}litcoin_600_80updated:{RESET}")
    print(f"  Label:      {label_600_80}")
    print(f"  Prediction: {pred_600_80} {BOLD}✗ (became wrong){RESET}")
    print()

    print("=" * 80)
    print(f"{BOLD}TEXT:{RESET}")
    print("=" * 80)

    # Highlight all entity mentions using position lists
    highlighted_text = highlight_all_entity_mentions(text, entity_a_positions, entity_b_positions)
    print(highlighted_text)
    print()

    print("=" * 80)


def collect_review(entry: pd.Series) -> Dict:
    """Collect user review for an entry."""
    label_600 = entry['label_class_model1']
    label_600_80 = entry['label_class_model2']

    # Conditional review based on label equality
    labels_same = (label_600 == label_600_80)

    if labels_same:
        # Only one question needed
        while True:
            try:
                label_review = input(f"{BOLD}Is the label correct? ([y]es/[n]o/[e]xit/na): {RESET}")
                label_review_converted = convert_response(label_review)
                if label_review_converted == "exit":
                    return {"exit": True}
                break
            except Exception as e:
                print(e)
                continue

        label_600_review = label_review_converted
        label_600_80_review = label_review_converted
    else:
        # Ask about both labels separately
        while True:
            try:
                label_600_review = input(f"{BOLD}Is litcoin_600 label correct? ([y]es/[n]o/[e]xit/na): {RESET}")
                label_600_review = convert_response(label_600_review)
                if label_600_review == "exit":
                    return {"exit": True}
                break
            except Exception as e:
                print(e)
                continue

        while True:
            try:
                label_600_80_review = input(f"{BOLD}Is litcoin_600_80updated label correct? ([y]es/[n]o/[e]xit/na): {RESET}")
                label_600_80_review = convert_response(label_600_80_review)
                if label_600_80_review == "exit":
                    return {"exit": True}
                break
            except Exception as e:
                print(e)
                continue

    # Collect notes
    note = input(f"{BOLD}Any additional notes? (press Enter to skip): {RESET}").strip()

    # Mark for further review
    while True:
        try:
            tbd = input(f"{BOLD}Mark this entry for further review? ([y]es/[n]o): {RESET}")
            tbd = convert_response(tbd)
            if tbd == "exit":
                return {"exit": True}
            break
        except Exception as e:
            print(e)
            continue

    return {
        "exit": False,
        "label_600_review": label_600_review,
        "label_600_80_review": label_600_80_review,
        "note": note,
        "tbd": tbd,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Interactive review tool for regression analysis"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10,
        help="Number of samples to review per fold (default: 10)"
    )
    args = parser.parse_args()

    # Get username
    username = input(f"{BOLD}Enter your username: {RESET}").strip()
    if not username:
        print("Username cannot be empty. Exiting.")
        return

    # Setup paths
    script_dir = Path(__file__).parent
    regression_dir = script_dir / "regression_analysis_typed_entity_marker_punct"
    output_file = regression_dir / f"{username}_review.jsonl"

    data_dir_600 = script_dir / "multi_sentence_split_litcoin_600"
    data_dir_600_80 = script_dir / "multi_sentence_split_litcoin_600_80updated"

    if not regression_dir.exists():
        print(f"Error: Regression analysis directory not found: {regression_dir}")
        return

    # Load original data from both models
    print("Loading original data from train_all.json files...")
    try:
        original_data_600 = load_original_data(data_dir_600)
        original_data_600_80 = load_original_data(data_dir_600_80)
        print(f"Loaded {len(original_data_600)} entries from litcoin_600")
        print(f"Loaded {len(original_data_600_80)} entries from litcoin_600_80updated")
    except FileNotFoundError as e:
        print(f"Error loading original data: {e}")
        return

    print(f"\nLoading regression cases with sample size: {args.sample_size} per fold")
    print(f"Output will be saved to: {output_file}\n")

    # Open output file
    with open(output_file, "w", encoding="utf-8") as f:
        exit_flag = False
        total_reviewed = 0

        # Process each fold (0-4)
        for fold_num in range(5):
            try:
                sampled_data = load_fold_samples(regression_dir, fold_num, args.sample_size)
                print(f"Loaded {len(sampled_data)} samples from fold {fold_num}")
            except FileNotFoundError as e:
                print(f"Skipping fold {fold_num}: {e}")
                continue

            # Review each entry in the fold
            for idx, (_, entry) in enumerate(sampled_data.iterrows()):
                # Display entry with original data
                display_entry(
                    entry,
                    total_reviewed,
                    len(sampled_data) * 5,
                    original_data_600,
                    original_data_600_80
                )

                review_result = collect_review(entry)

                if review_result.get("exit"):
                    exit_flag = True
                    break

                # Get entity names from original data
                key = (
                    entry['abstract_id_model1'],
                    entry['relation_id_model1'],
                    entry['entity_a_id_model1'],
                    entry['entity_b_id_model1']
                )
                orig_data = original_data_600[key]
                text = orig_data['text']
                entity_a_name = text[orig_data['entity_a'][0][0]:orig_data['entity_a'][0][1]]
                entity_b_name = text[orig_data['entity_b'][0][0]:orig_data['entity_b'][0][1]]

                record = {
                    "fold": int(entry['fold']),
                    "relation_id": entry['relation_id_model1'],
                    "abstract_id": entry['abstract_id_model1'],
                    "entity_a": entity_a_name,
                    "entity_b": entity_b_name,
                    "label_600": entry['label_class_model1'],
                    "pred_600": entry['predicted_class_model1'],
                    "label_600_80updated": entry['label_class_model2'],
                    "pred_600_80updated": entry['predicted_class_model2'],
                    "label_600_review": review_result["label_600_review"],
                    "label_600_80_review": review_result["label_600_80_review"],
                    "note": review_result["note"],
                    "tbd": review_result["tbd"],
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()  # Ensure data is written immediately

                total_reviewed += 1

            if exit_flag:
                break

        print()
        print("=" * 80)
        print(f"Review session completed. Total entries reviewed: {total_reviewed}")
        print(f"Results saved to: {output_file}")
        print("=" * 80)


if __name__ == "__main__":
    main()
