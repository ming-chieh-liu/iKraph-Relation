"""Convert user's multi-sentence split data to colleague's tagged format.

User's format per entry:
  text, entity_a: [[start, end, type], ...], entity_b: [...], type: "Association"

Colleague's expected format:
  text (with [SUBJECT]...[/SUBJECT] and [OBJECT]...[/OBJECT] tags),
  subject, object, relation
"""

import argparse
import json
import os
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "multi_sentence_split_litcoin_600"
OUTPUT_DIR = Path(__file__).resolve().parent / "data_tagged"
NUM_SPLITS = 5


def insert_tags(text: str, entity_a_spans: list, entity_b_spans: list) -> str:
    """Insert [SUBJECT]/[OBJECT] tags around entity mentions in text.

    Handles overlapping spans by nesting SUBJECT outside OBJECT.
    Processes insertions right-to-left to preserve earlier offsets.
    """
    # Build list of insertions: (position, priority, tag_string)
    # Priority ensures correct ordering at same position:
    #   - closing tags before opening tags at same position
    #   - at same position for opens: SUBJECT wraps outside OBJECT
    #   - at same position for closes: OBJECT closes before SUBJECT
    insertions = []
    for start, end, _ in entity_a_spans:
        # Opening tag: lower priority = inserted later (further right in final text)
        # We want SUBJECT to be outermost, so its open goes first (lower pos priority)
        insertions.append((start, 0, "[SUBJECT]"))
        insertions.append((end, 3, "[/SUBJECT]"))
    for start, end, _ in entity_b_spans:
        insertions.append((start, 1, "[OBJECT]"))
        insertions.append((end, 2, "[/OBJECT]"))

    # Sort by position descending, then by priority descending
    # (right-to-left insertion; at same position, higher priority = inserted first)
    insertions.sort(key=lambda x: (-x[0], -x[1]))

    for pos, _, tag in insertions:
        text = text[:pos] + tag + text[pos:]
    return text


def convert_entry(entry: dict) -> dict:
    """Convert a single data entry from user format to tagged format."""
    text = entry["text"]
    entity_a = entry["entity_a"]
    entity_b = entry["entity_b"]

    tagged_text = insert_tags(text, entity_a, entity_b)

    # Extract first mention text for subject/object
    subject = text[entity_a[0][0]:entity_a[0][1]]
    obj = text[entity_b[0][0]:entity_b[0][1]]

    return {
        "text": tagged_text,
        "subject": subject,
        "object": obj,
        "relation": entry["type"],
    }


def main():
    parser = argparse.ArgumentParser(description="Convert split data to tagged format.")
    parser.add_argument("--fold", type=int, required=True, help="Fold index (0-4). This split becomes validation; rest become training.")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Directory containing split_*.json files.")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Output directory for tagged data.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load all splits
    # Convention: fold N trains on splits N..N+3 (wrapping), validates on split (N+4)%5
    val_split = (args.fold + 4) % NUM_SPLITS
    train_entries = []
    val_entries = []
    for i in range(NUM_SPLITS):
        split_path = data_dir / f"split_{i}.json"
        with open(split_path) as f:
            split_data = json.load(f)
        if i == val_split:
            val_entries = split_data
        else:
            train_entries.extend(split_data)

    print(f"Fold {args.fold}: {len(train_entries)} train, {len(val_entries)} val entries")

    # Convert
    train_converted = [convert_entry(e) for e in train_entries]
    val_converted = [convert_entry(e) for e in val_entries]

    # Write output
    train_path = output_dir / f"fold_{args.fold}_train.json"
    val_path = output_dir / f"fold_{args.fold}_val.json"

    with open(train_path, "w") as f:
        json.dump(train_converted, f, indent=2)
    with open(val_path, "w") as f:
        json.dump(val_converted, f, indent=2)

    print(f"Wrote {train_path}")
    print(f"Wrote {val_path}")

    # Spot-check first entry
    print("\n--- Spot check (first train entry) ---")
    sample = train_converted[0]
    print(f"Subject: {sample['subject']}")
    print(f"Object:  {sample['object']}")
    print(f"Relation: {sample['relation']}")
    print(f"Text (first 300 chars): {sample['text'][:300]}")


if __name__ == "__main__":
    main()
