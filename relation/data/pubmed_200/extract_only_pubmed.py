import pandas as pd


def extract_batch_number(relation_id):
    """
    Extract batch number from relation_id.

    Format: "True.11683992.6401.101404.2.1.combination0"
    The 3rd number from the end is the batch number.

    Args:
        relation_id: String in format "prefix.X.Y.batch.Z.combination"

    Returns:
        Batch number as integer, or None if pattern doesn't match
    """
    parts = relation_id.split('.')
    if len(parts) >= 3:
        try:
            # Get 3rd element from the end
            batch_num = int(parts[-3])
            return batch_num
        except (ValueError, IndexError):
            return None
    return None


def filter_batches(input_file, output_file, batch_numbers=[2, 3]):
    """
    Filter JSON data to retain only specific batch numbers.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        batch_numbers: List of batch numbers to retain (default: [2, 3])
    """
    df = pd.read_json(input_file, orient='table')

    # Extract batch numbers and filter
    df['batch_number'] = df['relation_id'].apply(extract_batch_number)
    filtered_df = df[df['batch_number'].isin(batch_numbers)].drop(columns=['batch_number'])

    # Save to output file
    filtered_df.to_json(output_file, orient='table', indent=4)

    print(f"Original records: {len(df)}")
    print(f"Filtered records (batches {batch_numbers}): {len(filtered_df)}")
    print(f"Output saved to: {output_file}")


if __name__ == "__main__":
    input_file = "../litcoin_80updated_pubmed_200/new_annotated_train_litcoin_80updated_pubmed_200_pd.json"
    output_file = "./new_annotated_train_pubmed_200_pd.json"

    filter_batches(input_file, output_file, batch_numbers=[2, 3])
