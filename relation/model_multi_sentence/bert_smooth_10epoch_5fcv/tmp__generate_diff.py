#!/usr/bin/env python3
"""
Script to compare predictions between two models and identify regression cases.
Finds instances where model 1 (600 articles) was correct but model 2 (600+80 updated) is wrong.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, List

transform_method = "typed_entity_marker_punct"

# Model directories
MODEL1_DIR = Path(__file__).parent / f"fine_tuned_models_litcoin_600_{transform_method}"
MODEL2_DIR = Path(__file__).parent / f"fine_tuned_models_litcoin_600_80updated_{transform_method}"

# Output directory
OUTPUT_DIR = Path(__file__).parent / f"regression_analysis_{transform_method}"
# Fold patterns
FOLD_PATTERN = "NEWDATA_triplet_{}_False_split_{}_16_3e-05_roberta_ls0.02"

# Label mapping
LABEL_LIST = ["NOT", "Association", "Positive_Correlation", "Negative_Correlation",
              "Bind", "Cotreatment", "Comparison", "Drug_Interaction", "Conversion"]
LABEL_DICT = {idx: val for idx, val in enumerate(LABEL_LIST)}


def load_predictions(model_dir: Path, fold_num: int) -> pd.DataFrame:
    """Load predictions.csv from a specific model fold."""
    fold_name = FOLD_PATTERN.format(transform_method, fold_num)
    predictions_path = model_dir / fold_name / "eval_predictions.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    df = pd.read_csv(predictions_path, index_col=0)

    # Convert numeric labels to class names for comparison
    df['label_class'] = df['label'].map(LABEL_DICT)

    return df


def compare_predictions(df1: pd.DataFrame, df2: pd.DataFrame, fold_num: int) -> pd.DataFrame:
    """
    Compare predictions from two models and identify regression cases.

    Regression case: model 1 correct AND model 2 wrong

    Args:
        df1: Predictions from model 1 (baseline)
        df2: Predictions from model 2 (updated annotations)
        fold_num: Fold number for identification

    Returns:
        DataFrame containing only regression cases
    """
    # Assert that both dataframes have the same number of rows
    assert len(df1) == len(df2), (
        f"Fold {fold_num}: Row count mismatch! "
        f"Model 1: {len(df1)} rows, Model 2: {len(df2)} rows"
    )

    # Iterate through rows and verify alignment, collect regression cases
    regression_rows = []

    for idx in range(len(df1)):
        row1 = df1.iloc[idx]
        row2 = df2.iloc[idx]

        # Assert that key fields match between corresponding rows
        assert row1['abstract_id'] == row2['abstract_id'], (
            f"Fold {fold_num}, row {idx}: abstract_id mismatch! "
            f"Model 1: {row1['abstract_id']}, Model 2: {row2['abstract_id']}"
        )
        assert row1['entity_a_id'] == row2['entity_a_id'], (
            f"Fold {fold_num}, row {idx}: entity_a_id mismatch! "
            f"Model 1: {row1['entity_a_id']}, Model 2: {row2['entity_a_id']}"
        )
        assert row1['entity_b_id'] == row2['entity_b_id'], (
            f"Fold {fold_num}, row {idx}: entity_b_id mismatch! "
            f"Model 1: {row1['entity_b_id']}, Model 2: {row2['entity_b_id']}"
        )
        assert row1['relation_id'] == row2['relation_id'], (
            f"Fold {fold_num}, row {idx}: relation_id mismatch! "
            f"Model 1: {row1['relation_id']}, Model 2: {row2['relation_id']}"
        )

        # Check if this is a regression case
        # Model 1 correct: label_class == predicted_class
        # Model 2 wrong: label_class != predicted_class
        model1_correct = row1['label_class'] == row1['predicted_class']
        model2_wrong = row2['label_class'] != row2['predicted_class']

        if model1_correct and model2_wrong:
            # Create merged row with suffixes
            regression_row = {}
            for col in df1.columns:
                regression_row[f"{col}_model1"] = row1[col]
            for col in df2.columns:
                regression_row[f"{col}_model2"] = row2[col]
            regression_rows.append(regression_row)

    # Convert to DataFrame
    regression_cases = pd.DataFrame(regression_rows) if regression_rows else pd.DataFrame()

    # Add fold information
    if not regression_cases.empty:
        regression_cases['fold'] = fold_num

        # Reorder columns for clarity
        important_cols = [
            'fold',
            'abstract_id_model1',
            'relation_id_model1',
            'entity_a_id_model1',
            'entity_b_id_model1',
            'label_model1',  # ground truth (numeric)
            'label_class_model1',  # ground truth (class name)
            'predicted_class_model1',  # model 1 prediction (correct)
            'predicted_class_model2',  # model 2 prediction (wrong)
            'text_model1',
            'entity_a_model1',
            'entity_b_model1',
        ]

        # Include all columns, but put important ones first
        other_cols = [col for col in regression_cases.columns if col not in important_cols]
        regression_cases = regression_cases[important_cols + other_cols]

    return regression_cases


def main():
    """Main function to process all folds and generate diff files."""
    print("=" * 80)
    print("REGRESSION ANALYSIS: Model 1 (600) vs Model 2 (600+80updated)")
    print("=" * 80)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_regression_cases = []
    fold_stats = []

    # Process each fold
    for fold_num in range(5):
        print(f"Processing fold {fold_num}...")

        try:
            # Load predictions from both models
            df1 = load_predictions(MODEL1_DIR, fold_num)
            df2 = load_predictions(MODEL2_DIR, fold_num)

            # Compare and find regression cases
            regression_cases = compare_predictions(df1, df2, fold_num)

            # Save individual fold results
            output_file = OUTPUT_DIR / f"diff_split_{fold_num}.csv"
            regression_cases.to_csv(output_file, index=False)

            # Collect statistics
            total_samples = len(df1)
            num_regressions = len(regression_cases)
            regression_pct = (num_regressions / total_samples * 100) if total_samples > 0 else 0

            fold_stats.append({
                'fold': fold_num,
                'total_samples': total_samples,
                'regressions': num_regressions,
                'regression_pct': regression_pct
            })

            print(f"  Total samples:     {total_samples}")
            print(f"  Regression cases:  {num_regressions} ({regression_pct:.2f}%)")
            print(f"  Saved to: {output_file}")
            print()

            # Collect all regression cases
            all_regression_cases.append(regression_cases)

        except Exception as e:
            print(f"  Error processing fold {fold_num}: {e}")
            print()

    # Consolidate all regression cases
    if all_regression_cases:
        all_regressions_df = pd.concat(all_regression_cases, ignore_index=True)
        consolidated_file = OUTPUT_DIR / "diff_all_folds.csv"
        all_regressions_df.to_csv(consolidated_file, index=False)

        print("=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print()

        # Print fold-by-fold summary
        print("Fold-by-fold breakdown:")
        print(f"{'Fold':<6} {'Total Samples':<15} {'Regressions':<15} {'Regression %':<15}")
        print("-" * 51)
        for stats in fold_stats:
            print(f"{stats['fold']:<6} {stats['total_samples']:<15} "
                  f"{stats['regressions']:<15} {stats['regression_pct']:<15.2f}")

        print()

        # Print overall summary
        total_samples = sum(s['total_samples'] for s in fold_stats)
        total_regressions = sum(s['regressions'] for s in fold_stats)
        overall_regression_pct = (total_regressions / total_samples * 100) if total_samples > 0 else 0

        print("Overall:")
        print(f"  Total samples across all folds:  {total_samples}")
        print(f"  Total regression cases:           {total_regressions}")
        print(f"  Overall regression rate:          {overall_regression_pct:.2f}%")
        print()
        print(f"All regression cases saved to: {consolidated_file}")
        print()
    else:
        print("No regression cases found.")


if __name__ == "__main__":
    main()
