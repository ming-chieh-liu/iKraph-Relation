#!/usr/bin/env python3
"""
Comprehensive statistical analysis comparing Model 1 (600) vs Model 2 (600+80updated).
Analyzes how annotation updates affect model predictions across label-changed and label-unchanged cases.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

transform_method = "typed_entity_marker_punct"

# Model directories
MODEL1_DIR = Path(__file__).parent / f"fine_tuned_models_litcoin_600_{transform_method}"
MODEL2_DIR = Path(__file__).parent / f"fine_tuned_models_litcoin_600_80updated_{transform_method}"

# Output directory
OUTPUT_DIR = Path(__file__).parent / "regression_analysis"

# Data directory for 80 updated articles
UPDATED_ARTICLES_JSON = Path(__file__).parent.parent.parent / "data_multi_sentence" / "litcoin_80updated" / "processed" / "multi_sentence_all.json"

# Fold patterns
FOLD_PATTERN = "NEWDATA_triplet_{}_False_split_{}_16_3e-05_roberta_ls0.02"

# Label mapping
LABEL_LIST = ["NOT", "Association", "Positive_Correlation", "Negative_Correlation",
              "Bind", "Cotreatment", "Comparison", "Drug_Interaction", "Conversion"]
LABEL_DICT = {idx: val for idx, val in enumerate(LABEL_LIST)}


def load_predictions(model_dir: Path, fold_num: int) -> pd.DataFrame:
    """Load predictions from a specific model fold."""
    fold_name = FOLD_PATTERN.format(transform_method, fold_num)
    predictions_path = model_dir / fold_name / "eval_predictions.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    df = pd.read_csv(predictions_path, index_col=0)
    return df


def load_updated_article_ids() -> Set[str]:
    """Load article IDs from the 80 updated articles JSON file."""
    with open(UPDATED_ARTICLES_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract unique abstract_id values
    article_ids = set(item['abstract_id'] for item in data)
    return article_ids


def compute_metrics_for_fold(df: pd.DataFrame) -> Dict:
    """
    Compute F1, precision, recall, and accuracy for a single fold.
    Excludes NOT class (label 0) from F1, precision, and recall calculations.

    Args:
        df: DataFrame with 'label' (true) and 'predicted_class' columns

    Returns:
        Dictionary with metrics and sample count
    """
    # Convert predicted_class (string) to numeric labels
    label_to_idx = {label: idx for idx, label in enumerate(LABEL_LIST)}
    y_true = df['label'].values
    y_pred = df['predicted_class'].map(label_to_idx).values

    # Filter out NOT class (label 0) for F1, precision, recall
    non_not_mask = y_true != 0
    y_true_filtered = y_true[non_not_mask]
    y_pred_filtered = y_pred[non_not_mask]

    # Compute metrics (weighted average for multi-class, excluding NOT)
    metrics = {
        'eval_samples': len(df),
        'eval_samples_non_not': int(non_not_mask.sum()),
        'eval_f1': f1_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0),
        'eval_precision': precision_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0),
        'eval_recall': recall_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0),
        'eval_accuracy': accuracy_score(y_true, y_pred),  # Keep accuracy on all instances
    }

    return metrics


def filter_predictions_by_articles(df: pd.DataFrame, article_ids: Set[str]) -> pd.DataFrame:
    """
    Filter predictions DataFrame to only include rows from specified article IDs.

    Args:
        df: DataFrame with 'abstract_id' column
        article_ids: Set of article IDs to keep

    Returns:
        Filtered DataFrame
    """
    # Convert abstract_id to string for comparison
    df_filtered = df[df['abstract_id'].astype(str).isin(article_ids)].copy()
    return df_filtered


def analyze_single_fold(df1: pd.DataFrame, df2: pd.DataFrame, fold_num: int, updated_article_ids: Set[str] = None) -> Dict:
    """
    Analyze a single fold, comparing Model 1 vs Model 2.

    Args:
        df1: Model 1 predictions
        df2: Model 2 predictions
        fold_num: Fold number
        updated_article_ids: Set of article IDs for the 80 updated articles (optional)

    Returns:
        Dictionary containing all statistics for this fold
    """
    assert len(df1) == len(df2), f"Fold {fold_num}: Row count mismatch!"

    # Initialize counters
    stats = {
        'fold': fold_num,
        'total_instances': len(df1),
        'label_changed': {
            'count': 0,
            'model1_correct_new_label': 0,
            'model1_predicts_old_label': 0,
            'model2_correct_new_label': 0,
            'model2_predicts_old_label': 0,
            'prediction_changed': 0,
            'prediction_unchanged': 0,
        },
        'label_unchanged': {
            'count': 0,
            'prediction_changed': 0,
            'prediction_unchanged': 0,
            'wrong_to_correct': 0,
            'correct_to_wrong': 0,
            'wrong_to_wrong': 0,
            'correct_to_correct': 0,
        },
        'confusion_matrix': defaultdict(lambda: defaultdict(int)),
        'per_class_stats': defaultdict(lambda: {
            'total': 0,
            'correct_to_wrong': 0,
            'wrong_to_correct': 0,
        }),
    }

    # Add tracking for 80 updated articles if provided
    if updated_article_ids is not None:
        stats['label_unchanged_80articles'] = {
            'wrong_to_correct': 0,
            'correct_to_wrong': 0,
            'wrong_to_wrong': 0,
        }

    for idx in range(len(df1)):
        row1 = df1.iloc[idx]
        row2 = df2.iloc[idx]

        # Verify alignment
        assert row1['abstract_id'] == row2['abstract_id'], f"Fold {fold_num}, row {idx}: ID mismatch!"
        assert row1['relation_id'] == row2['relation_id'], f"Fold {fold_num}, row {idx}: relation_id mismatch!"

        # Get labels and predictions
        label1 = row1['label']  # Model 1 ground truth (numeric)
        label2 = row2['label']  # Model 2 ground truth (numeric)
        pred1 = row1['predicted_class']  # Model 1 prediction (string)
        pred2 = row2['predicted_class']  # Model 2 prediction (string)

        # Convert numeric labels to class names for comparison
        label1_class = LABEL_DICT[label1]
        label2_class = LABEL_DICT[label2]

        # Check if label changed
        if label1 != label2:
            # LABEL CHANGED CASE
            stats['label_changed']['count'] += 1

            # Does Model 1 predict the new label correctly?
            if pred1 == label2_class:
                stats['label_changed']['model1_correct_new_label'] += 1

            # Does Model 1 predict the old label?
            if pred1 == label1_class:
                stats['label_changed']['model1_predicts_old_label'] += 1

            # Does Model 2 predict the new label correctly?
            if pred2 == label2_class:
                stats['label_changed']['model2_correct_new_label'] += 1

            # Does Model 2 still predict the old label?
            if pred2 == label1_class:
                stats['label_changed']['model2_predicts_old_label'] += 1

            # Did prediction change?
            if pred1 != pred2:
                stats['label_changed']['prediction_changed'] += 1
            else:
                stats['label_changed']['prediction_unchanged'] += 1

        else:
            # LABEL UNCHANGED CASE
            stats['label_unchanged']['count'] += 1

            # Track per-class statistics
            stats['per_class_stats'][label1_class]['total'] += 1

            # Check correctness
            model1_correct = (pred1 == label1_class)
            model2_correct = (pred2 == label2_class)

            # Check if this instance is from the 80 updated articles
            is_from_80_articles = (updated_article_ids is not None and
                                   str(row1['abstract_id']) in updated_article_ids)

            # Did prediction change?
            if pred1 != pred2:
                stats['label_unchanged']['prediction_changed'] += 1

                # Categorize the change
                if not model1_correct and model2_correct:
                    stats['label_unchanged']['wrong_to_correct'] += 1
                    stats['per_class_stats'][label1_class]['wrong_to_correct'] += 1

                    # Track for 80 updated articles
                    if is_from_80_articles:
                        stats['label_unchanged_80articles']['wrong_to_correct'] += 1

                elif model1_correct and not model2_correct:
                    stats['label_unchanged']['correct_to_wrong'] += 1
                    stats['per_class_stats'][label1_class]['correct_to_wrong'] += 1

                    # Record confusion for regression cases
                    stats['confusion_matrix'][label1_class][pred2] += 1

                    # Track for 80 updated articles
                    if is_from_80_articles:
                        stats['label_unchanged_80articles']['correct_to_wrong'] += 1

                else:  # both wrong but different
                    stats['label_unchanged']['wrong_to_wrong'] += 1

                    # Track for 80 updated articles
                    if is_from_80_articles:
                        stats['label_unchanged_80articles']['wrong_to_wrong'] += 1

            else:  # prediction unchanged
                stats['label_unchanged']['prediction_unchanged'] += 1

                if model1_correct and model2_correct:
                    stats['label_unchanged']['correct_to_correct'] += 1
                # Note: wrong_to_wrong with same prediction is counted in unchanged

    return stats


def aggregate_stats(fold_stats: List[Dict]) -> Dict:
    """Aggregate statistics across all folds."""
    agg = {
        'total_instances': 0,
        'label_changed': {
            'count': 0,
            'model1_correct_new_label': 0,
            'model1_predicts_old_label': 0,
            'model2_correct_new_label': 0,
            'model2_predicts_old_label': 0,
            'prediction_changed': 0,
            'prediction_unchanged': 0,
        },
        'label_unchanged': {
            'count': 0,
            'prediction_changed': 0,
            'prediction_unchanged': 0,
            'wrong_to_correct': 0,
            'correct_to_wrong': 0,
            'wrong_to_wrong': 0,
            'correct_to_correct': 0,
        },
        'confusion_matrix': defaultdict(lambda: defaultdict(int)),
        'per_class_stats': defaultdict(lambda: {
            'total': 0,
            'correct_to_wrong': 0,
            'wrong_to_correct': 0,
        }),
    }

    # Check if any fold has 80 articles stats
    has_80_articles = any('label_unchanged_80articles' in fs for fs in fold_stats)
    if has_80_articles:
        agg['label_unchanged_80articles'] = {
            'wrong_to_correct': 0,
            'correct_to_wrong': 0,
            'wrong_to_wrong': 0,
        }

    for fold_stat in fold_stats:
        agg['total_instances'] += fold_stat['total_instances']

        # Aggregate label_changed
        for key in agg['label_changed']:
            agg['label_changed'][key] += fold_stat['label_changed'][key]

        # Aggregate label_unchanged
        for key in agg['label_unchanged']:
            agg['label_unchanged'][key] += fold_stat['label_unchanged'][key]

        # Aggregate confusion matrix
        for true_label, preds in fold_stat['confusion_matrix'].items():
            for pred_label, count in preds.items():
                agg['confusion_matrix'][true_label][pred_label] += count

        # Aggregate per-class stats
        for class_name, class_stats in fold_stat['per_class_stats'].items():
            agg['per_class_stats'][class_name]['total'] += class_stats['total']
            agg['per_class_stats'][class_name]['correct_to_wrong'] += class_stats['correct_to_wrong']
            agg['per_class_stats'][class_name]['wrong_to_correct'] += class_stats['wrong_to_correct']

        # Aggregate 80 articles stats if present
        if 'label_unchanged_80articles' in fold_stat:
            for key in agg['label_unchanged_80articles']:
                agg['label_unchanged_80articles'][key] += fold_stat['label_unchanged_80articles'][key]

    return agg


def analyze_80_updated_articles() -> Dict:
    """
    Analyze Model 2's performance on the 80 updated articles only.

    Returns:
        Dictionary containing aggregated metrics and per-fold results
    """
    print("=" * 80)
    print("ANALYZING 80 UPDATED ARTICLES")
    print("=" * 80)
    print()

    # Load article IDs
    print("Loading article IDs from litcoin_80updated...")
    article_ids = load_updated_article_ids()
    print(f"Found {len(article_ids)} unique article IDs in the 80 updated articles dataset")
    print()

    # Analyze each fold
    fold_results = []
    total_weighted_sum = {
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'accuracy': 0.0,
    }
    total_samples = 0
    total_samples_non_not = 0

    for fold_num in range(5):
        print(f"Processing fold {fold_num}...")

        # Load Model 2 predictions
        df2 = load_predictions(MODEL2_DIR, fold_num)

        # Filter for 80 updated articles
        df_filtered = filter_predictions_by_articles(df2, article_ids)

        # Compute metrics
        metrics = compute_metrics_for_fold(df_filtered)
        fold_results.append({
            'fold': fold_num,
            **metrics
        })

        # Update weighted sums (use non-NOT samples for F1/P/R weighting)
        samples = metrics['eval_samples']
        samples_non_not = metrics['eval_samples_non_not']
        total_samples += samples
        total_samples_non_not += samples_non_not
        total_weighted_sum['f1'] += metrics['eval_f1'] * samples_non_not
        total_weighted_sum['precision'] += metrics['eval_precision'] * samples_non_not
        total_weighted_sum['recall'] += metrics['eval_recall'] * samples_non_not
        total_weighted_sum['accuracy'] += metrics['eval_accuracy'] * samples

        print(f"  Instances from 80 articles: {samples} (non-NOT: {samples_non_not})")
        print(f"  F1:        {metrics['eval_f1']:.4f}")
        print(f"  Precision: {metrics['eval_precision']:.4f}")
        print(f"  Recall:    {metrics['eval_recall']:.4f}")
        print(f"  Accuracy:  {metrics['eval_accuracy']:.4f}")
        print()

    # Compute weighted averages
    weighted_averages = {
        'total_samples': total_samples,
        'total_samples_non_not': total_samples_non_not,
        'f1': total_weighted_sum['f1'] / total_samples_non_not if total_samples_non_not > 0 else 0.0,
        'precision': total_weighted_sum['precision'] / total_samples_non_not if total_samples_non_not > 0 else 0.0,
        'recall': total_weighted_sum['recall'] / total_samples_non_not if total_samples_non_not > 0 else 0.0,
        'accuracy': total_weighted_sum['accuracy'] / total_samples if total_samples > 0 else 0.0,
    }

    # Print summary
    print("-" * 80)
    print("WEIGHTED AVERAGES (80 Updated Articles Only - Excluding NOT class)")
    print("-" * 80)
    print(f"Total instances:     {weighted_averages['total_samples']}")
    print(f"Non-NOT instances:   {weighted_averages['total_samples_non_not']}")
    print(f"F1:                  {weighted_averages['f1']:.4f}")
    print(f"Precision:           {weighted_averages['precision']:.4f}")
    print(f"Recall:              {weighted_averages['recall']:.4f}")
    print(f"Accuracy (all):      {weighted_averages['accuracy']:.4f}")
    print()
    print("=" * 80)
    print()

    return {
        'article_count': len(article_ids),
        'weighted_averages': weighted_averages,
        'fold_results': fold_results,
    }


def print_statistics(agg_stats: Dict, fold_stats: List[Dict]):
    """Print formatted statistics to console."""
    print("=" * 80)
    print("COMPREHENSIVE REGRESSION STATISTICS")
    print("Model 1: litcoin_600 | Model 2: litcoin_600_80updated")
    print("=" * 80)
    print()

    total = agg_stats['total_instances']

    # LABEL CHANGED SECTION
    print("-" * 80)
    print("1. LABEL CHANGED ANALYSIS")
    print("-" * 80)

    lc = agg_stats['label_changed']
    print(f"Total instances with label changes:  {lc['count']} ({lc['count']/total*100:.2f}%)")
    print()

    if lc['count'] > 0:
        print("1a) Model predictions on NEW labels:")
        print(f"    Model 1 correct:                    {lc['model1_correct_new_label']} ({lc['model1_correct_new_label']/lc['count']*100:.2f}%)")
        print(f"    Model 2 correct:                    {lc['model2_correct_new_label']} ({lc['model2_correct_new_label']/lc['count']*100:.2f}%)")
        print()

        print("1b) Model predictions on OLD labels:")
        print(f"    Model 1 predicts old:               {lc['model1_predicts_old_label']} ({lc['model1_predicts_old_label']/lc['count']*100:.2f}%)")
        print(f"    Model 2 predicts old:               {lc['model2_predicts_old_label']} ({lc['model2_predicts_old_label']/lc['count']*100:.2f}%)")
        print()

        print("1c) Did Model 2 predictions change from Model 1?")
        print(f"    Prediction changed:                 {lc['prediction_changed']} ({lc['prediction_changed']/lc['count']*100:.2f}%)")
        print(f"    Prediction unchanged:               {lc['prediction_unchanged']} ({lc['prediction_unchanged']/lc['count']*100:.2f}%)")
        print()

    # LABEL UNCHANGED SECTION
    print("-" * 80)
    print("2. LABEL UNCHANGED ANALYSIS")
    print("-" * 80)

    lu = agg_stats['label_unchanged']
    print(f"Total instances with unchanged labels: {lu['count']} ({lu['count']/total*100:.2f}%)")
    print()

    if lu['count'] > 0:
        print("2a) Did predictions change?")
        print(f"    Prediction changed:                 {lu['prediction_changed']} ({lu['prediction_changed']/lu['count']*100:.2f}%)")
        print(f"    Prediction unchanged:               {lu['prediction_unchanged']} ({lu['prediction_unchanged']/lu['count']*100:.2f}%)")
        print()

        print("2b) Breakdown of prediction changes:")
        print(f"    Wrong → Correct (improvement):      {lu['wrong_to_correct']} ({lu['wrong_to_correct']/lu['count']*100:.2f}%)")
        print(f"    Correct → Wrong (regression):       {lu['correct_to_wrong']} ({lu['correct_to_wrong']/lu['count']*100:.2f}%)")
        print(f"    Wrong → Wrong (different):          {lu['wrong_to_wrong']} ({lu['wrong_to_wrong']/lu['count']*100:.2f}%)")
        print(f"    Correct → Correct (stable):         {lu['correct_to_correct']} ({lu['correct_to_correct']/lu['count']*100:.2f}%)")
        print()

        # Net impact
        net_improvement = lu['wrong_to_correct'] - lu['correct_to_wrong']
        print(f"    NET IMPACT: {'+' if net_improvement >= 0 else ''}{net_improvement} "
              f"({'+' if net_improvement >= 0 else ''}{net_improvement/lu['count']*100:.2f}%)")
        print()

        # 80 updated articles breakdown (if available)
        if 'label_unchanged_80articles' in agg_stats:
            lu80 = agg_stats['label_unchanged_80articles']
            print("2c) Among the 80 updated articles (label unchanged cases):")
            print(f"    Wrong → Correct (improvement):      {lu80['wrong_to_correct']}")
            print(f"    Correct → Wrong (regression):       {lu80['correct_to_wrong']}")
            print(f"    Wrong → Wrong (different):          {lu80['wrong_to_wrong']}")
            net_80 = lu80['wrong_to_correct'] - lu80['correct_to_wrong']
            print(f"    NET IMPACT: {'+' if net_80 >= 0 else ''}{net_80}")
            print()

    # CONFUSION MATRIX FOR REGRESSIONS
    print("-" * 80)
    print("3. CONFUSION PATTERNS (Correct → Wrong cases only)")
    print("-" * 80)

    cm = agg_stats['confusion_matrix']
    if cm:
        print(f"{'True Label':<30} {'Predicted As':<30} {'Count':<10}")
        print("-" * 70)
        for true_label in sorted(cm.keys()):
            for pred_label, count in sorted(cm[true_label].items(), key=lambda x: -x[1]):
                print(f"{true_label:<30} {pred_label:<30} {count:<10}")
        print()
    else:
        print("No regression cases found.")
        print()

    # PER-CLASS BREAKDOWN
    print("-" * 80)
    print("4. PER-CLASS BREAKDOWN (Label Unchanged Cases)")
    print("-" * 80)

    pcs = agg_stats['per_class_stats']
    if pcs:
        print(f"{'Class':<30} {'Total':<10} {'Correct→Wrong':<15} {'Wrong→Correct':<15} {'Net':<10}")
        print("-" * 80)
        for class_name in LABEL_LIST:
            if class_name in pcs:
                stats = pcs[class_name]
                net = stats['wrong_to_correct'] - stats['correct_to_wrong']
                net_str = f"{'+' if net >= 0 else ''}{net}"
                print(f"{class_name:<30} {stats['total']:<10} {stats['correct_to_wrong']:<15} "
                      f"{stats['wrong_to_correct']:<15} {net_str:<10}")
        print()

    # FOLD-BY-FOLD SUMMARY
    print("-" * 80)
    print("5. FOLD-BY-FOLD SUMMARY")
    print("-" * 80)

    print(f"{'Fold':<6} {'Label Changed':<15} {'Regressions':<15} {'Improvements':<15} {'Net':<10}")
    print("-" * 61)
    for fold_stat in fold_stats:
        fold = fold_stat['fold']
        lc_count = fold_stat['label_changed']['count']
        regress = fold_stat['label_unchanged']['correct_to_wrong']
        improve = fold_stat['label_unchanged']['wrong_to_correct']
        net = improve - regress
        net_str = f"{'+' if net >= 0 else ''}{net}"
        print(f"{fold:<6} {lc_count:<15} {regress:<15} {improve:<15} {net_str:<10}")

    print()
    print("=" * 80)


def main():
    """Main function to analyze all folds."""
    print("Loading predictions from all folds...")
    print()

    # Load the 80 updated article IDs for tracking
    updated_article_ids = load_updated_article_ids()
    print(f"Loaded {len(updated_article_ids)} article IDs from the 80 updated articles dataset")
    print()

    fold_stats = []

    for fold_num in range(5):
        print(f"Processing fold {fold_num}...")
        try:
            df1 = load_predictions(MODEL1_DIR, fold_num)
            df2 = load_predictions(MODEL2_DIR, fold_num)

            stats = analyze_single_fold(df1, df2, fold_num, updated_article_ids)
            fold_stats.append(stats)

            print(f"  Fold {fold_num}: {stats['total_instances']} instances analyzed")
            print(f"    Label changed:   {stats['label_changed']['count']}")
            print(f"    Label unchanged: {stats['label_unchanged']['count']}")
            print(f"    Regressions:     {stats['label_unchanged']['correct_to_wrong']}")
            print(f"    Improvements:    {stats['label_unchanged']['wrong_to_correct']}")
            if 'label_unchanged_80articles' in stats:
                lu80 = stats['label_unchanged_80articles']
                print(f"    80 articles - Wrong→Correct: {lu80['wrong_to_correct']}, Correct→Wrong: {lu80['correct_to_wrong']}, Wrong→Wrong: {lu80['wrong_to_wrong']}")
            print()

        except Exception as e:
            print(f"  Error processing fold {fold_num}: {e}")
            print()

    # Aggregate across folds
    print("Aggregating statistics across all folds...")
    agg_stats = aggregate_stats(fold_stats)
    print()

    # Print comprehensive statistics
    print_statistics(agg_stats, fold_stats)

    # Analyze 80 updated articles
    updated_articles_stats = analyze_80_updated_articles()

    # Save to JSON
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / "detailed_stats.json"

    # Convert defaultdict to regular dict for JSON serialization
    output_data = {
        'aggregated_stats': {
            'total_instances': agg_stats['total_instances'],
            'label_changed': agg_stats['label_changed'],
            'label_unchanged': agg_stats['label_unchanged'],
            'confusion_matrix': {k: dict(v) for k, v in agg_stats['confusion_matrix'].items()},
            'per_class_stats': dict(agg_stats['per_class_stats']),
        },
        'fold_stats': fold_stats,
        'updated_articles_analysis': updated_articles_stats,
    }

    # Need to convert nested defaultdicts
    for fold_stat in output_data['fold_stats']:
        fold_stat['confusion_matrix'] = {k: dict(v) for k, v in fold_stat['confusion_matrix'].items()}
        fold_stat['per_class_stats'] = dict(fold_stat['per_class_stats'])

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Detailed statistics saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
