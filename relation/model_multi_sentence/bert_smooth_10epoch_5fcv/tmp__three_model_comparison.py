#!/usr/bin/env python3
"""
Three-Model Comparison Analysis Script.

Compares predictions across three models to understand the impact of adding training data:
1. Model 1: litcoin_600 (baseline)
2. Model 2: litcoin_600_80updated (+ label updates)
3. Model 3: litcoin_600_80updated_pubmed_200incomplete (+ new data)

Performs four analyses:
1. Label Change Predictions Across 3 Models
2. pubmed_200incomplete Confusion Matrix
3. Prediction Shifts Model 1 → Model 2 (Unchanged Labels) - impact of label updates only
4. Prediction Shifts Model 1 → Model 3 (Unchanged Labels) - impact of label updates + new data
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

# Constants
TRANSFORM_METHOD = "typed_entity_marker_punct"

# Model directories
MODEL1_DIR = Path(__file__).parent / f"fine_tuned_models_litcoin_600_{TRANSFORM_METHOD}"
MODEL2_DIR = Path(__file__).parent / f"fine_tuned_models_litcoin_600_80updated_{TRANSFORM_METHOD}"
MODEL3_DIR = Path(__file__).parent / f"fine_tuned_models_litcoin_600_80updated_pubmed_200incomplete_{TRANSFORM_METHOD}"

# Output directory
OUTPUT_DIR = Path(__file__).parent / "three_model_comparison_analysis"

# Data directories
LITCOIN_80_ARTICLES_JSON = Path(__file__).parent.parent.parent / "data_multi_sentence" / "litcoin_80updated" / "processed" / "multi_sentence_all.json"
PUBMED_200_ABSTRACTS_JSON = Path(__file__).parent.parent.parent / "data_multi_sentence" / "pubmed_200incomplete" / "multi_sentence_split_pubmed_200incomplete" / "all_abstracts.json"

# Fold pattern
FOLD_PATTERN = "NEWDATA_triplet_{}_False_split_{}_16_3e-05_roberta_ls0.02"

# Label mapping
LABEL_LIST = ["NOT", "Association", "Positive_Correlation", "Negative_Correlation",
              "Bind", "Cotreatment", "Comparison", "Drug_Interaction", "Conversion"]
LABEL_DICT = {idx: val for idx, val in enumerate(LABEL_LIST)}
LABEL_TO_IDX = {val: idx for idx, val in enumerate(LABEL_LIST)}


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_predictions(model_dir: Path, fold_num: int) -> pd.DataFrame:
    """Load predictions from a specific model fold."""
    fold_name = FOLD_PATTERN.format(TRANSFORM_METHOD, fold_num)
    predictions_path = model_dir / fold_name / "eval_predictions.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    df = pd.read_csv(predictions_path, index_col=0)
    return df


def load_litcoin_80_article_ids() -> Set[str]:
    """Load article IDs from the 80 updated articles JSON file."""
    with open(LITCOIN_80_ARTICLES_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    article_ids = set(str(item['abstract_id']) for item in data)
    return article_ids


def load_pubmed_200_pmids() -> Set[str]:
    """Load PMIDs from the pubmed_200incomplete all_abstracts.json file."""
    with open(PUBMED_200_ABSTRACTS_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract from "all_pmids" key
    pmids = set(str(pmid) for pmid in data.get("all_pmids", []))
    return pmids


# ============================================================================
# Analysis 1: Label Change Predictions Across 3 Models
# ============================================================================

def analyze_label_changes_across_models(fold_num: int) -> Dict:
    """
    For instances where ground truth labels changed between Model 1 and Model 2,
    track what each of the 3 models predicts.

    Note: Model 3 has additional instances from pubmed_200. We only analyze
    instances that exist in all 3 models (join on relation_id).
    """
    df1 = load_predictions(MODEL1_DIR, fold_num)
    df2 = load_predictions(MODEL2_DIR, fold_num)
    df3 = load_predictions(MODEL3_DIR, fold_num)

    # Model 1 and Model 2 should have the same instances
    assert len(df1) == len(df2), f"Fold {fold_num}: Model 1 and Model 2 row count mismatch!"

    # Join Model 3 on relation_id to get only common instances
    df3_common = df3[df3['relation_id'].isin(df1['relation_id'])].copy()
    df3_common = df3_common.set_index('relation_id')

    results = {
        'fold': fold_num,
        'total_instances': len(df1),
        'label_changed_count': 0,
        'model1': {'predicts_old': 0, 'predicts_new': 0, 'predicts_other': 0},
        'model2': {'predicts_old': 0, 'predicts_new': 0, 'predicts_other': 0},
        'model3': {'predicts_old': 0, 'predicts_new': 0, 'predicts_other': 0},
        'details': []  # Store individual cases for debugging if needed
    }

    for idx in range(len(df1)):
        row1, row2 = df1.iloc[idx], df2.iloc[idx]

        # Verify alignment between Model 1 and Model 2
        assert row1['relation_id'] == row2['relation_id'], \
            f"Fold {fold_num}, row {idx}: relation_id mismatch between Model 1 and Model 2!"

        relation_id = row1['relation_id']

        # Get Model 3 prediction for this relation_id
        if relation_id not in df3_common.index:
            continue  # Skip if not in Model 3

        row3 = df3_common.loc[relation_id]

        # Get labels (numeric) and predictions (string)
        label1 = row1['label']  # Model 1 ground truth (old label)
        label2 = row2['label']  # Model 2 ground truth (new label)

        # Check if label changed
        if label1 != label2:
            results['label_changed_count'] += 1

            old_label_class = LABEL_DICT[label1]
            new_label_class = LABEL_DICT[label2]

            # Check each model's prediction
            for model_key, row in [('model1', row1), ('model2', row2), ('model3', row3)]:
                pred = row['predicted_class']
                if pred == old_label_class:
                    results[model_key]['predicts_old'] += 1
                elif pred == new_label_class:
                    results[model_key]['predicts_new'] += 1
                else:
                    results[model_key]['predicts_other'] += 1

    return results


# ============================================================================
# Analysis 2: pubmed_200incomplete Confusion Matrix
# ============================================================================

def generate_pubmed_200_confusion_matrix(fold_num: int) -> Dict:
    """
    Filter Model 3 predictions to only pubmed_200 articles and generate
    full 9x9 confusion matrix with metrics.
    """
    df3 = load_predictions(MODEL3_DIR, fold_num)
    pubmed_200_pmids = load_pubmed_200_pmids()

    # Filter to pubmed_200 articles
    df_filtered = df3[df3['abstract_id'].astype(str).isin(pubmed_200_pmids)].copy()

    if len(df_filtered) == 0:
        return {
            'fold': fold_num,
            'total_instances': 0,
            'confusion_matrix': {},
            'metrics': {}
        }

    # Get true and predicted labels
    y_true = df_filtered['label'].values
    y_pred = df_filtered['predicted_class'].map(LABEL_TO_IDX).values

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_LIST))))

    # Convert to dict for JSON serialization
    cm_dict = {}
    for i, true_label in enumerate(LABEL_LIST):
        cm_dict[true_label] = {}
        for j, pred_label in enumerate(LABEL_LIST):
            cm_dict[true_label][pred_label] = int(cm[i, j])

    # Calculate metrics
    # Exclude NOT class (label 0) for F1/precision/recall
    non_not_mask = y_true != 0
    y_true_filtered = y_true[non_not_mask]
    y_pred_filtered = y_pred[non_not_mask]

    metrics = {
        'total_samples': len(df_filtered),
        'total_samples_non_not': int(non_not_mask.sum()),
        'accuracy': float(accuracy_score(y_true, y_pred)),
    }

    if len(y_true_filtered) > 0:
        metrics['f1'] = float(f1_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0))
        metrics['precision'] = float(precision_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0))
        metrics['recall'] = float(recall_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0))
    else:
        metrics['f1'] = 0.0
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0

    return {
        'fold': fold_num,
        'total_instances': len(df_filtered),
        'unique_pmids': len(df_filtered['abstract_id'].astype(str).unique()),
        'confusion_matrix': cm_dict,
        'metrics': metrics
    }


# ============================================================================
# Analysis 3: Prediction Shifts Model 1 → Model 2 (Unchanged Labels)
# ============================================================================

def analyze_prediction_shifts_m1_to_m2(fold_num: int) -> Dict:
    """
    For instances where labels did NOT change (old data range),
    track prediction shifts from Model 1 to Model 2.
    This shows the impact of label updates only.

    Note: wrong_to_wrong only counts cases where prediction changed but both are still wrong.
    """
    df1 = load_predictions(MODEL1_DIR, fold_num)
    df2 = load_predictions(MODEL2_DIR, fold_num)

    # Model 1 and Model 2 should have the same instances
    assert len(df1) == len(df2), f"Fold {fold_num}: Model 1 and Model 2 row count mismatch!"

    results = {
        'fold': fold_num,
        'total_instances': len(df1),
        'label_unchanged_count': 0,
        'prediction_changed': 0,
        'prediction_unchanged': 0,
        'wrong_to_correct': 0,
        'correct_to_wrong': 0,
        'wrong_to_wrong': 0,
        'correct_to_correct': 0,
        'per_class_stats': defaultdict(lambda: {
            'total': 0,
            'wrong_to_correct': 0,
            'correct_to_wrong': 0,
            'wrong_to_wrong': 0,
            'correct_to_correct': 0,
        })
    }

    for idx in range(len(df1)):
        row1, row2 = df1.iloc[idx], df2.iloc[idx]

        # Verify alignment
        assert row1['relation_id'] == row2['relation_id'], \
            f"Fold {fold_num}, row {idx}: relation_id mismatch!"

        # Get labels
        label1 = row1['label']  # Model 1 ground truth
        label2 = row2['label']  # Model 2 ground truth

        # Only process unchanged labels
        if label1 == label2:
            results['label_unchanged_count'] += 1

            true_label_class = LABEL_DICT[label1]
            pred1 = row1['predicted_class']
            pred2 = row2['predicted_class']

            model1_correct = (pred1 == true_label_class)
            model2_correct = (pred2 == true_label_class)

            results['per_class_stats'][true_label_class]['total'] += 1

            # Did prediction change?
            if pred1 != pred2:
                results['prediction_changed'] += 1

                # Categorize the change
                if not model1_correct and model2_correct:
                    results['wrong_to_correct'] += 1
                    results['per_class_stats'][true_label_class]['wrong_to_correct'] += 1
                elif model1_correct and not model2_correct:
                    results['correct_to_wrong'] += 1
                    results['per_class_stats'][true_label_class]['correct_to_wrong'] += 1
                else:  # both wrong but different predictions
                    results['wrong_to_wrong'] += 1
                    results['per_class_stats'][true_label_class]['wrong_to_wrong'] += 1
            else:
                results['prediction_unchanged'] += 1

                if model1_correct and model2_correct:
                    results['correct_to_correct'] += 1
                    results['per_class_stats'][true_label_class]['correct_to_correct'] += 1
                # Note: wrong_to_wrong with same prediction is counted in prediction_unchanged

    return results


# ============================================================================
# Analysis 4: Prediction Shifts Model 1 → Model 3 (Unchanged Labels)
# ============================================================================

def analyze_prediction_shifts_m1_to_m3(fold_num: int) -> Dict:
    """
    For instances where labels did NOT change (old data range),
    track prediction shifts from Model 1 to Model 3.
    This shows the combined impact of label updates + new data.

    Note: Model 3 has additional instances from pubmed_200. We only analyze
    instances that exist in all 3 models (join on relation_id).
    Note: wrong_to_wrong only counts cases where prediction changed but both are still wrong.
    """
    df1 = load_predictions(MODEL1_DIR, fold_num)
    df2 = load_predictions(MODEL2_DIR, fold_num)
    df3 = load_predictions(MODEL3_DIR, fold_num)

    # Model 1 and Model 2 should have the same instances
    assert len(df1) == len(df2), f"Fold {fold_num}: Model 1 and Model 2 row count mismatch!"

    # Join Model 3 on relation_id to get only common instances
    df3_common = df3[df3['relation_id'].isin(df1['relation_id'])].copy()
    df3_common = df3_common.set_index('relation_id')

    results = {
        'fold': fold_num,
        'total_instances': len(df1),
        'label_unchanged_count': 0,
        'prediction_changed': 0,
        'prediction_unchanged': 0,
        'wrong_to_correct': 0,
        'correct_to_wrong': 0,
        'wrong_to_wrong': 0,
        'correct_to_correct': 0,
        'per_class_stats': defaultdict(lambda: {
            'total': 0,
            'wrong_to_correct': 0,
            'correct_to_wrong': 0,
            'wrong_to_wrong': 0,
            'correct_to_correct': 0,
        })
    }

    for idx in range(len(df1)):
        row1, row2 = df1.iloc[idx], df2.iloc[idx]

        # Verify alignment between Model 1 and Model 2
        assert row1['relation_id'] == row2['relation_id'], \
            f"Fold {fold_num}, row {idx}: relation_id mismatch between Model 1 and Model 2!"

        relation_id = row1['relation_id']

        # Get Model 3 prediction for this relation_id
        if relation_id not in df3_common.index:
            continue  # Skip if not in Model 3

        row3 = df3_common.loc[relation_id]

        # Get labels
        label1 = row1['label']  # Model 1 ground truth
        label2 = row2['label']  # Model 2 ground truth (same as Model 3)

        # Only process unchanged labels
        if label1 == label2:
            results['label_unchanged_count'] += 1

            true_label_class = LABEL_DICT[label1]
            pred1 = row1['predicted_class']
            pred3 = row3['predicted_class']

            model1_correct = (pred1 == true_label_class)
            model3_correct = (pred3 == true_label_class)

            results['per_class_stats'][true_label_class]['total'] += 1

            # Did prediction change?
            if pred1 != pred3:
                results['prediction_changed'] += 1

                # Categorize the change
                if not model1_correct and model3_correct:
                    results['wrong_to_correct'] += 1
                    results['per_class_stats'][true_label_class]['wrong_to_correct'] += 1
                elif model1_correct and not model3_correct:
                    results['correct_to_wrong'] += 1
                    results['per_class_stats'][true_label_class]['correct_to_wrong'] += 1
                else:  # both wrong but different predictions
                    results['wrong_to_wrong'] += 1
                    results['per_class_stats'][true_label_class]['wrong_to_wrong'] += 1
            else:
                results['prediction_unchanged'] += 1

                if model1_correct and model3_correct:
                    results['correct_to_correct'] += 1
                    results['per_class_stats'][true_label_class]['correct_to_correct'] += 1
                # Note: wrong_to_wrong with same prediction is counted in prediction_unchanged

    return results


# ============================================================================
# Aggregation Functions
# ============================================================================

def aggregate_label_change_results(fold_results: List[Dict]) -> Dict:
    """Aggregate Analysis 1 results across folds."""
    agg = {
        'total_instances': 0,
        'label_changed_count': 0,
        'model1': {'predicts_old': 0, 'predicts_new': 0, 'predicts_other': 0},
        'model2': {'predicts_old': 0, 'predicts_new': 0, 'predicts_other': 0},
        'model3': {'predicts_old': 0, 'predicts_new': 0, 'predicts_other': 0},
    }

    for fr in fold_results:
        agg['total_instances'] += fr['total_instances']
        agg['label_changed_count'] += fr['label_changed_count']
        for model_key in ['model1', 'model2', 'model3']:
            for pred_key in ['predicts_old', 'predicts_new', 'predicts_other']:
                agg[model_key][pred_key] += fr[model_key][pred_key]

    return agg


def aggregate_confusion_matrix_results(fold_results: List[Dict]) -> Dict:
    """Aggregate Analysis 2 results across folds."""
    # Aggregate confusion matrix
    agg_cm = {label: {pred: 0 for pred in LABEL_LIST} for label in LABEL_LIST}

    total_samples = 0
    total_samples_non_not = 0
    weighted_metrics = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0}

    for fr in fold_results:
        samples = fr['total_instances']
        samples_non_not = fr['metrics'].get('total_samples_non_not', 0)
        total_samples += samples
        total_samples_non_not += samples_non_not

        # Aggregate confusion matrix
        for true_label in LABEL_LIST:
            for pred_label in LABEL_LIST:
                agg_cm[true_label][pred_label] += fr['confusion_matrix'].get(true_label, {}).get(pred_label, 0)

        # Weight metrics by sample count
        if samples_non_not > 0:
            weighted_metrics['f1'] += fr['metrics'].get('f1', 0) * samples_non_not
            weighted_metrics['precision'] += fr['metrics'].get('precision', 0) * samples_non_not
            weighted_metrics['recall'] += fr['metrics'].get('recall', 0) * samples_non_not
        if samples > 0:
            weighted_metrics['accuracy'] += fr['metrics'].get('accuracy', 0) * samples

    # Compute weighted averages
    agg_metrics = {
        'total_samples': total_samples,
        'total_samples_non_not': total_samples_non_not,
    }
    if total_samples_non_not > 0:
        agg_metrics['f1'] = weighted_metrics['f1'] / total_samples_non_not
        agg_metrics['precision'] = weighted_metrics['precision'] / total_samples_non_not
        agg_metrics['recall'] = weighted_metrics['recall'] / total_samples_non_not
    else:
        agg_metrics['f1'] = 0.0
        agg_metrics['precision'] = 0.0
        agg_metrics['recall'] = 0.0

    if total_samples > 0:
        agg_metrics['accuracy'] = weighted_metrics['accuracy'] / total_samples
    else:
        agg_metrics['accuracy'] = 0.0

    return {
        'confusion_matrix': agg_cm,
        'metrics': agg_metrics,
    }


def aggregate_prediction_shift_results(fold_results: List[Dict]) -> Dict:
    """Aggregate prediction shift results across folds."""
    agg = {
        'total_instances': 0,
        'label_unchanged_count': 0,
        'prediction_changed': 0,
        'prediction_unchanged': 0,
        'wrong_to_correct': 0,
        'correct_to_wrong': 0,
        'wrong_to_wrong': 0,
        'correct_to_correct': 0,
        'per_class_stats': defaultdict(lambda: {
            'total': 0,
            'wrong_to_correct': 0,
            'correct_to_wrong': 0,
            'wrong_to_wrong': 0,
            'correct_to_correct': 0,
        })
    }

    for fr in fold_results:
        agg['total_instances'] += fr['total_instances']
        agg['label_unchanged_count'] += fr['label_unchanged_count']
        agg['prediction_changed'] += fr.get('prediction_changed', 0)
        agg['prediction_unchanged'] += fr.get('prediction_unchanged', 0)
        agg['wrong_to_correct'] += fr['wrong_to_correct']
        agg['correct_to_wrong'] += fr['correct_to_wrong']
        agg['wrong_to_wrong'] += fr['wrong_to_wrong']
        agg['correct_to_correct'] += fr['correct_to_correct']

        for class_name, stats in fr['per_class_stats'].items():
            for key in ['total', 'wrong_to_correct', 'correct_to_wrong', 'wrong_to_wrong', 'correct_to_correct']:
                agg['per_class_stats'][class_name][key] += stats[key]

    return agg


# ============================================================================
# Output Functions
# ============================================================================

def print_label_change_analysis(agg_results: Dict, fold_results: List[Dict]):
    """Print Analysis 1 results."""
    print("=" * 80)
    print("ANALYSIS 1: LABEL CHANGE PREDICTIONS ACROSS 3 MODELS")
    print("=" * 80)
    print()

    total_changed = agg_results['label_changed_count']
    print(f"Total instances with label changes: {total_changed}")
    print()

    if total_changed > 0:
        print("Prediction breakdown for label-changed instances:")
        print()
        print(f"{'Model':<15} {'Predicts OLD':<20} {'Predicts NEW':<20} {'Predicts OTHER':<20}")
        print("-" * 75)

        for model_key, model_name in [('model1', 'Model 1 (600)'),
                                       ('model2', 'Model 2 (+80upd)'),
                                       ('model3', 'Model 3 (+pubmed)')]:
            stats = agg_results[model_key]
            old_pct = stats['predicts_old'] / total_changed * 100
            new_pct = stats['predicts_new'] / total_changed * 100
            other_pct = stats['predicts_other'] / total_changed * 100
            print(f"{model_name:<15} {stats['predicts_old']:>6} ({old_pct:>5.1f}%)     "
                  f"{stats['predicts_new']:>6} ({new_pct:>5.1f}%)     "
                  f"{stats['predicts_other']:>6} ({other_pct:>5.1f}%)")

        print()
        print("Per-fold breakdown:")
        print(f"{'Fold':<6} {'Changed':<10} {'M1 New':<10} {'M2 New':<10} {'M3 New':<10}")
        print("-" * 46)
        for fr in fold_results:
            print(f"{fr['fold']:<6} {fr['label_changed_count']:<10} "
                  f"{fr['model1']['predicts_new']:<10} "
                  f"{fr['model2']['predicts_new']:<10} "
                  f"{fr['model3']['predicts_new']:<10}")

    print()


def print_confusion_matrix(agg_results: Dict, fold_results: List[Dict]):
    """Print Analysis 2 results."""
    print("=" * 80)
    print("ANALYSIS 2: PUBMED_200 CONFUSION MATRIX (Model 3)")
    print("=" * 80)
    print()

    metrics = agg_results['metrics']
    print(f"Total instances from pubmed_200: {metrics['total_samples']}")
    print(f"Non-NOT instances: {metrics['total_samples_non_not']}")
    print()

    print("Metrics (excluding NOT class for F1/P/R):")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print()

    # Print confusion matrix
    cm = agg_results['confusion_matrix']
    print("Confusion Matrix (rows=true, cols=predicted):")
    print()

    # Header
    header = f"{'True \\ Pred':<25}"
    for label in LABEL_LIST:
        header += f"{label[:8]:>10}"
    print(header)
    print("-" * (25 + 10 * len(LABEL_LIST)))

    # Rows
    for true_label in LABEL_LIST:
        row = f"{true_label:<25}"
        for pred_label in LABEL_LIST:
            count = cm[true_label][pred_label]
            row += f"{count:>10}"
        print(row)

    print()

    # Per-fold metrics
    print("Per-fold metrics:")
    print(f"{'Fold':<6} {'Samples':<10} {'F1':<10} {'Precision':<12} {'Recall':<10} {'Accuracy':<10}")
    print("-" * 58)
    for fr in fold_results:
        m = fr['metrics']
        print(f"{fr['fold']:<6} {fr['total_instances']:<10} {m.get('f1', 0):<10.4f} "
              f"{m.get('precision', 0):<12.4f} {m.get('recall', 0):<10.4f} {m.get('accuracy', 0):<10.4f}")

    print()


def print_prediction_shifts(agg_results: Dict, fold_results: List[Dict],
                            analysis_num: int, title: str):
    """Print prediction shift results."""
    print("=" * 80)
    print(f"ANALYSIS {analysis_num}: {title}")
    print("=" * 80)
    print()

    total = agg_results['label_unchanged_count']
    pred_changed = agg_results.get('prediction_changed', 0)
    pred_unchanged = agg_results.get('prediction_unchanged', 0)
    print(f"Total instances with unchanged labels: {total}")
    print()

    if total > 0:
        print("Did predictions change?")
        print(f"  Prediction changed:   {pred_changed:>6} ({pred_changed/total*100:.2f}%)")
        print(f"  Prediction unchanged: {pred_unchanged:>6} ({pred_unchanged/total*100:.2f}%)")
        print()

        w2c = agg_results['wrong_to_correct']
        c2w = agg_results['correct_to_wrong']
        w2w = agg_results['wrong_to_wrong']
        c2c = agg_results['correct_to_correct']
        net = w2c - c2w

        print("Breakdown of prediction changes:")
        print(f"  Wrong → Correct (improvement): {w2c:>6} ({w2c/total*100:.2f}%)")
        print(f"  Correct → Wrong (regression):  {c2w:>6} ({c2w/total*100:.2f}%)")
        print(f"  Wrong → Wrong (different):     {w2w:>6} ({w2w/total*100:.2f}%)")
        print(f"  Correct → Correct (stable):    {c2c:>6} ({c2c/total*100:.2f}%)")
        print()
        print(f"  NET IMPACT: {'+' if net >= 0 else ''}{net} ({'+' if net >= 0 else ''}{net/total*100:.2f}%)")
        print()

        # Per-class breakdown
        print("Per-class breakdown:")
        print(f"{'Class':<25} {'Total':<10} {'W→C':<10} {'C→W':<10} {'Net':<10}")
        print("-" * 65)

        pcs = agg_results['per_class_stats']
        for class_name in LABEL_LIST:
            if class_name in pcs:
                stats = pcs[class_name]
                class_net = stats['wrong_to_correct'] - stats['correct_to_wrong']
                net_str = f"{'+' if class_net >= 0 else ''}{class_net}"
                print(f"{class_name:<25} {stats['total']:<10} "
                      f"{stats['wrong_to_correct']:<10} {stats['correct_to_wrong']:<10} {net_str:<10}")

        print()

        # Per-fold summary
        print("Per-fold summary:")
        print(f"{'Fold':<6} {'Unchanged':<12} {'W→C':<10} {'C→W':<10} {'Net':<10}")
        print("-" * 48)
        for fr in fold_results:
            net_fold = fr['wrong_to_correct'] - fr['correct_to_wrong']
            net_str = f"{'+' if net_fold >= 0 else ''}{net_fold}"
            print(f"{fr['fold']:<6} {fr['label_unchanged_count']:<12} "
                  f"{fr['wrong_to_correct']:<10} {fr['correct_to_wrong']:<10} {net_str:<10}")

    print()


def save_results_to_json(all_results: Dict, output_path: Path):
    """Save all results to JSON file."""
    # Convert defaultdicts to regular dicts for JSON serialization
    def convert_defaultdict(obj):
        if isinstance(obj, defaultdict):
            return {k: convert_defaultdict(v) for k, v in obj.items()}
        elif isinstance(obj, dict):
            return {k: convert_defaultdict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_defaultdict(item) for item in obj]
        else:
            return obj

    serializable = convert_defaultdict(all_results)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all 4 analyses across 5 folds, aggregate, print, and save."""
    print("=" * 80)
    print("THREE-MODEL COMPARISON ANALYSIS")
    print("=" * 80)
    print()
    print(f"Model 1: {MODEL1_DIR.name}")
    print(f"Model 2: {MODEL2_DIR.name}")
    print(f"Model 3: {MODEL3_DIR.name}")
    print()

    # Load reference data
    print("Loading reference data...")
    litcoin_80_ids = load_litcoin_80_article_ids()
    pubmed_200_pmids = load_pubmed_200_pmids()
    print(f"  Litcoin 80 article IDs: {len(litcoin_80_ids)}")
    print(f"  Pubmed 200 PMIDs: {len(pubmed_200_pmids)}")
    print()

    # Analysis 1: Label Changes
    print("Running Analysis 1: Label Change Predictions...")
    analysis1_fold_results = []
    for fold_num in range(5):
        print(f"  Processing fold {fold_num}...")
        result = analyze_label_changes_across_models(fold_num)
        analysis1_fold_results.append(result)
    analysis1_agg = aggregate_label_change_results(analysis1_fold_results)
    print()

    # Analysis 2: pubmed_200 Confusion Matrix
    print("Running Analysis 2: pubmed_200 Confusion Matrix...")
    analysis2_fold_results = []
    for fold_num in range(5):
        print(f"  Processing fold {fold_num}...")
        result = generate_pubmed_200_confusion_matrix(fold_num)
        analysis2_fold_results.append(result)
    analysis2_agg = aggregate_confusion_matrix_results(analysis2_fold_results)
    print()

    # Analysis 3: Prediction Shifts Model 1 → Model 2
    print("Running Analysis 3: Prediction Shifts M1 → M2 (Unchanged Labels)...")
    analysis3_fold_results = []
    for fold_num in range(5):
        print(f"  Processing fold {fold_num}...")
        result = analyze_prediction_shifts_m1_to_m2(fold_num)
        analysis3_fold_results.append(result)
    analysis3_agg = aggregate_prediction_shift_results(analysis3_fold_results)
    print()

    # Analysis 4: Prediction Shifts Model 1 → Model 3
    print("Running Analysis 4: Prediction Shifts M1 → M3 (Unchanged Labels)...")
    analysis4_fold_results = []
    for fold_num in range(5):
        print(f"  Processing fold {fold_num}...")
        result = analyze_prediction_shifts_m1_to_m3(fold_num)
        analysis4_fold_results.append(result)
    analysis4_agg = aggregate_prediction_shift_results(analysis4_fold_results)
    print()

    # Print results
    print_label_change_analysis(analysis1_agg, analysis1_fold_results)
    print_confusion_matrix(analysis2_agg, analysis2_fold_results)
    print_prediction_shifts(analysis3_agg, analysis3_fold_results,
                            3, "PREDICTION SHIFTS MODEL 1 → MODEL 2 (Unchanged Labels)")
    print_prediction_shifts(analysis4_agg, analysis4_fold_results,
                            4, "PREDICTION SHIFTS MODEL 1 → MODEL 3 (Unchanged Labels)")

    # Save results
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / "three_model_comparison_results.json"

    all_results = {
        'analysis1_label_changes': {
            'aggregated': analysis1_agg,
            'fold_results': analysis1_fold_results,
        },
        'analysis2_pubmed_200_confusion_matrix': {
            'aggregated': analysis2_agg,
            'fold_results': analysis2_fold_results,
        },
        'analysis3_prediction_shifts_m1_to_m2': {
            'description': 'Prediction shifts from Model 1 to Model 2 (label updates only)',
            'aggregated': analysis3_agg,
            'fold_results': analysis3_fold_results,
        },
        'analysis4_prediction_shifts_m1_to_m3': {
            'description': 'Prediction shifts from Model 1 to Model 3 (label updates + new data)',
            'aggregated': analysis4_agg,
            'fold_results': analysis4_fold_results,
        },
    }

    save_results_to_json(all_results, output_file)
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
