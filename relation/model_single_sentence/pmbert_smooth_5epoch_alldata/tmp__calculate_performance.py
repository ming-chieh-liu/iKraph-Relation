import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
import re
import json

# Label definitions
LABEL_LIST = ["NOT", "Association", "Positive_Correlation", "Negative_Correlation", "Bind", "Cotreatment", "Comparison", "Drug_Interaction", "Conversion"]
LABEL_DICT = {label: i for i, label in enumerate(LABEL_LIST)}

# F1 calculation uses labels 1-8 (excluding NOT=0)
F1_LABELS = list(range(len(LABEL_LIST)))
F1_TRUE_LABELS = list(range(1, len(LABEL_LIST)))

def compute_metrics(label_ids, predictions):
	"""
	Calculate precision, recall, f1 using micro averaging on labels 1-8 (excluding NOT).

	Args:
		label_ids: array of true labels
		predictions: array of predicted labels

	Returns:
		dict with precision, recall, f1 scores
	"""
	precision, recall, f1, _ = precision_recall_fscore_support(
		label_ids,
		predictions,
		labels=F1_TRUE_LABELS,
		average='micro',
		zero_division=0
	)
	return {"precision": precision, "recall": recall, "f1": f1}

def extract_batch_number(relation_id):
	"""
	Extract batch number from relation_id.
	Format: True.11683992.6401.101404.2.1.combination0
	Batch number is the 3rd from last element (before the last two fields).

	Args:
		relation_id: string in format with dot-separated fields

	Returns:
		batch number as integer
	"""
	parts = relation_id.split('.')
	# The format is: label.abstract_id.entity_a.entity_b.X.batch.Y.combinationN
	# So batch number is at index -3
	batch_num = int(parts[-3])
	return batch_num

def load_predictions(predictions_dir):
	"""
	Load all prediction CSV files and categorize by model type.

	Args:
		predictions_dir: path to directory containing prediction files

	Returns:
		dict with keys 'pmbert' and 'roberta', each containing list of dataframes
	"""
	predictions_dir = Path(predictions_dir)

	pmbert_files = []
	roberta_files = []

	# Find all prediction files (not eval files)
	for file in sorted(predictions_dir.glob("*_predictions.csv")):
		if "eval" not in file.name:
			if "pmbert" in file.name:
				pmbert_files.append(file)
			elif "roberta" in file.name:
				roberta_files.append(file)

	# Load dataframes
	pmbert_dfs = [pd.read_csv(f) for f in pmbert_files]
	roberta_dfs = [pd.read_csv(f) for f in roberta_files]

	print(f"Loaded {len(pmbert_dfs)} pmbert files and {len(roberta_dfs)} roberta files")

	return {
		'pmbert': pmbert_dfs,
		'roberta': roberta_dfs
	}

def calculate_overall_metrics(dfs):
	"""
	Calculate metrics by averaging across all checkpoints.

	Args:
		dfs: list of dataframes from different checkpoints

	Returns:
		dict with averaged precision, recall, f1
	"""
	metrics_list = []

	for df in dfs:
		label_ids = df['label'].values
		predictions = df['predicted_class'].values
		metrics = compute_metrics(label_ids, predictions)
		metrics_list.append(metrics)

	# Average across checkpoints
	avg_metrics = { 
		'precision': np.mean([m['precision'] for m in metrics_list]),
		'recall': np.mean([m['recall'] for m in metrics_list]),
		'f1': np.mean([m['f1'] for m in metrics_list])
	}

	return avg_metrics

def calculate_metrics_by_batch(dfs):
	"""
	Calculate metrics by model and batch number.

	Args:
		dfs: list of dataframes from different checkpoints

	Returns:
		dict with batch numbers as keys and averaged metrics as values
	"""
	# First, combine all checkpoints and group by batch
	all_data = []

	for df in dfs:
		df_copy = df.copy()
		df_copy['batch'] = df_copy['relation_id'].apply(extract_batch_number)
		all_data.append(df_copy)

	combined_df = pd.concat(all_data, ignore_index=True)

	# Group by batch and calculate metrics
	batches = sorted(combined_df['batch'].unique())
	batch_metrics = {}

	for batch in batches:
		batch_df = combined_df[combined_df['batch'] == batch]
		label_ids = batch_df['label'].values
		predictions = batch_df['predicted_class'].values
		metrics = compute_metrics(label_ids, predictions)
		batch_metrics[batch] = metrics

	return batch_metrics

def analyze_predictions_by_class(dfs):
	"""
	Analyze prediction distribution by class across all checkpoints.

	Args:
		dfs: list of dataframes from different checkpoints

	Returns:
		DataFrame with class counts and percentages
	"""
	# Combine all checkpoints
	all_data = pd.concat(dfs, ignore_index=True)

	# Count predictions by class
	pred_counts = all_data['predicted_class'].value_counts().sort_index()
	true_counts = all_data['label'].value_counts().sort_index()

	# Create summary dataframe
	class_summary = pd.DataFrame({
		'Class_ID': range(len(LABEL_LIST)),
		'Class_Name': LABEL_LIST,
	})

	class_summary['True_Count'] = class_summary['Class_ID'].map(true_counts).fillna(0).astype(int)
	class_summary['Predicted_Count'] = class_summary['Class_ID'].map(pred_counts).fillna(0).astype(int)
	class_summary['True_Percentage'] = (class_summary['True_Count'] / len(all_data) * 100)
	class_summary['Predicted_Percentage'] = (class_summary['Predicted_Count'] / len(all_data) * 100)

	return class_summary

def load_mapping_results(mapping_file):
	"""
	Load mapping results and extract set of new_relation_ids.

	Args:
		mapping_file: path to mapping_results.json

	Returns:
		tuple of (all_mapped_ids, unchanged_mapped_ids)
	"""
	with open(mapping_file, 'r') as f:
		data = json.load(f)

	mapped_ids = {entry['new_relation_id'] for entry in data['matched_mappings']}
	unchanged_ids = {entry['new_relation_id'] for entry in data['matched_mappings'] if not entry['changed']}
	changed_ids = {entry['new_relation_id'] for entry in data['matched_mappings'] if entry['changed']}

	print(f"Loaded {len(mapped_ids)} mapped relation IDs")
	print(f"  - {len(unchanged_ids)} unchanged annotations")
	print(f"  - {len(changed_ids)} changed annotations")

	return mapped_ids, unchanged_ids, changed_ids

def get_unique_batches(dfs):
	"""
	Extract unique batch numbers from dataframes.

	Args:
		dfs: list of dataframes from different checkpoints

	Returns:
		sorted list of unique batch numbers
	"""
	all_batches = set()
	for df in dfs:
		df_copy = df.copy()
		df_copy['batch'] = df_copy['relation_id'].apply(extract_batch_number)
		all_batches.update(df_copy['batch'].unique())
	return sorted(all_batches)

def calculate_metrics_for_mapped_batch(dfs, mapped_ids, batch_number=None):
	"""
	Calculate metrics for relations that exist in mapping results, optionally filtered by batch.

	Args:
		dfs: list of dataframes from different checkpoints
		mapped_ids: set of relation_ids from mapping results
		batch_number: specific batch to filter (None for all batches)

	Returns:
		dict with averaged precision, recall, f1
	"""
	# Filter to mapped IDs and optionally batch number for each checkpoint
	filtered_dfs = []

	for df in dfs:
		df_copy = df.copy()
		df_copy['batch'] = df_copy['relation_id'].apply(extract_batch_number)
		# Filter to mapped IDs and optionally specific batch
		if batch_number is not None:
			filtered_df = df_copy[(df_copy['batch'] == batch_number) & (df_copy['relation_id'].isin(mapped_ids))]
		else:
			filtered_df = df_copy[df_copy['relation_id'].isin(mapped_ids)]
		filtered_dfs.append(filtered_df)

	# Calculate metrics for each checkpoint
	metrics_list = []
	for df in filtered_dfs:
		if len(df) > 0:
			label_ids = df['label'].values
			predictions = df['predicted_class'].values
			metrics = compute_metrics(label_ids, predictions)
			metrics_list.append(metrics)

	# Average across checkpoints
	if metrics_list:
		avg_metrics = {
			'precision': np.mean([m['precision'] for m in metrics_list]),
			'recall': np.mean([m['recall'] for m in metrics_list]),
			'f1': np.mean([m['f1'] for m in metrics_list])
		}
	else:
		avg_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

	return avg_metrics

def main():
	predictions_dir = "/data/mliu/iKraph/relation/model_single_sentence/pmbert_smooth_5epoch_alldata/predictions_with_extra_on_pubmed"
	mapping_file = "/data/mliu/iKraph/relation/data/litcoin_400_164_80updated/mapping_results.json"

	# Load predictions
	data = load_predictions(predictions_dir)

	# Load mapping results
	mapped_ids, unchanged_ids, changed_ids = load_mapping_results(mapping_file)

	print("\n" + "="*80)
	print("OVERALL METRICS (averaged across 3 checkpoints)")
	print("="*80)

	# Calculate overall metrics for each model
	for model_name in ['pmbert', 'roberta']:
		metrics = calculate_overall_metrics(data[model_name])
		print(f"\n{model_name.upper()}:")
		print(f"  Precision: {metrics['precision']:.4f}")
		print(f"  Recall:    {metrics['recall']:.4f}")
		print(f"  F1:        {metrics['f1']:.4f}")

	print("\n" + "="*80)
	print("PREDICTIONS BY CLASS COUNT (RoBERTa - Batch 1 Only)")
	print("="*80)

	# Filter for batch 1 only
	roberta_batch1_dfs = []
	for df in data['roberta']:
		df_copy = df.copy()
		df_copy['batch'] = df_copy['relation_id'].apply(extract_batch_number)
		batch1_df = df_copy[df_copy['batch'] == 1]
		roberta_batch1_dfs.append(batch1_df)

	roberta_class_summary = analyze_predictions_by_class(roberta_batch1_dfs)
	print("\nRoBERTa - Prediction Distribution (Batch 1):")
	print(f"{'ID':<5} {'Class':<25} {'True Count':<12} {'True %':<10} {'Pred Count':<12} {'Pred %':<10}")
	print("-" * 84)
	for _, row in roberta_class_summary.iterrows():
		print(f"{row['Class_ID']:<5} {row['Class_Name']:<25} {row['True_Count']:<12} {row['True_Percentage']:<10.2f} {row['Predicted_Count']:<12} {row['Predicted_Percentage']:<10.2f}")

	print("\n" + "="*80)
	print("METRICS BY BATCH NUMBER")
	print("="*80)

	# Calculate metrics by batch for each model
	for model_name in ['pmbert', 'roberta']:
		print(f"\n{model_name.upper()}:")
		batch_metrics = calculate_metrics_by_batch(data[model_name])

		print(f"{'Batch':<10} {'Precision':<12} {'Recall':<12} {'F1':<12}")
		print("-" * 46)
		for batch in sorted(batch_metrics.keys()):
			metrics = batch_metrics[batch]
			print(f"{batch:<10} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")

	print("\n" + "="*80)
	print("METRICS FOR MAPPED RELATIONS BY BATCH")
	print("="*80)

	# Get all unique batches
	all_batches = get_unique_batches(data['pmbert'])

	# Calculate metrics for mapped relations by batch for each model
	for model_name in ['pmbert', 'roberta']:
		print(f"\n{model_name.upper()}:")
		print(f"{'Batch':<10} {'Precision':<12} {'Recall':<12} {'F1':<12}")
		print("-" * 46)
		for batch in all_batches:
			metrics = calculate_metrics_for_mapped_batch(data[model_name], mapped_ids, batch_number=batch)
			print(f"{batch:<10} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")

	print("\n" + "="*80)
	print("METRICS FOR UNCHANGED MAPPED RELATIONS BY BATCH")
	print("="*80)

	# Calculate metrics for unchanged mapped relations by batch for each model
	for model_name in ['pmbert', 'roberta']:
		print(f"\n{model_name.upper()}:")
		print(f"{'Batch':<10} {'Precision':<12} {'Recall':<12} {'F1':<12}")
		print("-" * 46)
		for batch in all_batches:
			metrics = calculate_metrics_for_mapped_batch(data[model_name], unchanged_ids, batch_number=batch)
			print(f"{batch:<10} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")

	print("\n" + "="*80)
	print("METRICS FOR CHANGED MAPPED RELATIONS BY BATCH")
	print("="*80)

	# Calculate metrics for changed mapped relations by batch for each model
	for model_name in ['pmbert', 'roberta']:
		print(f"\n{model_name.upper()}:")
		print(f"{'Batch':<10} {'Precision':<12} {'Recall':<12} {'F1':<12}")
		print("-" * 46)
		for batch in all_batches:
			metrics = calculate_metrics_for_mapped_batch(data[model_name], changed_ids, batch_number=batch)
			print(f"{batch:<10} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")

if __name__ == "__main__":
	main()
