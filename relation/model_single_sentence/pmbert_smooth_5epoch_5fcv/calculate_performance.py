from pathlib import Path
import re
import json

# Unused legacy functions (kept for reference, commented out to avoid dependencies)
# These functions were used in previous version of the script that worked with CSV predictions

def load_results_from_directory(results_dir):
	"""
	Load eval and predict results from all subdirectories.

	Args:
		results_dir: path to directory containing model result subdirectories

	Returns:
		dict with keys 'pmbert' and 'roberta', each containing list of
		(split_num, eval_results, predict_results) tuples
	"""
	results_dir = Path(results_dir).resolve()
	print(f"Loading results from directory: {results_dir}")

	pmbert_results = []
	roberta_results = []

	# Scan all subdirectories
	for subdir in sorted(results_dir.iterdir()):
		if not subdir.is_dir():
			continue

		dir_name = subdir.name

		# Extract model type and split number from directory name
		if 'pmbert' in dir_name:
			model_type = 'pmbert'
		elif 'roberta' in dir_name:
			model_type = 'roberta'
		else:
			continue

		# Extract split number
		split_match = re.search(r'split_(\d+)', dir_name)
		if not split_match:
			continue
		split_num = int(split_match.group(1))

		# Load JSON files
		eval_file = subdir / 'eval_results.json'
		predict_file = subdir / 'predict_results.json'

		if not eval_file.exists() or not predict_file.exists():
			print(f"Warning: Missing results files in {dir_name}")
			continue

		with open(eval_file, 'r') as f:
			eval_results = json.load(f)
		with open(predict_file, 'r') as f:
			predict_results = json.load(f)

		# Store results
		result_tuple = (split_num, eval_results, predict_results)
		if model_type == 'pmbert':
			pmbert_results.append(result_tuple)
		else:
			roberta_results.append(result_tuple)

	print(f"Loaded {len(pmbert_results)} pmbert splits and {len(roberta_results)} roberta splits")

	return {
		'pmbert': pmbert_results,
		'roberta': roberta_results
	}

def aggregate_metrics(results_list):
	"""
	Aggregate metrics across splits using weighted averaging by n_samples.

	Args:
		results_list: list of (split_num, eval_results, predict_results) tuples

	Returns:
		dict with aggregated eval and predict metrics
	"""
	# Extract data
	eval_metrics = []
	predict_metrics = []
	weights = []

	for split_num, eval_res, predict_res in results_list:
		# Use eval_samples as weight
		n_samples = eval_res.get('eval_samples', 1)
		weights.append(n_samples)

		# Collect eval metrics
		eval_metrics.append({
			'accuracy': eval_res.get('eval_accuracy', 0),
			'f1': eval_res.get('eval_f1', 0),
			'precision': eval_res.get('eval_precision', 0),
			'recall': eval_res.get('eval_recall', 0),
		})

		# Collect predict metrics (no weighting needed, just average)
		predict_metrics.append({
			'accuracy': predict_res.get('predict_accuracy', 0),
			'f1': predict_res.get('predict_f1', 0),
			'precision': predict_res.get('predict_precision', 0),
			'recall': predict_res.get('predict_recall', 0),
		})

	# Calculate weighted average for eval metrics
	total_weight = sum(weights)
	eval_agg = {
		'accuracy': sum(m['accuracy'] * w for m, w in zip(eval_metrics, weights)) / total_weight,
		'f1': sum(m['f1'] * w for m, w in zip(eval_metrics, weights)) / total_weight,
		'precision': sum(m['precision'] * w for m, w in zip(eval_metrics, weights)) / total_weight,
		'recall': sum(m['recall'] * w for m, w in zip(eval_metrics, weights)) / total_weight,
	}

	# Calculate simple average for predict metrics
	n_splits = len(predict_metrics)
	predict_agg = {
		'accuracy': sum(m['accuracy'] for m in predict_metrics) / n_splits,
		'f1': sum(m['f1'] for m in predict_metrics) / n_splits,
		'precision': sum(m['precision'] for m in predict_metrics) / n_splits,
		'recall': sum(m['recall'] for m in predict_metrics) / n_splits,
	}

	return {
		'eval': eval_agg,
		'predict': predict_agg,
		'n_splits': n_splits,
		'total_samples': total_weight
	}

def main():
	results_dir = "."

	# Load all results
	data = load_results_from_directory(results_dir)

	print("\n" + "="*80)
	print("5-FOLD CROSS-VALIDATION RESULTS")
	print("="*80)

	# Aggregate and display results for each model
	for model_name in ['pmbert', 'roberta']:
		results = data[model_name]
		if not results:
			print(f"\nNo results found for {model_name.upper()}")
			continue

		agg = aggregate_metrics(results)

		print(f"\n{model_name.upper()} (averaged across {agg['n_splits']} folds, {agg['total_samples']} total samples)")
		print("-" * 80)

		print("\nEval Results (weighted by n_samples):")
		print(f"  Accuracy:  {agg['eval']['accuracy']:.4f}")
		print(f"  Precision: {agg['eval']['precision']:.4f}")
		print(f"  Recall:    {agg['eval']['recall']:.4f}")
		print(f"  F1:        {agg['eval']['f1']:.4f}")

		print("\nPredict Results (simple average):")
		print(f"  Accuracy:  {agg['predict']['accuracy']:.4f}")
		print(f"  Precision: {agg['predict']['precision']:.4f}")
		print(f"  Recall:    {agg['predict']['recall']:.4f}")
		print(f"  F1:        {agg['predict']['f1']:.4f}")

	print("\n" + "="*80)

if __name__ == "__main__":
	main()
