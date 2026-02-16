import os
import sys
import json
import csv
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from peft import PeftModel

# === CONFIG ===
if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <test_path>")
    sys.exit(1)

MODEL_DIR = "./fine_tuned_models_replicate_original_litcoin_600/fold_0/checkpoint-24850"
MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
TEST_PATH = sys.argv[1]
OUTPUT_DIR = "./model_predictions"

# Derive output filenames from input (e.g. fold_0_val.json -> eval_fold_0_val.csv)
test_stem = os.path.splitext(os.path.basename(TEST_PATH))[0]
CSV_PATH = os.path.join(OUTPUT_DIR, f"eval_{test_stem}.csv")
RESULTS_PATH = os.path.join(OUTPUT_DIR, f"all_results_{test_stem}.json")
def get_input_device(model):
    """Get the device where inputs should be sent (first layer's device)."""
    current = model
    for attr in ('base_model', 'model'):
        if hasattr(current, 'hf_device_map'):
            return torch.device(next(iter(current.hf_device_map.values())))
        if hasattr(current, attr):
            current = getattr(current, attr)
    if hasattr(current, 'hf_device_map'):
        return torch.device(next(iter(current.hf_device_map.values())))
    return next(model.parameters()).device

# === RELATION TYPES ===
RELATION_DEFS = {
    "Association": "General correlation or connection between entities.",
    "Positive_Correlation": "An increase in one entity is associated with an increase in the other.",
    "Negative_Correlation": "An increase in one entity is associated with a decrease in the other.",
    "Bind": "Physical binding between molecules or proteins.",
    "Cotreatment": "Two entities are administered together or as part of a combined treatment.",
    "Comparison": "A statement that compares two entities directly.",
    "Drug_Interaction": "Interaction between two drugs affecting efficacy or toxicity.",
    "Conversion": "One entity is transformed or converted into another.",
    "NOT": "No meaningful or valid relation exists between the subject and object."
}
ALL_RELATIONS = list(RELATION_DEFS.keys())
EVAL_RELATIONS = [r for r in ALL_RELATIONS if r != "NOT"]

RELATION2ID = {rel: i for i, rel in enumerate(ALL_RELATIONS)}
ID2RELATION = {i: rel for rel, i in RELATION2ID.items()}

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

# Load base model first with correct number of labels
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,  # base model path (NOT the checkpoint)
    num_labels=len(RELATION_DEFS),
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    quantization_config=bnb_config
)

# Load LoRA adapter (this merges the fine-tuned classification head)
model = PeftModel.from_pretrained(model, MODEL_DIR)
model.eval()
input_device = get_input_device(model)

# Load data
with open(TEST_PATH) as f:
    test_data = json.load(f)

preds, trues, rows = [], [], []

# Inference loop
for idx, row in tqdm(enumerate(test_data), total=len(test_data), desc="Evaluating"):
    subject = row["subject"]
    object_ = row["object"]
    text = row["text"]
    gold = row.get("relation", "NOT")

    # Construct input (same as training prompt)
    prompt = f"""
    You are a biomedical relation extraction model.
    Determine the relation between the subject and object from these types:
    {', '.join(ALL_RELATIONS)}

    Text: {text}
    Subject: {subject}
    Object: {object_}
    """.strip()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(input_device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=-1).item()
        pred = ID2RELATION[pred_id]

    # Record predictions
    preds.append(pred)
    trues.append(gold)

    rows.append({
        "index": idx,
        "text": text,
        "subject": subject,
        "object": object_,
        "gold_relation": gold,
        "predicted_relation": pred,
    })

    # Print raw info
    # print(f"\n[{idx}]")
    # print(f"Prompt: {prompt}")
    # print(f"Predicted: {pred} | Gold: {gold}")

    # Running F1 (excluding "NOT")
    mask = [t != "NOT" for t in trues]
    filtered_preds = [p for p, m in zip(preds, mask) if m]
    filtered_trues = [t for t in trues if t != "NOT"]

    if filtered_trues:
        running_f1 = f1_score(filtered_trues, filtered_preds, labels=EVAL_RELATIONS, average="micro")
        # print(f"Running Micro F1 (excl. NOT): {running_f1:.4f}")

# Final metrics (NOT excluded via labels param, not by filtering samples)
accuracy = accuracy_score(trues, preds)
precision, recall, f1, _ = precision_recall_fscore_support(
    trues, preds,
    labels=EVAL_RELATIONS,
    average="micro",
    zero_division=0,
)

all_results = {
    "n_samples": len(test_data),
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
}

print("\n=== FINAL EVALUATION RESULTS ===")
for k, v in all_results.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

# Save predictions and metrics
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(CSV_PATH, "w", newline='') as csvfile:
    fieldnames = ["index", "text", "subject", "object", "gold_relation", "predicted_relation"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

with open(RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=4)

print(f"\nSaved predictions to {CSV_PATH}")
print(f"Saved metrics to {RESULTS_PATH}")
