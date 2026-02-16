import os
import json
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import f1_score, accuracy_score

# ===== Relation definitions =====
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
RELATION2ID = {rel: i for i, rel in enumerate(RELATION_DEFS.keys())}
ID2RELATION = {i: r for r, i in RELATION2ID.items()}

MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
TRAIN_FILE = "./data_tagged/fold_0_train.json"
VAL_FILE = "./data_tagged/fold_0_val.json"
OUTPUT_DIR = "./fine_tuned_models_replicate_original_litcoin_600/fold_0"

# Tokenizer
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Prompt
def construct_input(entry):
    return f"""
        You are a biomedical relation extraction model.
        Determine the relation between the subject and object from these types:
        {', '.join(RELATION_DEFS.keys())}

        Text: {entry['text']}
        Subject: {entry['subject']}
        Object: {entry['object']}
        """.strip()

# Load data
def load_json(path):
    with open(path) as f:
        return json.load(f)

train_data = load_json(TRAIN_FILE)
val_data = load_json(VAL_FILE)

train_dataset = Dataset.from_list([
    {"text": construct_input(e), "label": RELATION2ID[e["relation"]]} for e in train_data
])
val_dataset = Dataset.from_list([
    {"text": construct_input(e), "label": RELATION2ID[e["relation"]]} for e in val_data
])
dataset = DatasetDict({"train": train_dataset, "val": val_dataset})

# Tokenize
def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.remove_columns(["text"])

# Model setup
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=len(RELATION_DEFS),
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    quantization_config=bnb_config
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="SEQ_CLS"
)
model = get_peft_model(model, lora_config)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # Exclude "NOT" for F1
    not_idx = RELATION2ID["NOT"]
    mask = labels != not_idx
    if np.any(mask):
        f1 = f1_score(labels[mask], preds[mask], average="micro")
    else:
        f1 = 0.0
    acc = accuracy_score(labels, preds)
    return {"f1_excl_not": f1, "accuracy": acc}

# Training setup
os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    learning_rate=2e-4,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to="none",
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="f1_excl_not",
    greater_is_better=True
)

trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["val"],
    compute_metrics=compute_metrics,
)

trainer.train()
