import os
import json
import warnings
import random
import numpy as np
import pandas as pd
import torch
import transformers
print("transformers:", transformers.__version__)

from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

warnings.filterwarnings("ignore")
set_seed(42)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

LABEL_MAP = {"benign": 0, "malicious": 1}
ID2LABEL = {0: "BENIGN", 1: "MALICIOUS"}
LABEL2ID = {"BENIGN": 0, "MALICIOUS": 1}


class PromptInjectionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def load_jsonl_dataset(file_path):
    print("\n" + "=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)

    required = {"prompt", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["prompt"] = df["prompt"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(LABEL_MAP)].copy()
    df = df[df["prompt"].str.len() > 0].copy()
    df["binary_label"] = df["label"].map(LABEL_MAP)

    print(f"Loaded {len(df)} valid samples")
    print("\nLabel distribution:")
    print(df["label"].value_counts())

    if "attack_type" in df.columns:
        print("\nAttack type distribution:")
        print(df["attack_type"].value_counts(dropna=False))

    return df


def merge_hard_negatives(df, csv_path=None):
    if csv_path and os.path.exists(csv_path):
        print(f"\nMerging hard negatives from: {csv_path}")

        try:
            if csv_path.endswith(".txt"):
                extra = pd.read_csv(csv_path, sep=None, engine="python")
            else:
                extra = pd.read_csv(csv_path)
        except Exception:
            print("Could not parse hard negatives file as CSV/TXT table. Skipping merge.")
            extra = None

        if extra is not None and {"prompt", "label"}.issubset(extra.columns):
            extra["prompt"] = extra["prompt"].astype(str).str.strip()
            extra["label"] = extra["label"].astype(str).str.lower().str.strip()
            extra = extra[extra["label"].isin(LABEL_MAP)].copy()
            extra = extra[extra["prompt"].str.len() > 0].copy()
            extra["binary_label"] = extra["label"].map(LABEL_MAP)

            for col in df.columns:
                if col not in extra.columns:
                    extra[col] = None

            extra = extra[df.columns]
            df = pd.concat([df, extra], ignore_index=True)
        else:
            print("Hard negatives file does not contain required columns {'prompt', 'label'}. Skipping merge.")

    df = df.drop_duplicates(subset=["prompt", "label"]).reset_index(drop=True)
    print(f"Dataset size after merge/dedup: {len(df)}")
    return df


def prepare_data(df):
    print("\n" + "=" * 60)
    print("PREPARING DATA")
    print("=" * 60)

    texts = df["prompt"].astype(str).tolist()
    labels = df["binary_label"].astype(int).tolist()

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=0.1,
        random_state=42,
        stratify=train_labels,
    )

    print(f"Train: {len(train_texts)}")
    print(f"Validation: {len(val_texts)}")
    print(f"Test: {len(test_texts)}")

    return (
        (train_texts, train_labels),
        (val_texts, val_labels),
        (test_texts, test_labels),
    )


def predict_probabilities(model, tokenizer, texts, max_length=512, batch_size=32):
    model.eval()
    device = next(model.parameters()).device
    probs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        encoding = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)
            batch_probs = torch.softmax(outputs.logits, dim=-1)[:, 1].detach().cpu().numpy().tolist()
            probs.extend(batch_probs)

    return probs


def find_best_threshold(model, tokenizer, val_texts, val_labels, min_recall=0.90):
    print("\n" + "=" * 60)
    print("THRESHOLD TUNING")
    print("=" * 60)

    probs = predict_probabilities(model, tokenizer, val_texts)
    best = None

    for t in np.arange(0.30, 0.96, 0.02):
        preds = [1 if p >= t else 0 for p in probs]

        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, preds, average="binary", zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(val_labels, preds, labels=[0, 1]).ravel()

        candidate = {
            "threshold": round(float(t), 2),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "tn": int(tn),
        }

        if recall >= min_recall:
            if best is None or candidate["fp"] < best["fp"] or (
                candidate["fp"] == best["fp"] and candidate["f1"] > best["f1"]
            ):
                best = candidate

    if best is None:
        for t in np.arange(0.30, 0.96, 0.02):
            preds = [1 if p >= t else 0 for p in probs]

            precision, recall, f1, _ = precision_recall_fscore_support(
                val_labels, preds, average="binary", zero_division=0
            )
            tn, fp, fn, tp = confusion_matrix(val_labels, preds, labels=[0, 1]).ravel()

            candidate = {
                "threshold": round(float(t), 2),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "tn": int(tn),
            }

            if best is None or candidate["f1"] > best["f1"]:
                best = candidate

    print("Best threshold found:")
    print(json.dumps(best, indent=2))
    return best


def save_error_reports(texts, labels, probs, threshold, output_dir):
    preds = [1 if p >= threshold else 0 for p in probs]

    rows = []
    for text, true_label, prob, pred in zip(texts, labels, probs, preds):
        rows.append({
            "prompt": text,
            "true_label": ID2LABEL[true_label],
            "predicted_label": ID2LABEL[pred],
            "malicious_probability": round(float(prob), 6),
            "benign_probability": round(float(1 - prob), 6),
            "correct": bool(true_label == pred),
            "error_type": (
                "FALSE_POSITIVE" if true_label == 0 and pred == 1
                else "FALSE_NEGATIVE" if true_label == 1 and pred == 0
                else "CORRECT"
            ),
        })

    full_df = pd.DataFrame(rows)
    full_df.to_csv(os.path.join(output_dir, "test_predictions_detailed.csv"), index=False)
    full_df[full_df["error_type"] == "FALSE_POSITIVE"].to_csv(
        os.path.join(output_dir, "false_positives.csv"), index=False
    )
    full_df[full_df["error_type"] == "FALSE_NEGATIVE"].to_csv(
        os.path.join(output_dir, "false_negatives.csv"), index=False
    )

    return full_df


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        num_labels = logits.shape[-1]

        loss = loss_fct(
            logits.view(-1, num_labels),
            labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss


def train_transformer_model(
    model_name,
    train_data,
    val_data,
    test_data,
    output_dir,
    num_epochs=5,
    batch_size=16,
    max_length=512,
    learning_rate=2e-5,
):
    print("\n" + "=" * 60)
    print(f"TRAINING MODEL: {model_name}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,  # Force FP32 explicitly
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Force all parameters to FP32
    model = model.float()
    
    print(f"Using device: {device}")
    print(f"Model dtype: {model.dtype}")

    train_texts, train_labels = train_data
    val_texts, val_labels = val_data
    test_texts, test_labels = test_data

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=np.array(train_labels),
    )
    class_weights = torch.tensor(weights, dtype=torch.float32)

    print("\nClass Weights:")
    print(class_weights)

    train_dataset = PromptInjectionDataset(
        train_texts, train_labels, tokenizer, max_length=max_length
    )
    val_dataset = PromptInjectionDataset(
        val_texts, val_labels, tokenizer, max_length=max_length
    )

    training_args = TrainingArguments(
    output_dir=os.path.join(output_dir, "results"),
    num_train_epochs=num_epochs,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,  
    per_device_eval_batch_size=batch_size * 2, 
    gradient_accumulation_steps=4,  
    weight_decay=0.01,
    logging_steps=50,  
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
    fp16=False,
    bf16=False,
    fp16_full_eval=False,
    bf16_full_eval=False,
    # Memory optimizations
    gradient_checkpointing=True, 
)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2)
        ],
    )

    trainer.train()

    best_threshold = find_best_threshold(
        trainer.model,
        tokenizer,
        val_texts,
        val_labels,
        min_recall=0.95,
    )
    threshold = best_threshold["threshold"]

    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)

    test_probs = predict_probabilities(
        trainer.model,
        tokenizer,
        test_texts,
        max_length=max_length,
        batch_size=batch_size,
    )
    test_preds = [1 if p >= threshold else 0 for p in test_probs]

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average="binary", zero_division=0
    )
    acc = accuracy_score(test_labels, test_preds)
    cm = confusion_matrix(test_labels, test_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix:")
    print(cm)

    metrics = {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "model_name": model_name,
        "device": str(device),
        "train_size": len(train_texts),
        "val_size": len(val_texts),
        "test_size": len(test_texts),
        "max_length": int(max_length),
        "batch_size": int(batch_size),
        "num_epochs": int(num_epochs),
        "learning_rate": float(learning_rate),
    }

    print("\nFinal Test Metrics:")
    print(json.dumps(metrics, indent=2))

    with open(os.path.join(output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "validation_threshold_search": best_threshold,
                "test_metrics": metrics,
            },
            f,
            indent=2,
        )

    save_error_reports(test_texts, test_labels, test_probs, threshold, output_dir)

    model_dir = os.path.join(output_dir, "prompt_injection_model")
    os.makedirs(model_dir, exist_ok=True)
    trainer.model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    print(f"\nModel saved to: {model_dir}")
    return trainer.model, tokenizer, metrics
 
      
  

def main():
    print("=" * 60)
    print("PROMPT INJECTION DETECTOR TRAINING")
    print("=" * 60)

    dataset_path = os.environ.get(
        "PROMPT_DATASET",
        "/kaggle/input/datasets/raw503/dataset-prompt-injection/Prompt_INJECTION_And_Benign_DATASET.jsonl",
    )
    hard_negatives_csv = os.environ.get(
        "HARD_NEGATIVES_CSV",
        "/kaggle/input/datasets/raw503/dataset-prompt-injection/hard_negative_benign_prompts.txt",
    )
    output_dir = os.environ.get("OUTPUT_DIR", "training_output")
    model_name = os.environ.get("MODEL_NAME", "microsoft/deberta-v3-base")
    num_epochs = int(os.environ.get("NUM_EPOCHS", "5"))
    batch_size = int(os.environ.get("BATCH_SIZE", "8"))
    max_length = int(os.environ.get("MAX_LENGTH", "512"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "2e-5"))

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    df = load_jsonl_dataset(dataset_path)
    df = merge_hard_negatives(df, hard_negatives_csv)

    df.to_csv(os.path.join(output_dir, "merged_training_dataset_preview.csv"), index=False)

    train_data, val_data, test_data = prepare_data(df)

    train_transformer_model(
        model_name=model_name,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        max_length=max_length,
        learning_rate=learning_rate,
    )

    print("\nTraining complete.")
