
"""
DiFrauD Deep Learning detector model by AVISHEK POUDEL

"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# Constants Declaration
DOMAINS = [
    "fake_news",
    "job_scams",
    "phishing",
    "political_statements",
    "product_reviews",
    "sms",
    "twitter_rumours",
]

DOMAIN_STATS = {
    "fake_news":           {"total": 20456, "deceptive": 8832,  "non_deceptive": 11624},
    "job_scams":           {"total": 14295, "deceptive": 599,   "non_deceptive": 13696},
    "phishing":            {"total": 15272, "deceptive": 6074,  "non_deceptive": 9198},
    "political_statements":{"total": 12497, "deceptive": 8042,  "non_deceptive": 4455},
    "product_reviews":     {"total": 20971, "deceptive": 10492, "non_deceptive": 10479},
    "sms":                 {"total": 6574,  "deceptive": 1274,  "non_deceptive": 5300},
    "twitter_rumours":     {"total": 5789,  "deceptive": 1969,  "non_deceptive": 3820},
}

MODEL_NAME  = "distilbert-base-uncased"
MAX_LENGTH  = 256
BATCH_SIZE  = 32
EPOCHS      = 5
LR          = 2e-5
WARMUP_RATIO= 0.1
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HF_BASE = "https://huggingface.co/datasets/difraud/difraud/resolve/main"


# Loading the data

def load_jsonl_from_hf(domain: str, split: str) -> List[Dict]:
    import urllib.request

    cache_dir = os.path.join("difraud_cache", domain)
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, f"{split}.jsonl")

    if not os.path.exists(local_path):
        url = f"{HF_BASE}/{domain}/{split}.jsonl"
        print(f"  Downloading {domain}/{split}.jsonl ...")
        urllib.request.urlretrieve(url, local_path)

    records = []
    with open(local_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_domain(domain: str) -> Tuple[List, List, List]:
    train = load_jsonl_from_hf(domain, "train")
    val   = load_jsonl_from_hf(domain, "validation")
    test  = load_jsonl_from_hf(domain, "test")
    return train, val, test

# Dataset Class

class FraudDataset(Dataset):

    def __init__(self, records: List[Dict], tokenizer, max_length: int = MAX_LENGTH):
        self.texts  = [r["text"]  for r in records]
        self.labels = [r["label"] for r in records]
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }

#Weighted Random Sampler
def make_weighted_sampler(records: List[Dict]) -> WeightedRandomSampler:

    labels  = [r["label"] for r in records]
    counts  = [labels.count(0), labels.count(1)]
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# Loading DistilBERT model with 2 heads

def build_model() -> AutoModelForSequenceClassification:

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    return model.to(DEVICE)


# Final model training

def compute_class_weights(records: List[Dict]) -> torch.Tensor:
    """
    Inverse-frequency class weights for the loss function.
    """
    labels = [r["label"] for r in records]
    n_neg  = labels.count(0)
    n_pos  = labels.count(1)
    total  = n_neg + n_pos
    w0     = total / (2 * n_neg)
    w1     = total / (2 * n_pos)
    return torch.tensor([w0, w1], dtype=torch.float).to(DEVICE)


def train_epoch(model, loader, optimizer, scheduler, loss_fn) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = loss_fn(outputs.logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader) -> Dict:
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
        preds   = outputs.logits.argmax(dim=-1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

    acc    = accuracy_score(all_labels, all_preds)
    f1     = f1_score(all_labels, all_preds, average="macro")
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    return {"accuracy": acc, "macro_f1": f1, "roc_auc": auc,
            "preds": all_preds, "labels": all_labels}


def train_on_domain(
    domain: str,
    train_records: List[Dict],
    val_records:   List[Dict],
    tokenizer,
) -> AutoModelForSequenceClassification:
    """
    Fine-tuning the DistilBERT on a single domain's training split.
    """

    print(f"  Training on: {domain}")
    print(f"  Train: {len(train_records)} | Val: {len(val_records)}")

    train_dataset = FraudDataset(train_records, tokenizer)
    val_dataset   = FraudDataset(val_records,   tokenizer)

    sampler    = make_weighted_sampler(train_records)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    model      = build_model()
    class_w    = compute_class_weights(train_records)
    loss_fn    = nn.CrossEntropyLoss(weight=class_w)

    optimizer  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps= len(train_loader) * EPOCHS
    warmup     = int(total_steps * WARMUP_RATIO)
    scheduler  = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    best_f1    = 0.0
    best_state = None
    patience   = 2
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn)
        metrics = evaluate(model, val_loader)
        print(
            f"  Epoch {epoch}/{EPOCHS} | loss={loss:.4f} | "
            f"val_macro_f1={metrics['macro_f1']:.4f} | "
            f"val_acc={metrics['accuracy']:.4f}"
        )
        if metrics["macro_f1"] > best_f1:
            best_f1    = metrics["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

    if best_state:
        model.load_state_dict(best_state)

    return model


# Evaluation metrics

def print_domain_results(domain: str, metrics: Dict):
    print(f"\n{domain}")
    print(f"     Accuracy   : {metrics['accuracy']:.3f}")
    print(f"     Macro F1   : {metrics['macro_f1']:.3f}")
    print(f"     ROC-AUC    : {metrics['roc_auc']:.3f}")
    print(classification_report(metrics["labels"], metrics["preds"],
                                target_names=["non-deceptive", "deceptive"]))


# Experiment

def mode_single(domain: str):
    """Training and evaluating on one domain."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train, val, test = load_domain(domain)
    model = train_on_domain(domain, train, val, tokenizer)

    test_dataset = FraudDataset(test, tokenizer)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    metrics = evaluate(model, test_loader)
    print_domain_results(domain, metrics)
    return metrics


def mode_all():
    
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)
    results    = {}

    for domain in DOMAINS:
        train, val, test = load_domain(domain)
        model = train_on_domain(domain, train, val, tokenizer)

        test_dataset = FraudDataset(test, tokenizer)
        test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        metrics      = evaluate(model, test_loader)
        results[domain] = metrics
        print_domain_results(domain, metrics)

    # Aggregate / mean
    avg_acc = np.mean([r["accuracy"] for r in results.values()])
    avg_f1  = np.mean([r["macro_f1"] for r in results.values()])
    print(f"\n{'='*60}")
    print(f"  AGGREGATE (mean across 7 domains)")
    print(f"  Accuracy : {avg_acc:.3f}")
    print(f"  Macro F1 : {avg_f1:.3f}")
    print(f"{'='*60}")
    return results


def mode_cross_domain(held_out: str):
    """
    Train on 6 domains, test on 7th domain.
    """
    tokenizer   = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_domains = [d for d in DOMAINS if d != held_out]

    # Combine all training splits
    all_train, all_val = [], []
    for domain in train_domains:
        tr, va, _ = load_domain(domain)
        all_train.extend(tr)
        all_val.extend(va)

    # Shuffle
    random.shuffle(all_train)
    random.shuffle(all_val)

    model = train_on_domain(f"all_except_{held_out}", all_train, all_val, tokenizer)

    # Evaluation on test split
    _, _, test = load_domain(held_out)
    test_dataset = FraudDataset(test, tokenizer)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    metrics      = evaluate(model, test_loader)
    print(f"\n  Cross-domain — held out: {held_out}")
    print_domain_results(held_out, metrics)
    return metrics


# Exploratory Data Analysis

def run_eda():
    
    print("  DiFrauD DATASET — Exploratory Data Analysis")
    total_total = sum(v["total"] for v in DOMAIN_STATS.values())
    total_dec   = sum(v["deceptive"] for v in DOMAIN_STATS.values())
    total_nond  = sum(v["non_deceptive"] for v in DOMAIN_STATS.values())

    print(f"  Total samples : {total_total:,}")
    print(f"  Deceptive     : {total_dec:,}  ({100*total_dec/total_total:.1f}%)")
    print(f"  Non-deceptive : {total_nond:,}  ({100*total_nond/total_total:.1f}%)\n")

    header = f"  {'Domain':<25} {'Total':>7} {'Decep':>7} {'Non-D':>7} {'Imbal':>7}"
    print(header)

    for domain, stats in DOMAIN_STATS.items():
        imbal = stats["deceptive"] / stats["non_deceptive"]
        print(
            f"  {domain:<25} {stats['total']:>7,} "
            f"{stats['deceptive']:>7,} {stats['non_deceptive']:>7,} "
            f"{imbal:>6.2f}x"
        )
    print()

    # Highlight extreme cases
    most_imbal = min(DOMAIN_STATS, key=lambda d: DOMAIN_STATS[d]["deceptive"] /
                                                  DOMAIN_STATS[d]["non_deceptive"])
    print(f"  Most imbalanced  : {most_imbal} "
          f"(ratio={DOMAIN_STATS[most_imbal]['deceptive']/DOMAIN_STATS[most_imbal]['non_deceptive']:.2f}x)")
    largest = max(DOMAIN_STATS, key=lambda d: DOMAIN_STATS[d]["total"])
    print(f"  Largest domain   : {largest} ({DOMAIN_STATS[largest]['total']:,} samples)")
    smallest = min(DOMAIN_STATS, key=lambda d: DOMAIN_STATS[d]["total"])
    print(f"  Smallest domain  : {smallest} ({DOMAIN_STATS[smallest]['total']:,} samples)")


# Command Line Interface (CLI)

def parse_args():
    p = argparse.ArgumentParser(description="DiFrauD Deep Learning Detector")
    p.add_argument(
        "--mode", choices=["eda", "single", "all", "cross"],
        default="eda",
        help="eda: print stats only | single: one domain | all: all domains | cross: cross-domain",
    )
    p.add_argument(
        "--domain",
        choices=DOMAINS,
        default="phishing",
        help="Domain for --mode single",
    )
    p.add_argument(
        "--held_out",
        choices=DOMAINS,
        default="twitter_rumours",
        help="Domain to hold out for --mode cross",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eda()

    if args.mode == "eda":
        print("EDA complete. Run with --mode single/all/cross to train a model.")
    elif args.mode == "single":
        mode_single(args.domain)
    elif args.mode == "all":
        mode_all()
    elif args.mode == "cross":
        mode_cross_domain(args.held_out)
