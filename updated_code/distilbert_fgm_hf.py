import os
import sys
import logging
import random
import re
import string
import gc
import warnings
import contextlib
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
import datasets
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# --- 1. CONFIGURATION & SETUP ---
logging.basicConfig(level=logging.ERROR)
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DATASET_NAME = "zefang-liu/phishing-email-dataset"
SAMPLES_TO_USE = 10000
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./results_phishing_detector_enhanced"
MAX_LENGTH = 128
LEARNING_RATE = 3e-5
BATCH_SIZE = 16 # Increase to 32 if memory allows
GRADIENT_ACCUMULATION_STEPS = 1
NUM_TRAIN_EPOCHS = 3 # Increase to 5+ for final runs
FGM_EPSILON = 0.01 # Adversarial perturbation strength (try 0.01 or 0.1)
ADV_LOSS_WEIGHT = 0.5
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20]

# --- 2. PII MASKING & DATA UTILITIES ---
# try:
#     import spacy
#     NER_NLP = spacy.load("en_core_web_sm")
# except Exception:
#     NER_NLP = None
#
# _EMAIL_REGEX = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
# _PHONE_REGEX = re.compile(r'\b(?:\+?\d[\d\-\s\(\)]{6,}\d)\b')
#
# def mask_sensitive_info(text: str) -> str:
#     if not text or NER_NLP is None:
#         text = _EMAIL_REGEX.sub('[EMAIL]', text)
#         text = _PHONE_REGEX.sub('[PHONE]', text)
#         text = re.sub(r'\b\d{7,}\b', '[ACCOUNT]', text)
#         return text
#     doc = NER_NLP(text)
#     masked_text = text
#     for ent in reversed(doc.ents):
#         replacement = None
#         if ent.label_ == "PERSON": replacement = "[NAME]"
#         elif ent.label_ in ["CARDINAL", "MONEY", "QUANTITY"]:
#             digits = re.sub(r'\D', '', ent.text)
#             if len(digits) >= 6: replacement = "[ACCOUNT]"
#         if replacement:
#             masked_text = masked_text[:ent.start_char] + replacement + masked_text[ent.end_char:]
#     masked_text = _PHONE_REGEX.sub('[PHONE]', masked_text)
#     masked_text = _EMAIL_REGEX.sub('[EMAIL]', masked_text)
#     return masked_text

def get_class_balance(df, split_name):
    counts = df['labels'].value_counts().to_dict()
    legit = counts.get(0, 0)
    phish = counts.get(1, 0)
    print(f"| {split_name:<12} | {legit:<10} | {phish:<10} | {legit+phish:<10} |")

# --- 3. DATA LOADING & SPLITTING ---
def load_and_preprocess_data(tokenizer):
    raw_dataset = load_dataset(DATASET_NAME)
    df_raw = raw_dataset['train'].to_pandas()
    df_raw.rename(columns={'Email Text': 'email', 'Email Type': 'phishing'}, inplace=True)
    df_raw['phishing'] = df_raw['phishing'].map({'Safe Email': 0, 'Phishing Email': 1})

    df_sampled, _ = train_test_split(df_raw, train_size=SAMPLES_TO_USE, stratify=df_raw['phishing'], random_state=SEED)

    # Masking logic commented out; now just lowercases the text
    # df_sampled['email'] = df_sampled['email'].apply(lambda x: mask_sensitive_info(x).lower() if isinstance(x, str) else '')
    df_sampled['email'] = df_sampled['email'].apply(lambda x: x.lower() if isinstance(x, str) else '')

    df_train_val, df_test = train_test_split(df_sampled, test_size=TEST_RATIO, stratify=df_sampled['phishing'], random_state=SEED)
    val_rel = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    df_train, df_val = train_test_split(df_train_val, test_size=val_rel, stratify=df_train_val['phishing'], random_state=SEED)

    print("\n--- Class Balance Report ---")
    print(f"| {'Split':<12} | {'Legitimate':<10} | {'Phishing':<10} | {'Total':<10} |")
    print("|" + "-"*50 + "|")
    get_class_balance(df_train.rename(columns={'phishing': 'labels'}), "Training")
    get_class_balance(df_val.rename(columns={'phishing': 'labels'}), "Validation")
    get_class_balance(df_test.rename(columns={'phishing': 'labels'}), "Testing")
    print("-" * 52)

    def tokenize_fn(examples):
        return tokenizer(examples['email'], truncation=True, padding='max_length', max_length=MAX_LENGTH)

    train_ds = Dataset.from_pandas(df_train[['email', 'phishing']].rename(columns={'phishing': 'labels'}), preserve_index=False).map(tokenize_fn, batched=True, remove_columns=['email'])
    val_ds = Dataset.from_pandas(df_val[['email', 'phishing']].rename(columns={'phishing': 'labels'}), preserve_index=False).map(tokenize_fn, batched=True, remove_columns=['email'])
    test_ds = Dataset.from_pandas(df_test[['email', 'phishing']].rename(columns={'phishing': 'labels'}), preserve_index=False).map(tokenize_fn, batched=True, remove_columns=['email'])

    return train_ds, val_ds, test_ds, df_test[['email', 'phishing']].rename(columns={'phishing': 'labels'})

# --- 4. FGM & NOISE INJECTION ---
class FGM:
    def __init__(self, model, epsilon=1.0):
        self.model, self.epsilon, self.backup = model, epsilon, {}
    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and "embeddings" in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.linalg.norm(param.grad)
                if norm != 0: param.data.add_(self.epsilon * param.grad / norm)
    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup: param.data.copy_(self.backup[name])
        self.backup = {}

class FGMTrainer(Trainer):
    def __init__(self, fgm_epsilon=FGM_EPSILON, adv_loss_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.fgm = FGM(self.model, epsilon=fgm_epsilon)
        self.adv_loss_weight = adv_loss_weight
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss_clean = outputs.loss
        if loss_clean.requires_grad:
            loss_clean.backward(retain_graph=True)
            self.fgm.attack()
            loss_adv = model(**inputs).loss
            self.fgm.restore()
            total_loss = loss_clean + self.adv_loss_weight * loss_adv
        else: total_loss = loss_clean
        return (total_loss, outputs) if return_outputs else total_loss

SUBSTITUTIONS = {'o': '0', 'l': '1', 'e': '3', 'a': '@', 's': '$', 'i': '1', 'z': '2', 'g': '9', 't': '7'}
ALPHANUM = string.ascii_lowercase + string.digits

def inject_character_noise(text: str, noise_level: float) -> str:
    if noise_level == 0.0 or not text: return text
    text_list = list(text)
    num_to_corrupt = max(1, int(len(text_list) * noise_level))
    indices = random.sample(range(len(text_list)), num_to_corrupt)
    indices.sort(reverse=True)
    for idx in indices:
        op = random.choice(['deletion', 'substitution', 'insertion'])
        if op == 'deletion': text_list.pop(idx)
        elif op == 'substitution': text_list[idx] = SUBSTITUTIONS.get(text_list[idx].lower(), random.choice(ALPHANUM))
        elif op == 'insertion': text_list.insert(idx + 1, random.choice(ALPHANUM))
    return "".join(text_list)

# --- 5. EVALUATION METRICS ---
def compute_metrics_fn(p):
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    probs = torch.nn.functional.softmax(torch.from_numpy(p.predictions), dim=-1)[:, 1].numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}

# --- 6. MAIN EXECUTION ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_ds, val_ds, test_ds, test_df_raw = load_and_preprocess_data(tokenizer)

args = TrainingArguments(
    output_dir=OUTPUT_DIR, per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_TRAIN_EPOCHS, learning_rate=LEARNING_RATE, eval_strategy="epoch",
    save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model="accuracy", report_to="none"
)

# 1. Train Baseline
model_b = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
trainer_b = Trainer(model=model_b, args=args, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer, compute_metrics=compute_metrics_fn)
trainer_b.train()
m_b = trainer_b.evaluate(test_ds)

# 2. Train Robust
model_r = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
trainer_r = FGMTrainer(model=model_r, args=args, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer, compute_metrics=compute_metrics_fn)
trainer_r.train()
m_r = trainer_r.evaluate(test_ds)

# Table 1
print("\n" + "="*95)
print("Table 1: Calculated Classification Performance")
print("="*95)
print(f"| {'Model Variant':<25} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'AUC':<6} |")
print("|" + "-"*93 + "|")
for name, m in [("Baseline DistilBERT", m_b), ("DistilBERT + FGM (Ours)", m_r)]:
    print(f"| {name:<25} | {m['eval_accuracy']:<6.3f} | {m['eval_precision']:<6.3f} | {m['eval_recall']:<6.3f} | {m['eval_f1']:<6.3f} | {m['eval_auc']:<6.3f} |")
print("="*95 + "\n")

# --- Table 2 Fix ---
print("--- Table 2: Model Accuracy under Increasing Character-Level Noise ---")
print(f"| {'Model Variant':<25} | {'0%':<8} | {'5%':<8} | {'10%':<8} | {'20%':<8} |")
print("|" + "-"*75 + "|")

# Global Settings to force silence
datasets.utils.logging.disable_progress_bar()
warnings.filterwarnings("ignore", message=".*pin_memory.*")

eval_args = TrainingArguments(
    output_dir="./temp",
    report_to="none",
    disable_tqdm=True,
    log_level="error"
)

for name, model in [("Baseline", model_b), ("Robust + FGM", model_r)]:
    row_accs = []
    for noise in NOISE_LEVELS:
        # Create Noisy Dataset
        noisy_emails = [inject_character_noise(t, noise) for t in test_df_raw['email']]
        noisy_df = pd.DataFrame({'email': noisy_emails, 'labels': test_df_raw['labels']})

        # Tokenize (using contextlib to ensure silence)
        with contextlib.redirect_stdout(None):
            noisy_ds = Dataset.from_pandas(noisy_df, preserve_index=False).map(
                lambda e: tokenizer(e['email'], truncation=True, padding='max_length', max_length=MAX_LENGTH),
                batched=True, remove_columns=['email']
            )

            # Evaluate (using contextlib to hide the results dictionary)
            eval_trainer = Trainer(model=model, args=eval_args, compute_metrics=compute_metrics_fn)
            metrics = eval_trainer.evaluate(noisy_ds)
            row_accs.append(f"{metrics['eval_accuracy']:.4f}")

        del noisy_ds, eval_trainer
        gc.collect()

    # Print the row ONLY after evaluation for that model is completely finished
    print(f"| {name:<25} | {row_accs[0]:<8} | {row_accs[1]:<8} | {row_accs[2]:<8} | {row_accs[3]:<8} |")

print("="*77 + "\n")

