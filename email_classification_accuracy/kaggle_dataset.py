import kagglehub
import pandas as pd
import os
import logging
import random
import re
import string
import gc
import contextlib
import io
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- SETUP ---
os.environ["WANDB_DISABLED"] = "true"
SEED = 42
KAGGLE_DATASET_ID = "naserabdullahalam/phishing-email-dataset"
SAMPLES_TO_USE = 10000
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 1
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 3e-5
FGM_EPSILON = 0.01
ADV_LOSS_WEIGHT = 0.3
OUTPUT_DIR = "./results"
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

def mask_sensitive_info(text: str) -> str:
    if not text: return ""
    text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[EMAIL]', str(text))
    return text

# --- ROBUSTNESS UTILITIES (Noise Injection) ---
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
        if op == 'deletion':
            text_list.pop(idx)
        elif op == 'substitution':
            text_list[idx] = SUBSTITUTIONS.get(text_list[idx].lower(), random.choice(ALPHANUM))
        elif op == 'insertion':
            text_list.insert(idx + 1, random.choice(ALPHANUM))
    return "".join(text_list)

# --- DATA LOADING ---
def load_and_preprocess_data(tokenizer):
    path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
    csv_path = os.path.join(path, "CEAS_08.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected file CEAS_08.csv not found in {path}")

    df = pd.read_csv(csv_path)
    df = df[['body', 'label']].rename(columns={'body': 'email', 'label': 'labels'}).dropna()
    df['labels'] = df['labels'].astype(int)

    phish_df = df[df['labels'] == 1]
    legit_df = df[df['labels'] == 0]
    min_size = min(len(phish_df), len(legit_df), SAMPLES_TO_USE // 2)

    df_balanced = pd.concat([
        phish_df.sample(min_size, random_state=SEED),
        legit_df.sample(min_size, random_state=SEED)
    ]).sample(frac=1, random_state=SEED)

    df_balanced['email'] = df_balanced['email'].apply(lambda x: mask_sensitive_info(x).lower())
    train_df, temp_df = train_test_split(df_balanced, test_size=0.3, random_state=SEED, stratify=df_balanced['labels'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df['labels'])

    def tokenize_func(examples):
        return tokenizer(examples['email'], truncation=True, padding='max_length', max_length=MAX_LENGTH)

    return (Dataset.from_pandas(train_df).map(tokenize_func, batched=True),
            Dataset.from_pandas(val_df).map(tokenize_func, batched=True),
            Dataset.from_pandas(test_df).map(tokenize_func, batched=True),
            test_df)

# --- FGM & TRAINER ---
class FGM:
    def __init__(self, model):
        self.model, self.backup = model, {}
    def attack(self, epsilon=FGM_EPSILON):
        for name, param in self.model.named_parameters():
            if param.requires_grad and "embeddings" in name:
                self.backup[name] = param.data.clone()
                norm = torch.linalg.norm(param.grad)
                if norm != 0: param.data.add_(epsilon * param.grad / norm)
    def restore(self):
        for name, param in self.backup.items():
            self.model.get_parameter(name).data.copy_(param)
        self.backup = {}

class FGMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        if self.model.training:
            loss.backward(retain_graph=True)
            fgm = FGM(model)
            fgm.attack()
            loss_adv = model(**inputs).loss
            fgm.restore()
            loss = loss + ADV_LOSS_WEIGHT * loss_adv
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# --- EXECUTION ---
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds, val_ds, test_ds, test_df_raw = load_and_preprocess_data(tokenizer)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        weight_decay=0.01,
        report_to="none"
    )

    print("\n--- Training Baseline ---")
    model_b = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    trainer_b = Trainer(model=model_b, args=args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics)
    trainer_b.train()
    res_b = trainer_b.evaluate(test_ds)

    print("\n--- Training Robust (FGM) ---")
    model_r = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    trainer_r = FGMTrainer(model=model_r, args=args, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics)
    trainer_r.train()
    res_r = trainer_r.evaluate(test_ds)

    # Table 1: Standard Evaluation
    print("\n" + "="*85)
    print(f"Standard Results (CEAS_08 Balanced Test Set N={len(test_ds)})")
    print("="*85)
    fmt = "| {:<25} | {:<8} | {:<8} | {:<8} | {:<8} |"
    print(fmt.format("Model Variant","Acc", "Prec", "Rec", "F1"))
    print("|" + "-"*27 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*10 + "|")
    for name, res in [("DistilBERT (Baseline)", res_b), ("DistilBERT + FGM", res_r)]:
        print(fmt.format(name, f"{res.get('eval_accuracy', 0):.3f}", f"{res.get('eval_precision', 0):.3f}",
                         f"{res.get('eval_recall', 0):.3f}", f"{res.get('eval_f1', 0):.3f}"))
    print("="*85)

    # --- Table 2: Robustness Comparison ---
    print("\n--- Robustness Accuracy under Increasing Character-Level Noise ---")
    print(f"| {'Model Variant':<25} | {'0%':<8} | {'5%':<8} | {'10%':<8} | {'20%':<8} |")
    print("|" + "-"*75 + "|")

    eval_args = TrainingArguments(output_dir="./temp", report_to="none", disable_tqdm=True, log_level="error")

    # Correcting the contextlib redirect to avoid 'NoneType' flush error
    with open(os.devnull, 'w') as fnull:
        for name, model in [("Baseline", model_b), ("Robust + FGM", model_r)]:
            row_accs = []
            for noise in NOISE_LEVELS:
                # Create Noisy Test Set
                noisy_emails = [inject_character_noise(t, noise) for t in test_df_raw['email']]
                noisy_df = pd.DataFrame({'email': noisy_emails, 'labels': test_df_raw['labels']})

                with contextlib.redirect_stdout(fnull): # Redirect to devnull instead of None
                    noisy_ds = Dataset.from_pandas(noisy_df).map(
                        lambda e: tokenizer(e['email'], truncation=True, padding='max_length', max_length=MAX_LENGTH),
                        batched=True
                    )

                    eval_trainer = Trainer(model=model, args=eval_args, compute_metrics=compute_metrics)
                    metrics = eval_trainer.evaluate(noisy_ds)
                    row_accs.append(f"{metrics['eval_accuracy']:.4f}")

                del noisy_ds, eval_trainer
                gc.collect()

            print(f"| {name:<25} | {row_accs[0]:<8} | {row_accs[1]:<8} | {row_accs[2]:<8} | {row_accs[3]:<8} |")
    print("="*77 + "\n")

