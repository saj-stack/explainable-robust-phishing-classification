import os
import logging
import random
import string
import gc
import warnings
import contextlib
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
from sklearn.utils.class_weight import compute_class_weight

# --- CONFIGURATION ---
logging.basicConfig(level=logging.ERROR)
os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Dataset & Model
DATASET_NAME = "zefang-liu/phishing-email-dataset"
SAMPLES_TO_USE = 10000
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./results_char_noise"

# Training
MAX_LENGTH = 128
LEARNING_RATE = 3e-5
BATCH_SIZE = 16 # Increase to 32 if memory allows
NUM_TRAIN_EPOCHS = 3 # Increase to 5+ for final runs
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

# Character Noise Parameters
CHAR_NOISE_PROB = 0.3  # 30% of training samples get character noise
CHAR_NOISE_LEVEL = 0.10  # 10% character corruption during training
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20]  # Test noise levels

# --- CHARACTER NOISE INJECTION ---
SUBSTITUTIONS = {'o': '0', 'l': '1', 'e': '3', 'a': '@', 's': '$', 'i': '1', 'z': '2', 'g': '9', 't': '7'}
ALPHANUM = string.ascii_lowercase + string.digits

def inject_character_noise(text: str, noise_level: float) -> str:
    if noise_level == 0.0 or not text:
        return text
    text_list = list(text)
    num_to_corrupt = max(1, int(len(text_list) * noise_level))
    indices = random.sample(range(len(text_list)), min(num_to_corrupt, len(text_list)))
    for idx in sorted(indices, reverse=True):
        op = random.choice(['deletion', 'substitution', 'insertion'])
        if op == 'deletion' and len(text_list) > 1:
            text_list.pop(idx)
        elif op == 'substitution':
            text_list[idx] = SUBSTITUTIONS.get(text_list[idx].lower(), random.choice(ALPHANUM))
        elif op == 'insertion':
            text_list.insert(idx + 1, random.choice(ALPHANUM))
    return "".join(text_list)

# --- WEIGHTED TRAINER (Class Weighting) ---
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        if self.class_weights is not None:
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        else:
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# --- DATA LOADING ---
print("Loading tokenizer and data...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

raw_dataset = load_dataset(DATASET_NAME)
df_raw = raw_dataset['train'].to_pandas()
df_raw.rename(columns={'Email Text': 'email', 'Email Type': 'phishing'}, inplace=True)
df_raw['phishing'] = df_raw['phishing'].map({'Safe Email': 0, 'Phishing Email': 1})

# Sample data
df_sampled, _ = train_test_split(df_raw, train_size=SAMPLES_TO_USE,
                                  stratify=df_raw['phishing'], random_state=SEED)
df_sampled['email'] = df_sampled['email'].apply(lambda x: x.lower() if isinstance(x, str) else '')

# Split
df_train_val, df_test = train_test_split(df_sampled, test_size=TEST_RATIO,
                                          stratify=df_sampled['phishing'], random_state=SEED)
val_rel = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
df_train, df_val = train_test_split(df_train_val, test_size=val_rel,
                                     stratify=df_train_val['phishing'], random_state=SEED)

print("\n--- Class Balance ---")
for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    counts = df['phishing'].value_counts()
    print(f"{name:8} | Legit: {counts.get(0, 0):4} | Phish: {counts.get(1, 0):4}")

# Compute class weights
train_labels = df_train['phishing'].values
classes = np.unique(train_labels)
class_weights_array = compute_class_weight('balanced', classes=classes, y=train_labels)
class_weights = torch.tensor(class_weights_array, dtype=torch.float32)

print(f"\nClass Weights: Legit={class_weights[0]:.4f}, Phish={class_weights[1]:.4f}\n")

def tokenize_fn(examples):
    return tokenizer(examples['email'], truncation=True, padding='max_length', max_length=MAX_LENGTH)

# Apply character noise augmentation to training data
def augment_train(examples):
    augmented = []
    for email in examples['email']:
        if random.random() < CHAR_NOISE_PROB:
            augmented.append(inject_character_noise(email, CHAR_NOISE_LEVEL))
        else:
            augmented.append(email)
    examples['email'] = augmented
    return examples

train_ds = Dataset.from_pandas(df_train[['email', 'phishing']].rename(columns={'phishing': 'labels'}),
                                preserve_index=False)
train_ds = train_ds.map(augment_train, batched=True)
train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=['email'])

val_ds = Dataset.from_pandas(df_val[['email', 'phishing']].rename(columns={'phishing': 'labels'}),
                              preserve_index=False).map(tokenize_fn, batched=True, remove_columns=['email'])

test_ds = Dataset.from_pandas(df_test[['email', 'phishing']].rename(columns={'phishing': 'labels'}),
                               preserve_index=False).map(tokenize_fn, batched=True, remove_columns=['email'])

test_df_raw = df_test[['email', 'phishing']].rename(columns={'phishing': 'labels'})

# --- METRICS ---
def compute_metrics_fn(p):
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    probs = torch.nn.functional.softmax(torch.from_numpy(p.predictions), dim=-1)[:, 1].numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}

# --- TRAINING ---
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
    logging_steps=100
)

print("\nTraining Character Noise...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics_fn,
    class_weights=class_weights
)

trainer.train()
test_metrics = trainer.evaluate(test_ds)

print(f"Character Noise - Test Accuracy: {test_metrics['eval_accuracy']:.4f}")

# --- Performance Table ---
print("\n" + "="*95)
print("Table 1: Performance on Clean Test Data")
print("="*95)
print(f"| {'Model':<30} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'AUC':<6} |")
print("|" + "-"*93 + "|")
print(f"| {'Character Noise':<30} | {test_metrics['eval_accuracy']:<6.3f} | {test_metrics['eval_precision']:<6.3f} | "
      f"{test_metrics['eval_recall']:<6.3f} | {test_metrics['eval_f1']:<6.3f} | {test_metrics['eval_auc']:<6.3f} |")
print("="*95 + "\n")

# --- Robustness Table ---
print("="*50)
print("Table 2: Accuracy Under Character-Level Noise")
print("="*50)
print(f"| {'Model':<20} | {'0%':<6} | {'5%':<6} | {'10%':<6} | {'20%':<6} |")
print("|" + "-"*48 + "|")

datasets.utils.logging.disable_progress_bar()
warnings.filterwarnings("ignore", message=".*pin_memory.*")

eval_args = TrainingArguments(
    output_dir="./temp",
    report_to="none",
    disable_tqdm=True,
    log_level="error"
)

row_accs = []
for noise in NOISE_LEVELS:
    noisy_emails = [inject_character_noise(t, noise) for t in test_df_raw['email']]
    noisy_df = pd.DataFrame({'email': noisy_emails, 'labels': test_df_raw['labels']})
    with contextlib.redirect_stdout(None):
        noisy_ds = Dataset.from_pandas(noisy_df, preserve_index=False).map(
            lambda e: tokenizer(e['email'], truncation=True, padding='max_length', max_length=MAX_LENGTH),
            batched=True, remove_columns=['email']
        )
        eval_trainer = Trainer(model=model, args=eval_args, compute_metrics=compute_metrics_fn)
        metrics = eval_trainer.evaluate(noisy_ds)
        row_accs.append(f"{metrics['eval_accuracy']:.4f}")
    del noisy_ds, eval_trainer
    gc.collect()

print(f"| {'Character Noise':<20} | {row_accs[0]:<6} | {row_accs[1]:<6} | {row_accs[2]:<6} | {row_accs[3]:<6} |")
print("="*50 + "\n")

print("Training complete!")
print(f"Class weights: Legit={class_weights[0]:.4f}, Phish={class_weights[1]:.4f}")
print(f"Character noise: {CHAR_NOISE_PROB*100:.0f}% samples at {CHAR_NOISE_LEVEL*100:.0f}% corruption")

