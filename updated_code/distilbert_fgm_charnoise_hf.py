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
OUTPUT_DIR = "./results_hybrid"

# Training
MAX_LENGTH = 128
LEARNING_RATE = 3e-5
BATCH_SIZE = 16 # Increase to 32 if memory allows
NUM_TRAIN_EPOCHS = 3 # Increase to 5+ for final runs
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

# FGM & Character Noise
FGM_EPSILON = 0.1  # Adversarial perturbation strength (try 0.01 or 0.1)
ADV_LOSS_WEIGHT = 0.5
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

# --- FGM ADVERSARIAL TRAINING ---
class FGM:
    def __init__(self, model, epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and "embeddings" in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.linalg.norm(param.grad)
                if norm != 0:
                    param.data.add_(self.epsilon * param.grad / norm)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

# --- WEIGHTED TRAINER (Class Weighting Only) ---
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        # Compute weighted loss if class weights provided
        if self.class_weights is not None:
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        else:
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# --- HYBRID TRAINER (FGM + Character Noise + Class Weighting) ---
class HybridTrainer(Trainer):
    def __init__(self, fgm_epsilon=FGM_EPSILON, adv_loss_weight=ADV_LOSS_WEIGHT,
                 class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.fgm = FGM(self.model, epsilon=fgm_epsilon)
        self.adv_loss_weight = adv_loss_weight
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        # Compute weighted loss if class weights provided
        if self.class_weights is not None:
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
            loss_clean = loss_fct(logits.view(-1, 2), labels.view(-1))
        else:
            loss_clean = outputs.loss

        # FGM adversarial training
        if loss_clean.requires_grad:
            loss_clean.backward(retain_graph=True)
            self.fgm.attack()
            # Adversarial loss
            adv_outputs = model(**inputs)
            if self.class_weights is not None:
                adv_logits = adv_outputs.get("logits")
                loss_adv = loss_fct(adv_logits.view(-1, 2), labels.view(-1))
            else:
                loss_adv = adv_outputs.loss
            self.fgm.restore()
            total_loss = loss_clean + self.adv_loss_weight * loss_adv
        else:
            total_loss = loss_clean

        return (total_loss, outputs) if return_outputs else total_loss

# --- DATA LOADING ---
def load_and_preprocess_data(tokenizer):
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

    # Compute class weights for handling imbalance
    train_labels = df_train['phishing'].values
    classes = np.unique(train_labels)
    class_weights_array = compute_class_weight('balanced', classes=classes, y=train_labels)
    class_weights = torch.tensor(class_weights_array, dtype=torch.float32)
    print(f"\nClass Weights: Legit={class_weights[0]:.4f}, Phish={class_weights[1]:.4f}")
    print("(Weights > 1.0 indicate minority class, < 1.0 indicate majority class)\n")

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
    train_ds = train_ds.map(augment_train, batched=True)  # Add character noise
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=['email'])
    val_ds = Dataset.from_pandas(df_val[['email', 'phishing']].rename(columns={'phishing': 'labels'}),
                                  preserve_index=False).map(tokenize_fn, batched=True, remove_columns=['email'])
    test_ds = Dataset.from_pandas(df_test[['email', 'phishing']].rename(columns={'phishing': 'labels'}),
                                   preserve_index=False).map(tokenize_fn, batched=True, remove_columns=['email'])

    return train_ds, val_ds, test_ds, df_test[['email', 'phishing']].rename(columns={'phishing': 'labels'}), class_weights

# --- METRICS ---
def compute_metrics_fn(p):
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    probs = torch.nn.functional.softmax(torch.from_numpy(p.predictions), dim=-1)[:, 1].numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}

# --- MAIN EXECUTION ---
print("Loading tokenizer and data...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_ds, val_ds, test_ds, test_df_raw, class_weights = load_and_preprocess_data(tokenizer)

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

# Train 2 models: Baseline and Hybrid (FGM + Char Noise)
# FGM Only model is commented out
models_to_train = [
    ("Baseline", WeightedTrainer, {"class_weights": class_weights}),
    # ("FGM Only", HybridTrainer, {"class_weights": class_weights}),  # COMMENTED OUT
    ("Hybrid (FGM+CharNoise)", HybridTrainer, {"class_weights": class_weights})
]

# Note: Baseline uses clean training data, Hybrid uses augmented data from load_and_preprocess_data

print("\n" + "="*80)
print("TRAINING MODELS")
print("="*80)

results = {}

# Reload clean training data for baseline
df_raw = load_dataset(DATASET_NAME)['train'].to_pandas()
df_raw.rename(columns={'Email Text': 'email', 'Email Type': 'phishing'}, inplace=True)
df_raw['phishing'] = df_raw['phishing'].map({'Safe Email': 0, 'Phishing Email': 1})
df_sampled, _ = train_test_split(df_raw, train_size=SAMPLES_TO_USE,
                                  stratify=df_raw['phishing'], random_state=SEED)
df_sampled['email'] = df_sampled['email'].apply(lambda x: x.lower() if isinstance(x, str) else '')
df_train_val, _ = train_test_split(df_sampled, test_size=TEST_RATIO,
                                    stratify=df_sampled['phishing'], random_state=SEED)
val_rel = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
df_train, df_val = train_test_split(df_train_val, test_size=val_rel,
                                     stratify=df_train_val['phishing'], random_state=SEED)

def tokenize_fn(examples):
    return tokenizer(examples['email'], truncation=True, padding='max_length', max_length=MAX_LENGTH)

train_ds_clean = Dataset.from_pandas(df_train[['email', 'phishing']].rename(columns={'phishing': 'labels'}),
                                preserve_index=False).map(tokenize_fn, batched=True, remove_columns=['email'])
val_ds_clean = Dataset.from_pandas(df_val[['email', 'phishing']].rename(columns={'phishing': 'labels'}),
                              preserve_index=False).map(tokenize_fn, batched=True, remove_columns=['email'])

trained_models = []

for model_name, trainer_class, extra_args in models_to_train:
    print(f"\nTraining {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    # Use clean data for Baseline, augmented for Hybrid
    if model_name == "Hybrid (FGM+CharNoise)":
        trainer = trainer_class(model=model, args=args, train_dataset=train_ds,
                               eval_dataset=val_ds, tokenizer=tokenizer,
                               compute_metrics=compute_metrics_fn, **extra_args)
    else:
        trainer = trainer_class(model=model, args=args, train_dataset=train_ds_clean,
                               eval_dataset=val_ds_clean, tokenizer=tokenizer,
                               compute_metrics=compute_metrics_fn, **extra_args)

    trainer.train()
    metrics = trainer.evaluate(test_ds)
    results[model_name] = metrics
    trained_models.append((model_name, model))
    print(f"{model_name} - Test Accuracy: {metrics['eval_accuracy']:.4f}")

# --- TABLE 1: Clean Test Performance ---
print("\n" + "="*95)
print("Table 1: Performance on Clean Test Data")
print("="*95)
print(f"| {'Model':<30} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'AUC':<6} |")
print("|" + "-"*93 + "|")

# Updated to only show Baseline and Hybrid
for model_name in ["Baseline", "Hybrid (FGM+CharNoise)"]:  # Removed "FGM Only"
    m = results[model_name]
    print(f"| {model_name:<30} | {m['eval_accuracy']:<6.3f} | {m['eval_precision']:<6.3f} | "
          f"{m['eval_recall']:<6.3f} | {m['eval_f1']:<6.3f} | {m['eval_auc']:<6.3f} |")

print("="*95 + "\n")

# --- TABLE 2: Robustness to Character Noise ---
print("="*85)
print("Table 2: Accuracy Under Character-Level Noise")
print("="*85)
print(f"| {'Model':<30} | {'0%':<10} | {'5%':<10} | {'10%':<10} | {'20%':<10} |")
print("|" + "-"*83 + "|")

datasets.utils.logging.disable_progress_bar()
warnings.filterwarnings("ignore", message=".*pin_memory.*")

eval_args = TrainingArguments(
    output_dir="./temp",
    report_to="none",
    disable_tqdm=True,
    log_level="error"
)

for model_name, model in trained_models:
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

    print(f"| {model_name:<30} | {row_accs[0]:<10} | {row_accs[1]:<10} | {row_accs[2]:<10} | {row_accs[3]:<10} |")

print("="*85 + "\n")

print("Training complete")
print(f"Class weights applied: Legit={class_weights[0]:.4f}, Phish={class_weights[1]:.4f}")

