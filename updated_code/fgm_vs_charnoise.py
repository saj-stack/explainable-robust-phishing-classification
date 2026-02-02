import os
import logging
import random
import string
import gc
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# =============================================================================
# CONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.ERROR)
os.environ["WANDB_DISABLED"] = "true"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SAMPLES_TO_USE     = 16000
MODEL_NAME         = "distilbert-base-uncased"
MAX_LENGTH         = 256
BATCH_SIZE         = 16                # increase to 32 if memory allows
GRADIENT_ACC_STEPS = 1
NUM_TRAIN_EPOCHS   = 3                 # increase to 5+ for final runs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# CHARACTER-NOISE HELPERS
# =============================================================================
SUBSTITUTIONS = {
    'o': '0', 'l': '1', 'e': '3', 'a': '@',
    's': '$', 'i': '1', 'z': '2', 'g': '9', 't': '7'
}
ALPHANUM = string.ascii_lowercase + string.digits


def inject_character_noise(text, noise_level):
    """Corrupt noise_level fraction of characters via deletion / substitution / insertion."""
    if not text or noise_level == 0.0:
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


# =============================================================================
# TRAINER CLASSES
# =============================================================================

# -----------------------------------------------------------------------------
# 1. Baseline Trainer  —  weighted CE loss only, nothing else
# -----------------------------------------------------------------------------
class BaselineTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss     = loss_fct(logits.view(-1, 2), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# -----------------------------------------------------------------------------
# 2. FGM Trainer  —  weighted CE loss + adversarial perturbation in embedding space
# -----------------------------------------------------------------------------
class FGMTrainer(Trainer):
    """
    FGM steps (per batch, training only):
      A. Compute original embeddings from input_ids.
      B. Forward pass using those embeddings  ->  clean_loss.
      C. Gradient of clean_loss w.r.t. embeddings  (autograd.grad).
      D. L2-normalise the gradient, scale by epsilon  ->  perturbation.
      E. Perturbed embeddings  ->  adversarial forward pass  ->  adv_loss.
      F. total_loss = clean_loss + adv_weight * adv_loss.
    """

    def __init__(self, class_weights=None, epsilon=0.1, adv_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.epsilon       = epsilon
        self.adv_weight    = adv_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")

        # --- standard forward (kept for the return value) ---
        outputs    = model(**inputs)
        logits     = outputs.get("logits")
        loss_fct   = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss_clean = loss_fct(logits.view(-1, 2), labels.view(-1))

        # --- FGM block (training only) ---
        if model.training:
            input_ids      = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")

            # A. original embeddings
            original_embeds = model.distilbert.embeddings(input_ids)

            # B. clean forward through embeddings path
            original_embeds.retain_grad()
            clean_out  = model(
                inputs_embeds=original_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
            clean_loss = loss_fct(clean_out.logits.view(-1, 2), labels.view(-1))

            # C. gradient w.r.t. embeddings
            grads = torch.autograd.grad(clean_loss, original_embeds, retain_graph=True)[0]

            # D. normalise & scale
            norm         = torch.norm(grads, p=2, dim=-1, keepdim=True) + 1e-10
            perturbation = self.epsilon * grads / norm

            # E. adversarial forward
            adv_embeds = original_embeds + perturbation
            adv_out    = model(
                inputs_embeds=adv_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss_adv = loss_fct(adv_out.logits.view(-1, 2), labels.view(-1))

            # F. combined loss
            total_loss = clean_loss + self.adv_weight * loss_adv
        else:
            total_loss = loss_clean

        return (total_loss, outputs) if return_outputs else total_loss


# -----------------------------------------------------------------------------
# 3. CharNoise Trainer  —  same loss as Baseline; the augmentation is in the data
# -----------------------------------------------------------------------------
class CharNoiseTrainer(Trainer):
    """Identical loss to BaselineTrainer.  The difference is that the training
    dataset passed to this trainer has already been augmented with char noise."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss     = loss_fct(logits.view(-1, 2), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# =============================================================================
# DATA PREPARATION
# =============================================================================
tokenizer     = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print("Loading and sampling dataset...")
raw_ds = (
    load_dataset("zefang-liu/phishing-email-dataset", split="train")
    .shuffle(seed=SEED)
    .select(range(SAMPLES_TO_USE))
)
df = raw_ds.to_pandas().rename(columns={"Email Text": "email", "Email Type": "phishing"})
df["phishing"] = df["phishing"].map({"Safe Email": 0, "Phishing Email": 1})
df["email"]    = df["email"].fillna("").astype(str).str.lower()

df_train, df_test = train_test_split(df, test_size=0.15, stratify=df["phishing"], random_state=SEED)
df_train, df_val  = train_test_split(df_train, test_size=0.15, stratify=df_train["phishing"], random_state=SEED)

class_weights = torch.tensor(
    compute_class_weight("balanced", classes=np.array([0, 1]), y=df_train["phishing"].values),
    dtype=torch.float32,
)


def preprocess_to_ds(df_in, augment=False):
    """Tokenise a DataFrame slice.
    augment=True  ->  30 % of samples get 10 % character noise before tokenisation."""
    emails = df_in["email"].tolist()
    if augment:
        emails = [
            inject_character_noise(e, 0.1) if random.random() < 0.3 else e
            for e in emails
        ]
    tokenized = tokenizer(emails, truncation=True, max_length=MAX_LENGTH)
    return Dataset.from_dict({
        "input_ids":      tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels":         df_in["phishing"].tolist(),
    })


# =============================================================================
# SHARED TrainingArguments FACTORY
# =============================================================================
def make_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACC_STEPS,
        fp16=True,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
    )


# =============================================================================
# TRAINING  —  three models, sequentially
# =============================================================================
val_ds = preprocess_to_ds(df_val, augment=False)   # shared, always clean

# ---------------------------------------------------------------------------
# MODEL 1  —  Baseline  (no FGM, no CharNoise)
# ---------------------------------------------------------------------------
print("\n>>> Training  [1/3]  Baseline  (no FGM, no CharNoise)…")
train_ds_clean = preprocess_to_ds(df_train, augment=False)

model_baseline = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

trainer_baseline = BaselineTrainer(
    model=model_baseline,
    args=make_training_args("./results_baseline"),
    train_dataset=train_ds_clean,
    eval_dataset=val_ds,
    class_weights=class_weights,
    data_collator=data_collator,
)
trainer_baseline.train()

del train_ds_clean, trainer_baseline
gc.collect()
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# MODEL 2  —  FGM Only  (adversarial training, clean data, no CharNoise)
# ---------------------------------------------------------------------------
print("\n>>> Training  [2/3]  FGM Only  (no CharNoise)…")
train_ds_clean = preprocess_to_ds(df_train, augment=False)   # clean again

model_fgm = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

trainer_fgm = FGMTrainer(
    model=model_fgm,
    args=make_training_args("./results_fgm"),
    train_dataset=train_ds_clean,
    eval_dataset=val_ds,
    class_weights=class_weights,
    data_collator=data_collator,
    epsilon=0.1,
    adv_weight=0.5,
)
trainer_fgm.train()

del train_ds_clean, trainer_fgm
gc.collect()
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# MODEL 3  —  CharNoise Only  (augmented data, no FGM)
# ---------------------------------------------------------------------------
print("\n>>> Training  [3/3]  CharNoise Only  (no FGM)…")
train_ds_aug = preprocess_to_ds(df_train, augment=True)      # augmented

model_charnoise = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

trainer_charnoise = CharNoiseTrainer(
    model=model_charnoise,
    args=make_training_args("./results_charnoise"),
    train_dataset=train_ds_aug,
    eval_dataset=val_ds,
    class_weights=class_weights,
    data_collator=data_collator,
)
trainer_charnoise.train()

del train_ds_aug, trainer_charnoise
gc.collect()
torch.cuda.empty_cache()


# =============================================================================
# ROBUSTNESS EVALUATION  —  identical test for all three models
# =============================================================================
print("\n" + "=" * 58)
print("   Robustness Evaluation Under Character-Level Noise")
print("=" * 58)

NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20]
EVAL_BATCH   = 32

test_emails = df_test["email"].tolist()
test_labels = df_test["phishing"].tolist()

# --- header ---
header = f"{'Model':<14} |"
for n in NOISE_LEVELS:
    header += f"  {int(n * 100):>3} % |"
print(header)
print("-" * len(header))

# --- loop over the three trained models ---
for name, model in [("Baseline", model_baseline),
                    ("FGM",      model_fgm),
                    ("CharNoise", model_charnoise)]:
    model.eval()
    model.to(DEVICE)
    acc_row = []

    for noise in NOISE_LEVELS:
        noisy_text = [inject_character_noise(e, noise) for e in test_emails]
        all_preds  = []

        for i in range(0, len(noisy_text), EVAL_BATCH):
            batch  = noisy_text[i : i + EVAL_BATCH]
            inputs = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(DEVICE)

            with torch.no_grad():
                logits = model(**inputs).logits
                preds  = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)

        acc_row.append(accuracy_score(test_labels, all_preds))

    # --- print row ---
    row = f"{name:<14} |"
    for a in acc_row:
        row += f" {a:.4f} |"
    print(row)

print("-" * len(header))
print("\nDone.")