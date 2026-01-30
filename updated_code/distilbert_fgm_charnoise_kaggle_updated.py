import kagglehub
import pandas as pd
import os
import logging
import random
import string
import gc
import contextlib
import numpy as np
import torch
from datasets import Dataset
import datasets
import warnings
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# --- CONFIGURATION ---
logging.basicConfig(level=logging.ERROR)
os.environ["WANDB_DISABLED"] = "true"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

KAGGLE_DATASET_ID = "naserabdullahalam/phishing-email-dataset"
SAMPLES_TO_USE = 15000
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
GRADIENT_ACC_STEPS = 1
NUM_TRAIN_EPOCHS = 3

# --- NOISE INJECTION ---
SUBSTITUTIONS = {'o': '0', 'l': '1', 'e': '3', 'a': '@', 's': '$', 'i': '1', 'z': '2', 'g': '9', 't': '7'}
ALPHANUM = string.ascii_lowercase + string.digits

def inject_character_noise(text: str, noise_level: float) -> str:
    if not text or noise_level == 0.0: return text
    text_list = list(text)
    num_to_corrupt = max(1, int(len(text_list) * noise_level))
    indices = random.sample(range(len(text_list)), min(num_to_corrupt, len(text_list)))
    for idx in sorted(indices, reverse=True):
        op = random.choice(['deletion', 'substitution', 'insertion'])
        if op == 'deletion' and len(text_list) > 1: text_list.pop(idx)
        elif op == 'substitution': text_list[idx] = SUBSTITUTIONS.get(text_list[idx].lower(), random.choice(ALPHANUM))
        elif op == 'insertion': text_list.insert(idx + 1, random.choice(ALPHANUM))
    return "".join(text_list)

# --- REFACTORED HYBRID TRAINER ---
class HybridTrainer(Trainer):
    def __init__(self, class_weights=None, use_fgm=True, epsilon=0.01, adv_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.use_fgm = use_fgm
        self.epsilon = epsilon
        self.adv_weight = adv_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        
        # 1. Clean Forward Pass
        outputs = model(**inputs)
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss_clean = loss_fct(outputs.logits.view(-1, 2), labels.view(-1))

        # 2. Adversarial Training (FGM) - Only if enabled and training
        if self.use_fgm and model.training:
            # Get original embeddings for the batch
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")
            original_embeds = model.distilbert.embeddings(input_ids)
            
            # Use autograd.grad to find perturbation without double-backward
            original_embeds.retain_grad()
            temp_outputs = model(inputs_embeds=original_embeds, attention_mask=attention_mask, labels=labels)
            temp_loss = loss_fct(temp_outputs.logits.view(-1, 2), labels.view(-1))
            
            # Compute gradient of loss w.r.t embeddings
            grads = torch.autograd.grad(temp_loss, original_embeds, retain_graph=True)[0]
            
            # Apply Perturbation
            norm = torch.norm(grads, p=2, dim=-1, keepdim=True) + 1e-10
            perturbation = self.epsilon * grads / norm
            adv_embeds = original_embeds + perturbation
            
            # Adversarial Forward Pass
            adv_outputs = model(inputs_embeds=adv_embeds, attention_mask=attention_mask, labels=labels)
            loss_adv = loss_fct(adv_outputs.logits.view(-1, 2), labels.view(-1))
            
            # Combine losses (HuggingFace Trainer will call backward() ONCE on this total)
            total_loss = loss_clean + self.adv_weight * loss_adv
        else:
            total_loss = loss_clean

        return (total_loss, outputs) if return_outputs else total_loss

# --- DATA PREPARATION ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def load_data():
    path = kagglehub.dataset_download(KAGGLE_DATASET_ID)
    df = pd.read_csv(os.path.join(path, "CEAS_08.csv"))
    df = df[['body', 'label']].rename(columns={'body': 'email', 'label': 'labels'}).dropna()
    
    # Sample and Balance
    min_size = min(len(df[df['labels']==1]), len(df[df['labels']==0]), SAMPLES_TO_USE // 2)
    df_balanced = pd.concat([
        df[df['labels']==1].sample(min_size, random_state=SEED),
        df[df['labels']==0].sample(min_size, random_state=SEED)
    ]).sample(frac=1, random_state=SEED)
    
    df_balanced['email'] = df_balanced['email'].astype(str).str.lower()
    return train_test_split(df_balanced, test_size=0.3, stratify=df_balanced['labels'], random_state=SEED)

train_df, temp_df = load_data()
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['labels'], random_state=SEED)

class_weights = torch.tensor(compute_class_weight('balanced', classes=np.array([0, 1]), y=train_df['labels'].values), dtype=torch.float32)

def preprocess(df_in, augment=False):
    emails = df_in['email'].tolist()
    if augment:
        emails = [inject_character_noise(e, 0.1) if random.random() < 0.3 else e for e in emails]
    tokenized = tokenizer(emails, truncation=True, max_length=MAX_LENGTH)
    return Dataset.from_dict({'input_ids': tokenized['input_ids'], 'attention_mask': tokenized['attention_mask'], 'labels': df_in['labels'].tolist()})

# --- TRAINING ---
args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACC_STEPS,
    fp16=True,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    eval_strategy="epoch",
    save_strategy="no",
    report_to="none"
)

# 1. Baseline
print("\n>>> Training True Baseline (No FGM)...")
train_ds = preprocess(train_df, augment=False)
val_ds = preprocess(val_df)
model_b = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
trainer_b = HybridTrainer(model=model_b, args=args, train_dataset=train_ds, eval_dataset=val_ds, class_weights=class_weights, data_collator=data_collator, use_fgm=False)
trainer_b.train()

# 2. Hybrid
print("\n>>> Training Hybrid (FGM + CharNoise)...")
train_ds_aug = preprocess(train_df, augment=True)
model_h = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
trainer_h = HybridTrainer(model=model_h, args=args, train_dataset=train_ds_aug, eval_dataset=val_ds, class_weights=class_weights, data_collator=data_collator, use_fgm=True)
trainer_h.train()

# --- EVALUATION ---
print("\nRobustness Comparison:")
test_emails = test_df['email'].tolist()
test_labels = test_df['labels'].tolist()

for name, model in [("Baseline", model_b), ("Hybrid", model_h)]:
    model.eval().to("cuda")
    accs = []
    for noise in [0.0, 0.05, 0.1, 0.2]:
        noisy = [inject_character_noise(e, noise) for e in test_emails]
        preds = []
        for i in range(0, len(noisy), 32):
            batch = tokenizer(noisy[i:i+32], truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt").to("cuda")
            with torch.no_grad():
                preds.extend(torch.argmax(model(**batch).logits, dim=1).cpu().numpy())
        accs.append(accuracy_score(test_labels, preds))
    print(f"{name:<10} | 0%: {accs[0]:.4f} | 5%: {accs[1]:.4f} | 10%: {accs[2]:.4f} | 20%: {accs[3]:.4f}")
