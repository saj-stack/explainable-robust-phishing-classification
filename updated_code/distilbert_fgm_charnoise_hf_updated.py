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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# --- CONFIGURATION ---
logging.basicConfig(level=logging.ERROR)
os.environ["WANDB_DISABLED"] = "true"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SAMPLES_TO_USE = 16000 # Change the sample size here
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16 # Increase to 32 if memory allows
GRADIENT_ACC_STEPS = 1
NUM_TRAIN_EPOCHS = 3 # Increase to 5+ for final runs

# --- NOISE FUNCTIONS ---
SUBSTITUTIONS = {'o': '0', 'l': '1', 'e': '3', 'a': '@', 's': '$', 'i': '1', 'z': '2', 'g': '9', 't': '7'}
ALPHANUM = string.ascii_lowercase + string.digits

def inject_character_noise(text, noise_level):
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
    def __init__(self, class_weights=None, use_fgm=True, epsilon=0.1, adv_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.use_fgm = use_fgm
        self.epsilon = epsilon
        self.adv_weight = adv_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        
        # 1. Standard Forward Pass (Clean)
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss_clean = loss_fct(logits.view(-1, 2), labels.view(-1))

        # 2. Adversarial Training (FGM) - Only if enabled and in training mode
        if self.use_fgm and model.training:
            # Get word embeddings
            embeddings = model.get_input_embeddings().weight
            
            # Use autograd.grad to avoid double-backward issues
            # We need the gradient of the loss with respect to the embeddings
            # Note: This requires the model to return hidden states or we use a hook
            # For simplicity in DistilBert, we can perturb the 'inputs_embeds' instead
            
            # Step A: Get original embeddings for the current batch
            input_ids = inputs.get("input_ids")
            original_embeds = model.distilbert.embeddings(input_ids)
            
            # Step B: Calculate gradient w.r.t embeddings
            original_embeds.retain_grad()
            clean_outputs = model(inputs_embeds=original_embeds, attention_mask=inputs.get("attention_mask"), labels=labels)
            clean_loss = loss_fct(clean_outputs.logits.view(-1, 2), labels.view(-1))
            
            # Explicitly compute gradients for embeddings only
            grads = torch.autograd.grad(clean_loss, original_embeds, retain_graph=True)[0]
            
            # Step C: Perturb
            norm = torch.norm(grads, p=2, dim=-1, keepdim=True) + 1e-10
            perturbation = self.epsilon * grads / norm
            adv_embeds = original_embeds + perturbation
            
            # Step D: Adversarial Forward Pass
            adv_outputs = model(inputs_embeds=adv_embeds, attention_mask=inputs.get("attention_mask"), labels=labels)
            loss_adv = loss_fct(adv_outputs.logits.view(-1, 2), labels.view(-1))
            
            # Total Loss: Clean + Lambda * Adv
            total_loss = loss_clean + self.adv_weight * loss_adv
        else:
            total_loss = loss_clean

        return (total_loss, outputs) if return_outputs else total_loss

# --- DATA PREPARATION ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

print("Loading and sampling dataset...")
raw_ds = load_dataset("zefang-liu/phishing-email-dataset", split='train').shuffle(seed=SEED).select(range(SAMPLES_TO_USE))
df = raw_ds.to_pandas().rename(columns={'Email Text': 'email', 'Email Type': 'phishing'})
df['phishing'] = df['phishing'].map({'Safe Email': 0, 'Phishing Email': 1})
df['email'] = df['email'].fillna('').astype(str).str.lower()

df_train, df_test = train_test_split(df, test_size=0.15, stratify=df['phishing'], random_state=SEED)
df_train, df_val = train_test_split(df_train, test_size=0.15, stratify=df_train['phishing'], random_state=SEED)

class_weights = torch.tensor(
    compute_class_weight('balanced', classes=np.array([0, 1]), y=df_train['phishing'].values), 
    dtype=torch.float32
)

def preprocess_to_ds(df_in, augment=False):
    emails = df_in['email'].tolist()
    if augment:
        emails = [inject_character_noise(e, 0.1) if random.random() < 0.3 else e for e in emails]
    tokenized = tokenizer(emails, truncation=True, max_length=MAX_LENGTH)
    return Dataset.from_dict({
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': df_in['phishing'].tolist()
    })

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

# 1. Baseline Model (use_fgm=False)
print("\n>>> Training True Baseline (No FGM, No CharNoise)...")
train_ds = preprocess_to_ds(df_train, augment=False)
val_ds = preprocess_to_ds(df_val)
model_baseline = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

trainer_b = HybridTrainer(
    model=model_baseline, 
    args=args, 
    train_dataset=train_ds, 
    eval_dataset=val_ds, 
    class_weights=class_weights, 
    data_collator=data_collator,
    use_fgm=False # True Baseline
)
trainer_b.train()

del train_ds
gc.collect()

# 2. Hybrid Model (use_fgm=True + CharNoise)
print("\n>>> Training Hybrid (FGM + CharNoise)...")
train_ds_aug = preprocess_to_ds(df_train, augment=True)
model_hybrid = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

trainer_h = HybridTrainer(
    model=model_hybrid, 
    args=args, 
    train_dataset=train_ds_aug, 
    eval_dataset=val_ds, 
    class_weights=class_weights, 
    data_collator=data_collator,
    use_fgm=True # Adversarial enabled
)
trainer_h.train()

# --- ROBUSTNESS EVALUATION ---
print("\n" + "="*50)
print("Robustness Under Character-Level Noise")
print("="*50)
print(f"{'Model':<15} | 0% | 5% | 10% | 20%")
print("-" * 45)

test_emails = df_test['email'].tolist()
test_labels = df_test['phishing'].tolist()

for name, model in [("Baseline", model_baseline), ("Hybrid", model_hybrid)]:
    model.eval()
    model.to("cuda")
    acc_row = []
    
    for noise in [0.0, 0.05, 0.10, 0.20]:
        noisy_text = [inject_character_noise(e, noise) for e in test_emails]
        all_preds = []
        for i in range(0, len(noisy_text), 32):
            batch = noisy_text[i:i+32]
            inputs = tokenizer(batch, truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits = model(**inputs).logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
        acc_row.append(f"{accuracy_score(test_labels, all_preds):.4f}")
    
    print(f"{name:<15} | {acc_row[0]} | {acc_row[1]} | {acc_row[2]} | {acc_row[3]}")
