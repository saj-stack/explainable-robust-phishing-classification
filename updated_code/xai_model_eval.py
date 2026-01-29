!pip install transformers datasets accelerate torch numpy pandas scikit-learn lime spacy shap captum
!python -m spacy download en_core_web_sm

import os, logging, random, re, string, torch, time, tracemalloc, gc
import numpy as np
import pandas as pd
import spacy
import shap
from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lime.lime_text import LimeTextExplainer
from captum.attr import IntegratedGradients

# --- 1. CONFIGURATIONS ---
os.environ["WANDB_DISABLED"] = "true"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATASET_NAME = "zefang-liu/phishing-email-dataset"
SAMPLES_TO_USE = 10000
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128 #Increase to 256 for final runs
BATCH_SIZE = 16 # Increase to 32 if memory allows
NUM_TRAIN_EPOCHS = 3 # Increase to 5+ for final runs
FGM_EPSILON = 0.1
ADV_LOSS_WEIGHT = 0.5
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
LEARNING_RATE = 3e-5

# XAI EFFORT PARAMETERS
LIME_SAMPLES = 800
SHAP_MAX_EVALS = 400
IG_STEPS = 100

# NOISE PARAMETERS FOR TRAINING (30% samples / 10% noise)
TRAIN_NOISE_SAMPLE_PCT = 0.30
TRAIN_NOISE_INTENSITY = 0.10

SUBSTITUTIONS = {'o': '0', 'l': '1', 'e': '3', 'a': '@', 's': '$', 'i': '1', 'z': '2', 'g': '9', 't': '7'}
ALPHANUM = string.ascii_lowercase + string.digits

# --- 2. NOISE INJECTION ---
def inject_character_noise(text: str, noise_level: float) -> str:
    if noise_level == 0.0 or not isinstance(text, str) or not text:
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

def mask_sensitive_info(text: str) -> str:
    if not text: return ""
    text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[EMAIL]', text)
    text = re.sub(r'\b(?:\+?\d[\d\-\s\(\)]{6,}\d)\b', '[PHONE]', text)
    return text

# --- 3. DATA PREP ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
raw_dataset = load_dataset(DATASET_NAME)
df_raw = raw_dataset['train'].to_pandas()
df_raw.rename(columns={'Email Text': 'email', 'Email Type': 'phishing'}, inplace=True)
df_raw['phishing'] = df_raw['phishing'].map({'Safe Email': 0, 'Phishing Email': 1})
df_sampled = df_raw.sample(n=min(SAMPLES_TO_USE, len(df_raw)), random_state=SEED)
df_sampled['email'] = df_sampled['email'].apply(lambda x: mask_sensitive_info(x).lower() if isinstance(x, str) else '')

df_train_val, _ = train_test_split(df_sampled, test_size=TEST_RATIO, random_state=SEED, stratify=df_sampled['phishing'])
df_train, df_val = train_test_split(df_train_val, test_size=VAL_RATIO/(TRAIN_RATIO+VAL_RATIO), random_state=SEED, stratify=df_train_val['phishing'])

df_train_noisy = df_train.copy()
num_noisy = int(len(df_train_noisy) * TRAIN_NOISE_SAMPLE_PCT)
noisy_indices = df_train_noisy.sample(n=num_noisy, random_state=SEED).index
df_train_noisy.loc[noisy_indices, 'email'] = df_train_noisy.loc[noisy_indices, 'email'].apply(lambda x: inject_character_noise(x, TRAIN_NOISE_INTENSITY))

def tokenize_fn(ex): return tokenizer(ex['email'], truncation=True, padding='max_length', max_length=MAX_LENGTH)

train_ds_clean = Dataset.from_pandas(df_train[['email', 'phishing']].rename(columns={'phishing': 'labels'}), preserve_index=False).map(tokenize_fn, batched=True, remove_columns=['email'])
train_ds_noisy = Dataset.from_pandas(df_train_noisy[['email', 'phishing']].rename(columns={'phishing': 'labels'}), preserve_index=False).map(tokenize_fn, batched=True, remove_columns=['email'])
val_ds = Dataset.from_pandas(df_val[['email', 'phishing']].rename(columns={'phishing': 'labels'}), preserve_index=False).map(tokenize_fn, batched=True, remove_columns=['email'])

# --- 4. FGM LOGIC ---
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
        for name, param in self.backup.items():
            self.model.get_parameter(name).data.copy_(param)
        self.backup = {}

class FGMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self.fgm = FGM(model, epsilon=FGM_EPSILON)
        outputs = model(**inputs)
        loss_clean = outputs.loss
        if loss_clean.requires_grad:
            loss_clean.backward(retain_graph=True)
            self.fgm.attack()
            loss_adv = model(**inputs).loss
            self.fgm.restore()
            loss_clean = loss_clean + ADV_LOSS_WEIGHT * loss_adv
        return (loss_clean, outputs) if return_outputs else loss_clean

# --- 5. TRAINING ---
print("\n>>> Training BASELINE")
baseline_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
baseline_trainer = Trainer(
    model=baseline_model,
    args=TrainingArguments(output_dir="./baseline", per_device_train_batch_size=BATCH_SIZE, num_train_epochs=NUM_TRAIN_EPOCHS, eval_strategy="epoch", report_to="none"),
    train_dataset=train_ds_clean, eval_dataset=val_ds, tokenizer=tokenizer,
    compute_metrics=lambda p: {"accuracy": accuracy_score(p[1], np.argmax(p[0], axis=-1))}
)
baseline_trainer.train()

print(f"\n>>> Training FGM-ROBUST")
fgm_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
fgm_trainer = FGMTrainer(
    model=fgm_model,
    args=TrainingArguments(output_dir="./fgm_robust", per_device_train_batch_size=BATCH_SIZE, num_train_epochs=NUM_TRAIN_EPOCHS, eval_strategy="epoch", report_to="none"),
    train_dataset=train_ds_noisy, eval_dataset=val_ds, tokenizer=tokenizer,
    compute_metrics=lambda p: {"accuracy": accuracy_score(p[1], np.argmax(p[0], axis=-1))}
)
fgm_trainer.train()

# --- 6. XAI PREDICTOR ---
class Predictor:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model.eval(), tokenizer
        self.device = next(model.parameters()).device
    def predict_proba(self, texts):
        inputs = self.tokenizer(list(texts), truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs).logits
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        return probs

def predict_forward_func(inputs, model):
    return model(inputs_embeds=inputs).logits

models = {"BASELINE": baseline_model, "FGM-ROBUST": fgm_model}
explainer_lime = LimeTextExplainer(class_names=['LEGITIMATE', 'PHISHING'])

def is_valid_token(token):
    # Filters out special tokens, subword indicators, and punctuation
    token = token.lower().strip()
    if token in ['[cls]', '[sep]', '[pad]', '[unk]', '##']:
        return False
    if token.startswith('##'):
        return False
    # Ensure it contains at least one alphabetic character and no purely punctuation strings
    if not any(char.isalpha() for char in token):
        return False
    # Check if the token is just punctuation
    if all(char in string.punctuation for char in token):
        return False
    return len(token) >= 2

# --- 7. INTERACTIVE LOOP ---
while True:
    email_body = input('\n>>> Enter Email (or "quit"): ')
    if email_body.lower() == 'quit': break

    input_text = mask_sensitive_info(email_body).lower()

    for name, model in models.items():
        print(f"\n{'='*25} MODEL: {name} {'='*25}")
        pred_obj = Predictor(model, tokenizer)
        ig = IntegratedGradients(predict_forward_func)

        probs = pred_obj.predict_proba([input_text])[0]
        pred_idx = np.argmax(probs)
        target_class = int(pred_idx)

        label = "PHISHING" if target_class == 1 else "LEGITIMATE"
        print(f"PREDICTION: {label} (Conf: {probs[target_class]:.4f})")

        # LIME
        exp_lime = explainer_lime.explain_instance(input_text, pred_obj.predict_proba, num_samples=LIME_SAMPLES)

        # SHAP
        explainer_shap = shap.Explainer(lambda x: pred_obj.predict_proba(x), tokenizer)
        shap_values = explainer_shap([input_text], max_evals=SHAP_MAX_EVALS)

        # IG
        inputs_ig = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(model.device)
        embeddings = model.get_input_embeddings()(inputs_ig['input_ids'])
        attributions = ig.attribute(embeddings, target=target_class, n_steps=IG_STEPS, additional_forward_args=(model,))
        token_attrs = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()

        print("\n[LIME TOP 10]")
        lime_list = [(t, s) for t, s in exp_lime.as_list() if is_valid_token(t)]
        for t, s in lime_list[:10]: print(f"  {t:20} : {s:+0.4f}")

        print("\n[SHAP TOP 10]")
        s_vals, s_tokens = shap_values.values[0][:, target_class], shap_values.data[0]
        s_idx = np.argsort(np.abs(s_vals))[::-1]
        count = 0
        for i in s_idx:
            if is_valid_token(s_tokens[i]) and count < 10:
                print(f"  {s_tokens[i]:20} : {s_vals[i]:+0.4f}")
                count += 1

        print("\n[IG TOP 10]")
        ig_tokens = tokenizer.convert_ids_to_tokens(inputs_ig['input_ids'][0])
        ig_idx = np.argsort(np.abs(token_attrs))[::-1]
        count = 0
        for i in ig_idx:
            if is_valid_token(ig_tokens[i]) and count < 10:
                print(f"  {ig_tokens[i]:20} : {token_attrs[i]:+0.4f}")
                count += 1

    gc.collect()
