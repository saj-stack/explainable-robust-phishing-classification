import os, logging, random, re, string, torch, time, tracemalloc
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
MAX_LENGTH = 128
BATCH_SIZE = 32
NUM_TRAIN_EPOCHS = 3
FGM_EPSILON = 0.01
ADV_LOSS_WEIGHT = 0.3
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 50

try:
    NER_NLP = spacy.load("en_core_web_sm")
except:
    NER_NLP = None

# --- 2. UTILITIES & PREPROCESSING ---
def mask_sensitive_info(text: str) -> str:
    if not text: return ""
    text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[EMAIL]', text)
    text = re.sub(r'\b(?:\+?\d[\d\-\s\(\)]{6,}\d)\b', '[PHONE]', text)
    text = re.sub(r'\b\d{7,}\b', '[ACCOUNT]', text)
    if NER_NLP:
        doc = NER_NLP(text)
        masked_text = text
        for ent in reversed(doc.ents):
            replacement = "[NAME]" if ent.label_ == "PERSON" else None
            if ent.label_ in ["CARDINAL", "MONEY", "QUANTITY"]:
                digits = re.sub(r'\D', '', ent.text)
                if len(digits) >= 6: replacement = "[ACCOUNT]"
            if replacement:
                masked_text = masked_text[:ent.start_char] + replacement + masked_text[ent.end_char:]
        return masked_text
    return text

def load_and_preprocess_data(tokenizer):
    raw_dataset = load_dataset(DATASET_NAME)
    df_raw = raw_dataset['train'].to_pandas()
    df_raw.rename(columns={'Email Text': 'email', 'Email Type': 'phishing'}, inplace=True)
    df_raw['phishing'] = df_raw['phishing'].map({'Safe Email': 0, 'Phishing Email': 1})
    df_sampled = df_raw.sample(n=min(SAMPLES_TO_USE, len(df_raw)), random_state=SEED)
    df_sampled['email'] = df_sampled['email'].apply(lambda x: mask_sensitive_info(x).lower() if isinstance(x, str) else '')
    df_train_val, _ = train_test_split(df_sampled, test_size=TEST_RATIO, random_state=SEED, stratify=df_sampled['phishing'])
    df_train, df_val = train_test_split(df_train_val, test_size=VAL_RATIO/(TRAIN_RATIO+VAL_RATIO), random_state=SEED, stratify=df_train_val['phishing'])
    def tokenize_fn(ex): return tokenizer(ex['email'], truncation=True, padding='max_length', max_length=MAX_LENGTH)
    train_ds = Dataset.from_pandas(df_train[['email', 'phishing']].rename(columns={'phishing': 'labels'}), preserve_index=False).map(tokenize_fn, batched=True, remove_columns=['email'])
    val_ds = Dataset.from_pandas(df_val[['email', 'phishing']].rename(columns={'phishing': 'labels'}), preserve_index=False).map(tokenize_fn, batched=True, remove_columns=['email'])
    return train_ds, val_ds

# --- 3. FGM ROBUST TRAINING LOGIC ---
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

# --- 4. MODEL TRAINING ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_ds, val_ds = load_and_preprocess_data(tokenizer)

print("\n--- Training Baseline DistilBERT ---")
baseline_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
baseline_trainer = Trainer(
    model=baseline_model,
    args=TrainingArguments(output_dir="./baseline", per_device_train_batch_size=BATCH_SIZE, num_train_epochs=NUM_TRAIN_EPOCHS, 
                           learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, warmup_steps=WARMUP_STEPS, eval_strategy="epoch", report_to="none"),
    train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer,
    compute_metrics=lambda p: {"accuracy": accuracy_score(p[1], np.argmax(p[0], axis=-1))}
)
baseline_trainer.train()

print("\n--- Training FGM-Robust DistilBERT ---")
fgm_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
fgm_trainer = FGMTrainer(
    model=fgm_model,
    args=TrainingArguments(output_dir="./fgm_robust", per_device_train_batch_size=BATCH_SIZE, num_train_epochs=NUM_TRAIN_EPOCHS, 
                           learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, warmup_steps=WARMUP_STEPS, eval_strategy="epoch", report_to="none"),
    train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer,
    compute_metrics=lambda p: {"accuracy": accuracy_score(p[1], np.argmax(p[0], axis=-1))}
)
fgm_trainer.train()

# --- 5. XAI PREPARATIONS ---
class Predictor:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model.eval(), tokenizer
        self.device = next(model.parameters()).device
    def predict_proba(self, texts):
        inputs = self.tokenizer(list(texts), truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt").to(self.device)
        with torch.no_grad():
            return torch.softmax(self.model(**inputs).logits, dim=1).cpu().numpy()

def predict_forward_func(inputs, model):
    return model(inputs_embeds=inputs).logits

# MOVE INITIALIZATION OUTSIDE THE LOOP
models_to_compare = {"BASELINE": baseline_model, "FGM-ROBUST": fgm_model}
explainer_lime = LimeTextExplainer(class_names=['LEGITIMATE', 'PHISHING'], random_state=SEED)
ig = IntegratedGradients(predict_forward_func)

# Pre-initialize SHAP Explainers to avoid memory leaks
shap_explainers = {}
for name, model in models_to_compare.items():
    p = Predictor(model, tokenizer)
    shap_explainers[name] = shap.Explainer(lambda x: p.predict_proba(x), tokenizer)

# --- 6. INTERACTIVE LOOP ---
while True:
    try:
        email_body = input('\n>>> Enter Email (or "quit"): ')
        if email_body.lower() in ['quit', 'exit']: break
        if not email_body.strip(): continue

        input_text = mask_sensitive_info(email_body).lower()

        for name, model in models_to_compare.items():
            print(f"\n{'='*40}\nMODEL: {name}\n{'='*40}")
            predictor = Predictor(model, tokenizer)
            
            probs = predictor.predict_proba([input_text])[0]
            pred_idx = int(np.argmax(probs))
            print(f"PREDICTION: {['LEGITIMATE', 'PHISHING'][pred_idx]} (Confidence: {probs[pred_idx]:.4f})")

            # Benchmarking LIME
            tracemalloc.start()
            start_t = time.time()
            exp_lime = explainer_lime.explain_instance(input_text, predictor.predict_proba, num_samples=1500)
            lime_time, (_, lime_peak) = time.time() - start_t, tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Benchmarking SHAP (Reusing pre-initialized explainer)
            tracemalloc.start()
            start_t = time.time()
            shap_values = shap_explainers[name]([input_text], max_evals=500)
            shap_time, (_, shap_peak) = time.time() - start_t, tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Benchmarking IG
            tracemalloc.start()
            start_t = time.time()
            inputs_ig = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(model.device)
            embeddings = model.get_input_embeddings()(inputs_ig['input_ids'])
            attributions = ig.attribute(embeddings, target=pred_idx, n_steps=50, additional_forward_args=(model,))
            token_attrs = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
            ig_time, (_, ig_peak) = time.time() - start_t, tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Display Results
            print("\n[METHOD 1: LIME TOP 10 TOKENS]")
            for t, s in [(t, s) for t, s in exp_lime.as_list() if len(t.strip(string.punctuation)) >= 2][:10]:
                print(f" {t:30s} {s:+0.4f}")

            print("\n[METHOD 2: SHAP TOP 10 TOKENS]")
            s_vals, s_tokens = shap_values.values[0][:, pred_idx], shap_values.data[0]
            for i in np.argsort(np.abs(s_vals))[::-1][:10]:
                print(f" {s_tokens[i]:30s} {s_vals[i]:+.4f}")

            print("\n[METHOD 3: IG TOP 10 TOKENS]")
            all_tokens = tokenizer.convert_ids_to_tokens(inputs_ig['input_ids'][0])
            filtered_ig = [i for i in np.argsort(np.abs(token_attrs))[::-1] if all_tokens[i] not in tokenizer.all_special_tokens and len(all_tokens[i].strip(string.punctuation)) >= 2]
            for i in filtered_ig[:10]:
                print(f" {all_tokens[i]:30s} {token_attrs[i]:+.4f}")

            print("\n" + "-"*85)
            print(f"{'Method':<12} | {'Time (s)':<10} | {'Mem (KB)':<10}")
            print("-" * 85)
            print(f"{'LIME':<12} | {lime_time:<10.3f} | {lime_peak/1024:<10.2f}")
            print(f"{'SHAP':<12} | {shap_time:<10.3f} | {shap_peak/1024:<10.2f}")
            print(f"{'IG':<12} | {ig_time:<10.3f} | {ig_peak/1024:<10.2f}")

            # Memory Management Cleanup
            del attributions, embeddings, inputs_ig
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    except Exception as e:
        print(f"An error occurred: {e}. Moving to next email.")
        continue

