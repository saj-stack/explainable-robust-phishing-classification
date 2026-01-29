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
MAX_LENGTH = 128
BATCH_SIZE = 16 # Increase to 32 if memory allows
NUM_TRAIN_EPOCHS = 3 # Increase to 5+ for final runs
FGM_EPSILON = 0.1 # Adversarial perturbation strength (try 0.01 or 0.1)
ADV_LOSS_WEIGHT = 0.5
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 50

# XAI Parameters
LIME_SAMPLES = 1000
SHAP_MAX_EVALS = 500
IG_STEPS = 100

# Character Noise Configuration
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.20]  # Test noise levels for robustness

try:
    NER_NLP = spacy.load("en_core_web_sm")
except:
    NER_NLP = None

# --- 2. CHARACTER NOISE INJECTION ---
SUBSTITUTIONS = {'o': '0', 'l': '1', 'e': '3', 'a': '@', 's': '$', 'i': '1', 'z': '2', 'g': '9', 't': '7'}
ALPHANUM = string.ascii_lowercase + string.digits

def inject_character_noise(text: str, noise_level: float) -> str:
    """
    Inject character-level noise into text.
    
    Args:
        text: Input text
        noise_level: Fraction of characters to corrupt (0.0 to 1.0)
    
    Returns:
        Noisy text with character substitutions, deletions, and insertions
    """
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

# --- 3. UTILITIES & PREPROCESSING ---
def mask_sensitive_info(text: str) -> str:
    if not text: return ""
    text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '[EMAIL]', text)
    text = re.sub(r'\b(?:\+?\d[\d\-\s\(\)]{6,}\d)\b', '[PHONE]', text)
    text = re.sub(r'\b\d{7,}\b', '[ACCOUNT]', text)
    # --- SPACY NER PART COMMENTED ---
    # if NER_NLP:
    #     doc = NER_NLP(text)
    #     masked_text = text
    #     for ent in reversed(doc.ents):
    #         replacement = "[NAME]" if ent.label_ == "PERSON" else None
    #         if ent.label_ in ["CARDINAL", "MONEY", "QUANTITY"]:
    #             digits = re.sub(r'\D', '', ent.text)
    #             if len(digits) >= 6: replacement = "[ACCOUNT]"
    #         if replacement:
    #             masked_text = masked_text[:ent.start_char] + replacement + masked_text[ent.end_char:]
    #     return masked_text
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

# --- 4. FGM ROBUST TRAINING LOGIC ---
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

# --- 5. MODEL TRAINING ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
train_ds, val_ds = load_and_preprocess_data(tokenizer)

print("\n--- Training Baseline ---")
baseline_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
baseline_trainer = Trainer(
    model=baseline_model,
    args=TrainingArguments(output_dir="./baseline", per_device_train_batch_size=BATCH_SIZE, num_train_epochs=NUM_TRAIN_EPOCHS,
                           learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, warmup_steps=WARMUP_STEPS, eval_strategy="epoch", report_to="none"),
    train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer,
    compute_metrics=lambda p: {"accuracy": accuracy_score(p[1], np.argmax(p[0], axis=-1))}
)
baseline_trainer.train()

print("\n--- Training FGM-Robust ---")
fgm_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
fgm_trainer = FGMTrainer(
    model=fgm_model,
    args=TrainingArguments(output_dir="./fgm_robust", per_device_train_batch_size=BATCH_SIZE, num_train_epochs=NUM_TRAIN_EPOCHS,
                           learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY, warmup_steps=WARMUP_STEPS, eval_strategy="epoch", report_to="none"),
    train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer,
    compute_metrics=lambda p: {"accuracy": accuracy_score(p[1], np.argmax(p[0], axis=-1))}
)
fgm_trainer.train()

# --- 6. XAI PREPARATIONS ---
class Predictor:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model.eval(), tokenizer
        self.device = next(model.parameters()).device
    
    def predict_proba(self, texts):
        inputs = self.tokenizer(list(texts), truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs).logits
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
        del inputs, outputs
        return probs

def predict_forward_func(inputs, model):
    return model(inputs_embeds=inputs).logits

models_to_compare = {"BASELINE": baseline_model, "FGM-ROBUST": fgm_model}
explainer_lime = LimeTextExplainer(class_names=['LEGITIMATE', 'PHISHING'], random_state=SEED)
ig = IntegratedGradients(predict_forward_func)

shap_explainers = {}
for name, model in models_to_compare.items():
    p = Predictor(model, tokenizer)
    shap_explainers[name] = shap.Explainer(lambda x: p.predict_proba(x), tokenizer)

# --- 7. INTERACTIVE LOOP WITH CHARACTER NOISE TESTING ---
print("\n" + "="*85)
print("PHISHING EMAIL DETECTION WITH XAI - Ready for input!")
print("="*85)
print("Commands:")
print("  - Enter email text for XAI analysis")
print("  - Type 'noise' to test current email with character noise")
print("  - Type 'quit' to exit")
print("="*85)

current_email = None
current_raw_text = None

while True:
    try:
        email_body = input('\n>>> Enter Email (or "quit"/"noise"): ')
        
        if email_body.lower() in ['quit', 'exit']:
            print("Exiting...")
            break
        
        if email_body.lower() == 'noise':
            if current_email is None:
                print("No email loaded. Please enter an email first.")
                continue
            
            # Test robustness with character noise
            print("\n" + "="*85)
            print("CHARACTER NOISE ROBUSTNESS ANALYSIS")
            print("="*85)
            
            for name, model in models_to_compare.items():
                print(f"\n{'-'*40}\nMODEL: {name}\n{'-'*40}")
                predictor = Predictor(model, tokenizer)
                
                results = []
                for noise_level in NOISE_LEVELS:
                    # Apply noise to the raw masked text
                    noisy_text = inject_character_noise(current_email, noise_level)
                    
                    # Tokenize and decode to ensure consistency
                    tokens_fixed = tokenizer(noisy_text, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)
                    processed_text = tokenizer.decode(tokens_fixed['input_ids'])
                    
                    probs = predictor.predict_proba([processed_text])[0]
                    pred_idx = int(np.argmax(probs))
                    pred_label = ['LEGITIMATE', 'PHISHING'][pred_idx]
                    confidence = probs[pred_idx]
                    results.append((noise_level, pred_label, confidence, noisy_text))
                
                # Display results table
                print(f"\n{'Noise %':<10} | {'Prediction':<12} | {'Confidence':<12} | {'Sample Text'}")
                print("-" * 100)
                for noise_level, pred_label, confidence, noisy_text in results:
                    sample = noisy_text[:55] + "..." if len(noisy_text) > 55 else noisy_text
                    print(f"{noise_level*100:<10.0f} | {pred_label:<12} | {confidence:<12.4f} | {sample}")
                
                del predictor
                gc.collect()
            
            continue
        
        if not email_body.strip():
            continue
        
        raw_masked = mask_sensitive_info(email_body).lower()
        current_email = raw_masked  # Store for noise testing
        current_raw_text = email_body
        
        tokens_fixed = tokenizer(raw_masked, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False)
        input_text = tokenizer.decode(tokens_fixed['input_ids'])
        
        for name, model in models_to_compare.items():
            print(f"\n{'='*40}\nMODEL: {name}\n{'='*40}")
            predictor = Predictor(model, tokenizer)
            
            probs = predictor.predict_proba([input_text])[0]
            pred_idx = int(np.argmax(probs))
            print(f"PREDICTION: {['LEGITIMATE', 'PHISHING'][pred_idx]} (Confidence: {probs[pred_idx]:.4f})")
            
            # --- LIME ---
            tracemalloc.start()
            start_t = time.time()
            exp_lime = explainer_lime.explain_instance(input_text, predictor.predict_proba, num_samples=LIME_SAMPLES)
            lime_time = time.time() - start_t
            _, lime_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # --- SHAP ---
            tracemalloc.start()
            start_t = time.time()
            shap_values = shap_explainers[name]([input_text], max_evals=SHAP_MAX_EVALS)
            shap_time = time.time() - start_t
            _, shap_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # --- IG ---
            tracemalloc.start()
            start_t = time.time()
            inputs_ig = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(model.device)
            embeddings = model.get_input_embeddings()(inputs_ig['input_ids'])
            attributions = ig.attribute(embeddings, target=pred_idx, n_steps=IG_STEPS, additional_forward_args=(model,))
            token_attrs = attributions.sum(dim=-1).squeeze(0).cpu().detach().numpy()
            ig_time = time.time() - start_t
            _, ig_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            def is_valid_token(token):
                # FIXED: Corrected string concatenation and escaping to avoid SyntaxError
                chars_to_strip = string.punctuation + '[](){}«»"\' '
                clean_token = token.strip(chars_to_strip)
                return len(clean_token) >= 2 and any(c.isalnum() for c in clean_token)
            
            print("\n[METHOD 1: LIME TOP 10 TOKENS]")
            lime_valid = [(t, s) for t, s in exp_lime.as_list() if is_valid_token(t)]
            for t, s in lime_valid[:10]: print(f"  {t:30s} {s:+0.4f}")
            
            print("\n[METHOD 2: SHAP TOP 10 TOKENS]")
            s_vals, s_tokens = shap_values.values[0][:, pred_idx], shap_values.data[0]
            shap_sorted = np.argsort(np.abs(s_vals))[::-1]
            shap_valid = [i for i in shap_sorted if is_valid_token(s_tokens[i])]
            for i in shap_valid[:10]: print(f"  {s_tokens[i]:30s} {s_vals[i]:+.4f}")
            
            print("\n[METHOD 3: IG TOP 10 TOKENS]")
            all_tokens = tokenizer.convert_ids_to_tokens(inputs_ig['input_ids'][0])
            ig_sorted = np.argsort(np.abs(token_attrs))[::-1]
            ig_valid = [i for i in ig_sorted if all_tokens[i] not in tokenizer.all_special_tokens and is_valid_token(all_tokens[i])]
            for i in ig_valid[:10]: print(f"  {all_tokens[i]:30s} {token_attrs[i]:+.4f}")
            
            print("\n" + "-"*85)
            print(f"{'Method':<12} | {'Time (s)':<10} | {'Mem (KB)':<10}")
            print("-" * 85)
            print(f"{'LIME':<12} | {lime_time:<10.3f} | {lime_peak/1024:<10.2f}")
            print(f"{'SHAP':<12} | {shap_time:<10.3f} | {shap_peak/1024:<10.2f}")
            print(f"{'IG':<12} | {ig_time:<10.3f} | {ig_peak/1024:<10.2f}")
            
            del attributions, embeddings, inputs_ig, token_attrs, all_tokens, exp_lime, shap_values, s_vals, s_tokens, predictor, probs
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        print("\n(Type 'noise' to test this email with character-level noise)")
    
    except Exception as e:
        print(f"\n[ERROR] {e}")
        gc.collect()
        continue

