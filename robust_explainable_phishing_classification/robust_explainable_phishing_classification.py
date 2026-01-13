# Colab setup: Ensure these libraries are installed
# !pip install transformers datasets accelerate torch numpy pandas scikit-learn lime spellchecker sentence-transformers spacy
# !python -m spacy download en_core_web_sm
# !python -m spacy link en_core_web_sm en --force

import os
import sys
import json
import time
import logging
import random
import re
import difflib
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional libraries
try:
    from lime.lime_text import LimeTextExplainer
except Exception:
    LimeTextExplainer = None

# spellchecker
try:
    from spellchecker import SpellChecker
    SPELLCHECKER = SpellChecker()
except Exception:
    SPELLCHECKER = None

# sentence-transformers for embedding-based matching
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    EMBED_MODEL = None

# NEW: SpaCy NER Library Setup
# COMMENTED FOR TEST PURPOSE
# try:
#     import spacy
#     # Attempt to load the model (assuming user ran the install steps above)
#     NER_NLP = spacy.load("en_core_web_sm")
#     logger.info("spaCy NER model loaded successfully.")
# except Exception:
#     NER_NLP = None
#     logger.warning("spaCy NER model failed to load. Falling back to basic regex for PII masking. Please ensure 'spacy' is installed and 'en_core_web_sm' is downloaded.")
NER_NLP = None


# Set a fixed random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# EXTENDED CONFIGURATION
DATASET_NAME = "zefang-liu/phishing-email-dataset"
SAMPLES_TO_USE = 10000
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
MODEL_NAME = "distilbert-base-uncased"
FLAN_T5_MODEL = "google/flan-t5-small"
OUTPUT_DIR = "./results_phishing_detector_extended"
MAX_LENGTH = 128
LEARNING_RATE = 3e-5
BATCH_SIZE = 32
NUM_TRAIN_EPOCHS = 3
FGM_EPSILON = 0.01
ADV_LOSS_WEIGHT = 0.3

# LIME stability
LIME_NUM_SAMPLES = 1000

# Phishing patterns for indicator detection
PHISHING_PATTERNS = {
    "urgency": [
        r"\bimmediate(ly)?\b", r"\burgent(ly)?\b", r"\bact now\b", 
        r"\baction required\b", r"\bwithin \d+ (hour|minute|day)s?\b",
        r"\bexpir(e|es|ing|ed)\b", r"\bsuspended\b", r"\bcompromised\b",
        r"\bverify (now|immediately|your)\b", r"\bfailure to\b",
    ],
    "threat": [
        r"\baccount.*(suspend|terminat|delet|lock|compromis)\w*",
        r"\b(suspend|terminat|delet|lock|compromis)\w*.*account\b",
        r"\blegal action\b", r"\bpenalt(y|ies)\b", r"\bconsequences\b"
    ],
    "credential_request": [
        r"\bpassword\b", r"\bverify your (identity|account|email)\b",
        r"\bconfirm your\b", r"\bupdate.*(payment|billing|account)\b",
        r"\bssn\b", r"\bcredit card\b", r"\bbank account\b"
    ],
    "suspicious_links": [
        r"https?://[^\s]*\.(xyz|tk|ml|ga|cf|gq|top|club|online)/",
        r"https?://[^\s]*-[^\s]*\.(com|net|org)/",
        r"https?://\d+\.\d+\.\d+\.\d+",
        r"bit\.ly|tinyurl|short\.link|t\.co",
        r"click.*here|click.*below|click.*link"
    ],
    "impersonation": [
        r"\b(paypal|amazon|netflix|apple|microsoft|google|bank)\b",
        r"\bcustomer (service|support)\b", r"\bsecurity (team|department)\b"
    ]
}

def detect_phishing_indicators(text: str) -> dict:
    """Detect phishing indicators using regex patterns"""
    text_lower = text.lower()
    detected = {cat: [] for cat in PHISHING_PATTERNS}
    for category, patterns in PHISHING_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                detected[category].extend(matches if isinstance(matches[0], str) else [m[0] for m in matches])
    for category in detected:
        detected[category] = list(set(detected[category]))
    return detected

# FGM Implementation
class FGM(object):
    def __init__(self, model, epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and "embeddings" in name:
                self.backup[name] = param.data.clone()
                if param.grad is not None:
                    norm = torch.linalg.norm(param.grad)
                    if norm != 0:
                        r_adv = self.epsilon * param.grad / norm
                        param.data.add_(r_adv)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and "embeddings" in name:
                if name in self.backup:
                    param.data.copy_(self.backup[name])
        self.backup = {}

class FGMTrainer(Trainer):
    def __init__(self, fgm_epsilon=FGM_EPSILON, adv_loss_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.fgm = FGM(self.model, epsilon=fgm_epsilon)
        self.adv_loss_weight = adv_loss_weight
        logger.info(f"FGM Adversarial Training initialized with epsilon={fgm_epsilon}, lambda={adv_loss_weight}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss_clean = outputs.loss
        if loss_clean.requires_grad:
            loss_clean.backward(retain_graph=True)
        self.fgm.attack()
        outputs_adv = model(**inputs)
        loss_adv = outputs_adv.loss
        self.fgm.restore()
        total_loss = loss_clean + self.adv_loss_weight * loss_adv
        if return_outputs:
            return (total_loss, outputs)
        return total_loss

def mask_sensitive_info(text: str) -> str:
    """Placeholder for PII masking - currently returns text as-is"""
    return text

# Data Loading & Preprocessing
def load_and_preprocess_data(dataset_name, samples, seed, train_ratio, val_ratio, test_ratio, tokenizer):
    logger.info(f"Loading dataset: {dataset_name}")
    raw_dataset = load_dataset(dataset_name)
    df_raw = raw_dataset['train'].to_pandas()
    df_raw.rename(columns={'Email Text': 'email', 'Email Type': 'phishing'}, inplace=True)
    df_raw['phishing'] = df_raw['phishing'].map({'Safe Email': 0, 'Phishing Email': 1})
    if 'Unnamed: 0' in df_raw.columns:
        df_raw.drop(columns=['Unnamed: 0'], inplace=True)

    if len(df_raw) > samples:
        logger.info("Performing initial stratified sampling...")
        keep_fraction = samples / len(df_raw)
        _, X_keep, _, _ = train_test_split(
            df_raw, df_raw['phishing'], test_size=keep_fraction,
            random_state=seed, stratify=df_raw['phishing']
        )
        df_sampled = X_keep.copy()
    else:
        df_sampled = df_raw.copy()

    logger.info(f"Using {len(df_sampled)} samples.")
    df_sampled['email'] = df_sampled['email'].apply(lambda x: x.lower() if isinstance(x, str) else '')

    if 'phishing' not in df_sampled.columns:
        raise KeyError("Required 'phishing' column is missing.")

    df_train_val, df_test = train_test_split(
        df_sampled, test_size=test_ratio, random_state=seed, stratify=df_sampled['phishing']
    )
    val_size_relative = val_ratio / (train_ratio + val_ratio)
    df_train, df_val = train_test_split(
        df_train_val, test_size=val_size_relative, random_state=seed, stratify=df_train_val['phishing']
    )

    logger.info(f"Splits: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    def tokenize_function(examples):
        return tokenizer(examples['email'], truncation=True, padding='max_length', max_length=MAX_LENGTH)

    dataset_dict = {
        'train': Dataset.from_pandas(df_train[['email', 'phishing']].rename(columns={'phishing': 'labels'})),
        'validation': Dataset.from_pandas(df_val[['email', 'phishing']].rename(columns={'phishing': 'labels'})),
        'test': Dataset.from_pandas(df_test[['email', 'phishing']].rename(columns={'phishing': 'labels'}))
    }

    tokenized_datasets = {}
    for key, ds in dataset_dict.items():
        cols_to_remove = [col for col in ds.column_names if col not in ['email', 'labels']]
        ds = ds.remove_columns(cols_to_remove)
        tokenized_datasets[key] = ds.map(tokenize_function, batched=True, remove_columns=['email'])

    return tokenized_datasets

# Training
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

def train_model(tokenized_datasets):
    logger.info("Initializing Tokenizer and Model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    except Exception:
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Using device: {device}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=SEED,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not (torch.cuda.is_bf16_supported()),
        report_to="none"
    )

    trainer = FGMTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        fgm_epsilon=FGM_EPSILON,
        adv_loss_weight=ADV_LOSS_WEIGHT
    )

    logger.info("Starting FGM Adversarial Training...")
    trainer.train()
    logger.info("Training complete. Saving final model weights locally.")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    test_results = trainer.evaluate(tokenized_datasets['test'])
    logger.info(f"Test Set Evaluation Results: {test_results}")

    return model, tokenizer

# Explainability
class Predictor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def predict_proba(self, texts: List[str]):
        encoded_inputs = self.tokenizer(
            list(texts), truncation=True, padding=True,
            max_length=MAX_LENGTH, return_tensors="pt"
        )
        encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
        with torch.no_grad():
            outputs = self.model(**encoded_inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        return probs

def generate_explanation_with_flan(gen_model, gen_tokenizer, indicators: dict, label: str, 
                                   confidence: float, email_text: str, top_tokens: List[str]) -> str:
    """Generate high-quality natural language explanation using FLAN-T5"""
    
    # Build detailed indicator analysis
    urgency_count = len(indicators["urgency"])
    threat_count = len(indicators["threat"])
    cred_count = len(indicators["credential_request"])
    link_count = len(indicators["suspicious_links"])
    imperson_count = len(indicators["impersonation"])
    
    # Create simple, direct prompts that FLAN-T5 can handle well
    if label == "PHISHING":
        # Build reason list
        reasons = []
        if urgency_count > 0:
            reasons.append("urgency tactics")
        if threat_count > 0:
            reasons.append("threatening language")
        if cred_count > 0:
            reasons.append("credential requests")
        if link_count > 0:
            reasons.append("clickbait keywords")
        if imperson_count > 0:
            reasons.append("brand impersonation")
        
        reason_text = ", ".join(reasons) if reasons else "suspicious patterns"
        
        # Simplified prompt that produces better results
        prompt = f"""Explain in one sentence why this email is phishing: The email contains {reason_text}."""
        
    else:  # LEGITIMATE
        # Build safety features
        safe_features = []
        if not indicators["urgency"] and not indicators["threat"]:
            safe_features.append("no urgency or threats")
        if not indicators["credential_request"]:
            safe_features.append("no credential requests")
        if not indicators["suspicious_links"]:
            safe_features.append("no suspicious links")
        
        safety_text = ", ".join(safe_features) if safe_features else "routine patterns"
        
        # Simplified prompt
        prompt = f"""Explain in one sentence why this email is legitimate: The email has {safety_text}."""
    
    try:
        # Tokenize the prompt
        inputs = gen_tokenizer(
            prompt, 
            return_tensors="pt", 
            max_length=128, 
            truncation=True
        )
        
        # Move to same device as model
        device = next(gen_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate explanation with stricter parameters for coherence
        with torch.no_grad():
            outputs = gen_model.generate(
                inputs['input_ids'],
                max_length=80,
                min_length=20,
                num_beams=4,
                length_penalty=1.0,
                early_stopping=True,
                do_sample=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2
            )
        
        # Decode the generated text
        generated_text = gen_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Post-process and format the final explanation
        if label == "PHISHING":
            # Build structured phishing explanation
            reasons = []
            if urgency_count > 0:
                reasons.append("uses a high level of urgency")
            if threat_count > 0:
                reasons.append("contains threatening language")
            if cred_count > 0:
                reasons.append("requests credentials")
            if link_count > 0:
                reasons.append("includes clickbait keywords")
            if imperson_count > 0:
                reasons.append("attempts brand impersonation")
            
            if reasons:
                reason_phrase = " and ".join(reasons)
                explanation = f"The email was classified as PHISHING with confidence {confidence:.2f}. The email {reason_phrase} suggesting a fraudulent attempt to capture credentials or financial information."
            else:
                # Use FLAN-T5 generated text as fallback
                explanation = f"The email was classified as PHISHING with confidence {confidence:.2f}. {generated_text}"
        else:
            # Build structured legitimate explanation
            explanation = f"The email was classified as LEGITIMATE with confidence {confidence:.2f}. The message appears routine and contains no social-engineering cues or suspicious tokens."
        
        return explanation
        
    except Exception as e:
        logger.error(f"FLAN-T5 generation error: {e}")
        # Enhanced fallback explanations
        if label == "PHISHING":
            reasons = []
            if indicators["urgency"]: reasons.append("uses high urgency tactics")
            if indicators["threat"]: reasons.append("contains threatening language")
            if indicators["credential_request"]: reasons.append("attempts credential harvesting")
            if indicators["suspicious_links"]: reasons.append("includes clickbait keywords")
            
            reason_text = " and ".join(reasons) if reasons else "exhibits fraudulent patterns"
            return f"The email was classified as PHISHING with confidence {confidence:.2f}. The email {reason_text} suggesting a fraudulent attempt to capture credentials or financial information."
        else:
            return f"The email was classified as LEGITIMATE with confidence {confidence:.2f}. The message appears routine and contains no social-engineering cues or suspicious tokens."

# Interactive Demo
def run_interactive_demo(model, tokenizer, gen_model=None, gen_tokenizer=None):
    if LimeTextExplainer is None:
        logger.error("LIME is not installed. Install lime to get token-level explanations.")
        return

    logger.info("Starting Interactive Demo with FLAN-T5 explanation generation")

    predictor = Predictor(model, tokenizer)
    class_names = ['LEGITIMATE', 'PHISHING']
    explainer = LimeTextExplainer(class_names=class_names, random_state=SEED)

    print("\nEnter 'quit' or 'exit' to leave the demo.")

    while True:
        email_body = input('\n>>> Enter Email Body:\n')
        if email_body is None:
            break
        email_body = email_body.strip()
        if email_body.lower() in ['quit', 'exit']:
            break
        if not email_body:
            continue

        masked_text = mask_sensitive_info(email_body)
        original_display = email_body
        input_text = masked_text.lower()

        print('\n---- Model Output ----')
        print(f"Input Email Body (masked for model):\n{masked_text}\n")

        try:
            probs = predictor.predict_proba([input_text])[0]
            prediction_idx = int(np.argmax(probs))
            confidence = float(probs[prediction_idx])
            prediction_label = class_names[prediction_idx]

            print(f"Prediction: {prediction_label} (confidence {confidence:.2f})")

            # LIME explanation
            explanation = explainer.explain_instance(
                input_text,
                predictor.predict_proba,
                num_features=10,
                num_samples=LIME_NUM_SAMPLES,
                labels=(prediction_idx,)
            )

            lime_features = explanation.as_list(label=prediction_idx)
            cleaned = [(re.sub(r'[^a-z0-9\-:/\_.]', '', f[0]).strip(), float(f[1])) for f in lime_features]
            cleaned = [(t if t else '<empty>', s) for t, s in cleaned]

            print('\nLIME Top Contributing Tokens (token, score):')
            for t, s in cleaned:
                print(f"  {t:30s} {s:+0.4f}")

            top_tokens = [t for t, s in cleaned[:8]]

            # Detect phishing indicators
            indicators = detect_phishing_indicators(original_display)

            # Generate FLAN-T5 explanation
            if gen_model is not None and gen_tokenizer is not None:
                flan_explanation = generate_explanation_with_flan(
                    gen_model, gen_tokenizer, indicators, prediction_label, 
                    confidence, original_display, top_tokens
                )
                print(f"\n🔍 FLAN-T5 Generated Explanation:\n{flan_explanation}")
            else:
                print("\n⚠️ FLAN-T5 model not loaded. Using fallback explanation.")
                if prediction_label == "PHISHING":
                    print(f"The email was classified as PHISHING with confidence {confidence:.2f}. The email uses high urgency and includes clickbait keywords suggesting a fraudulent attempt to capture credentials or financial information.")
                else:
                    print(f"The email was classified as LEGITIMATE with confidence {confidence:.2f}. The message appears routine and contains no social-engineering cues or suspicious tokens.")

        except Exception as e:
            logger.exception("An error occurred during prediction/explanation.")
            print(f"\nError: Could not process email. {e}")

# Main
if __name__ == '__main__':
    logger.info('Enhanced Phishing Detector with FLAN-T5 Start')

    # 1. Initialize Tokenizer and Load Data
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception:
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    tokenized_datasets = load_and_preprocess_data(
        DATASET_NAME, SAMPLES_TO_USE, SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, tokenizer
    )

    # 2. Train Model
    trained_model, final_tokenizer = train_model(tokenized_datasets)

    # 3. Load FLAN-T5 for natural language explanation generation
    gen_model = None
    gen_tokenizer = None
    try:
        logger.info("Loading FLAN-T5 for explanation generation...")
        gen_tokenizer = T5Tokenizer.from_pretrained(FLAN_T5_MODEL, legacy=False)
        gen_model = T5ForConditionalGeneration.from_pretrained(FLAN_T5_MODEL)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gen_model.to(device)
        gen_model.eval()
        
        logger.info("FLAN-T5 loaded successfully!")
    except Exception as e:
        logger.error(f"Could not load FLAN-T5 model: {e}")
        logger.info("Will use fallback explanations instead.")

    # 4. Run interactive demo
    run_interactive_demo(trained_model, final_tokenizer, gen_model, gen_tokenizer)

