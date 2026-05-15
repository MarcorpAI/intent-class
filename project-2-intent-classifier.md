# Project 2: Fine-Tuned Text Classifier — Customer Support Intent Detection

## Overview

A domain-specific text classification system fine-tuned on a customer support dataset to detect user intent from raw message text. The model is trained using the HuggingFace `Trainer` API, pushed to HF Hub with a proper model card, and served via a FastAPI inference endpoint. Covers the full ML lifecycle: data → training → evaluation → deployment.

---

## Goals

- Demonstrate the ability to *train* a transformer model, not just call one
- Produce a reproducible training pipeline with logged metrics
- Publish a real, usable model to HuggingFace Hub
- Serve predictions via a clean REST API

---

## Problem Statement

Given a raw customer message (e.g. *"I haven't received my order and it's been 2 weeks"*), classify it into one of **N intent categories** so that it can be routed to the right support queue or handled by an automated agent.

This is a **multi-class single-label classification** problem.

---

## Dataset

**Source:** [`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) on HuggingFace Datasets.

- ~27,000 examples
- 27 intent categories (e.g. `cancel_order`, `track_refund`, `change_shipping_address`, `complaint`, etc.)
- Clean, labeled, ready to use — no preprocessing required beyond tokenization

**Why this dataset:** It's realistic, multi-class, and directly tied to a production NLP use case. It will resonate with any hiring team that has built or maintained a support automation system.

---

## Intent Classes (subset)

```
cancel_order          | change_order         | check_refund_policy
complaint             | contact_customer_service | create_account
delete_account        | delivery_options     | get_invoice
payment_issue         | place_order          | track_order
track_refund          | change_shipping_address | registration_problems
```

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Training Pipeline                  │
│                                                         │
│  HF Dataset → Tokenizer → DistilBERT → Trainer API      │
│                                    │                    │
│                             HF Hub (model push)         │
└────────────────────────────────────┬────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────┐
│                    Inference Service                    │
│                                                         │
│  FastAPI → load model from HF Hub → predict intent      │
│         → return label + confidence scores              │
└─────────────────────────────────────────────────────────┘
```

---

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Base Model | `distilbert-base-uncased` (HuggingFace) |
| Training Framework | HuggingFace `transformers` + `Trainer` API |
| Dataset Library | HuggingFace `datasets` |
| Evaluation | `evaluate` library (accuracy, F1, classification report) |
| Experiment Tracking | HF Hub training logs + optional `wandb` |
| Serving | FastAPI + Uvicorn |
| Model Registry | HuggingFace Hub (public model repo) |
| Notebook | Jupyter / Google Colab |

---

## Model Choice — DistilBERT

`distilbert-base-uncased` is chosen over BERT-base for the following reasons:

- 40% smaller, 60% faster than BERT — trains quickly even without high-end GPU
- Retains 97% of BERT's performance on GLUE benchmarks
- Standard and well-understood — shows you can make principled model selection decisions
- Can be fine-tuned on Google Colab T4 in under 30 minutes

---

## Training Pipeline — Detailed Flow

### Step 1: Data Loading & Splitting
```
Load dataset from HF Hub
→ Train split: 80%
→ Validation split: 10%
→ Test split: 10%
→ Encode intent labels as integer IDs
→ Build id2label / label2id mappings
```

### Step 2: Tokenization
```
Apply DistilBERT tokenizer
→ max_length: 128 tokens (sufficient for short support messages)
→ padding: "max_length"
→ truncation: True
→ return_tensors: "pt"
```

### Step 3: Model Initialization
```
AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=N,
    id2label=id2label,
    label2id=label2id
)
```

### Step 4: Training Configuration (TrainingArguments)

| Hyperparameter | Value | Rationale |
|---|---|---|
| `num_train_epochs` | 4 | Sufficient for convergence on this dataset size |
| `per_device_train_batch_size` | 32 | Fits in Colab T4 (16GB) |
| `per_device_eval_batch_size` | 64 | |
| `learning_rate` | 2e-5 | Standard for transformer fine-tuning |
| `weight_decay` | 0.01 | Regularization |
| `warmup_ratio` | 0.1 | 10% of steps for LR warmup |
| `evaluation_strategy` | `"epoch"` | Evaluate after every epoch |
| `save_strategy` | `"epoch"` | Save best checkpoint |
| `load_best_model_at_end` | `True` | Use best val checkpoint |
| `metric_for_best_model` | `"f1"` | Optimize for macro F1 |

### Step 5: Evaluation
```
compute_metrics(eval_pred):
  → accuracy
  → macro F1
  → weighted F1
  → per-class precision, recall, F1 (classification_report)
```

### Step 6: Push to Hub
```
trainer.push_to_hub()
→ Uploads model weights, config, tokenizer
→ Auto-generates training metadata
```

---

## Evaluation Targets

| Metric | Target |
|---|---|
| Accuracy | ≥ 92% |
| Macro F1 | ≥ 0.90 |
| Weighted F1 | ≥ 0.92 |

These are realistic for DistilBERT on a clean 27-class dataset with sufficient training data per class.

---

## API Endpoints

### `POST /predict`
Classify a single customer message.

**Request:**
```json
{
  "text": "I want to cancel my order from yesterday"
}
```

**Response:**
```json
{
  "intent": "cancel_order",
  "confidence": 0.97,
  "all_scores": {
    "cancel_order": 0.97,
    "change_order": 0.02,
    "track_order": 0.01,
    "...": "..."
  }
}
```

---

### `POST /predict/batch`
Classify multiple messages in one call.

**Request:**
```json
{
  "texts": [
    "Where is my refund?",
    "I can't log into my account",
    "Do you offer express delivery?"
  ]
}
```

**Response:**
```json
{
  "predictions": [
    { "text": "Where is my refund?", "intent": "track_refund", "confidence": 0.94 },
    { "text": "I can't log into my account", "intent": "registration_problems", "confidence": 0.91 },
    { "text": "Do you offer express delivery?", "intent": "delivery_options", "confidence": 0.96 }
  ]
}
```

---

### `GET /health`
Returns model status and loaded label list.

---

## Project Structure

```
intent-classifier/
├── training/
│   ├── train.py              # Full training script
│   ├── config.py             # Hyperparameters and constants
│   └── evaluate.py           # Post-training evaluation script
├── notebooks/
│   └── training_walkthrough.ipynb   # Colab-ready notebook
├── app/
│   ├── main.py               # FastAPI app
│   ├── predictor.py          # Model loading + inference logic
│   └── schemas.py            # Pydantic request/response models
├── tests/
│   └── test_predictor.py     # Unit tests for inference
├── requirements.txt
├── requirements-train.txt    # Separate deps for training
├── model_card.md             # HF Hub model card (mirrors HF format)
└── README.md
```

---

## HuggingFace Hub Deliverables

- [ ] Model pushed to `{your-username}/distilbert-customer-support-intent`
- [ ] Full model card with:
  - Intended use
  - Training data description
  - Training hyperparameters
  - Evaluation results table
  - Inference code example
  - Limitations
- [ ] Training notebook pushed as a HF dataset or linked in model card

---

## Model Card Template (Key Sections)

```markdown
## Model Description
Fine-tuned DistilBERT for 27-class customer support intent detection.

## Intended Use
Route raw customer messages to the correct support queue or automation flow.

## Training Data
Bitext Customer Support Dataset (HuggingFace Hub), 27,000 examples, 27 classes.

## Evaluation Results
| Metric     | Score |
|------------|-------|
| Accuracy   | 93.4% |
| Macro F1   | 0.912 |
| Weighted F1| 0.931 |

## How to Use
from transformers import pipeline
classifier = pipeline("text-classification", model="{username}/distilbert-customer-support-intent")
result = classifier("I want to cancel my order")
```

---

## README Sections

1. Problem framing — why intent detection matters in NLP pipelines
2. Model architecture and training approach
3. Dataset description
4. Training results with metrics table
5. API usage with cURL + Python examples
6. Limitations and known failure modes
7. Link to HF Hub model

---

## Estimated Build Time

| Phase | Time |
|---|---|
| Data loading + tokenization | 30min |
| Training script + Trainer config | 45min |
| Training run (Colab T4) | 25min |
| Evaluation + metric logging | 20min |
| Push to HF Hub + model card | 30min |
| FastAPI serving endpoint | 45min |
| README + notebook cleanup | 30min |
| **Total** | **~3.5 hours** |
