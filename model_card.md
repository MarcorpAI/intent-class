# DistilBERT Customer Support Intent Classifier

## Model Description

Fine-tuned `distilbert-base-uncased` model for multi-class, single-label customer support intent detection.

## Intended Use

Use this model to route raw customer support messages to the appropriate queue or automation flow. Example intents include order cancellation, refund tracking, payment issues, delivery options, and account support.

## Training Data

The model is designed to train on `bitext/Bitext-customer-support-llm-chatbot-training-dataset` from HuggingFace Datasets. The training pipeline creates train, validation, and test splits with an 80/10/10 ratio.

## Training Procedure

- Base model: `distilbert-base-uncased`
- Max sequence length: 128
- Epochs: 4
- Train batch size: 32
- Eval batch size: 64
- Learning rate: 2e-5
- Weight decay: 0.01
- Warmup ratio: 0.10
- Best checkpoint metric: macro F1

## Evaluation Results

Pending the first full Colab T4 training run.

| Metric | Score |
| --- | --- |
| Accuracy | Pending |
| Macro F1 | Pending |
| Weighted F1 | Pending |

## How to Use

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="your-username/distilbert-customer-support-intent",
)
result = classifier("I want to cancel my order")
print(result)
```

## Limitations

- Best suited to customer support messages similar to the Bitext dataset.
- Predicts one intent per message, even when a message contains multiple requests.
- Production routing should validate confidence thresholds and monitor drift.

