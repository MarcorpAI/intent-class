# Customer Support Intent Classifier

Fine-tuned DistilBERT classifier for routing customer support messages to intent categories, with a reproducible training pipeline and a FastAPI inference service.

The project follows the PRD in `project-2-intent-classifier.md`: train on the Bitext customer support dataset, save/publish the model, and serve predictions through REST endpoints.

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install uv
UV_CACHE_DIR=.uv-cache .venv/bin/uv sync --group dev
```

Install the ML stack only when training or serving a real model:

```bash
UV_CACHE_DIR=.uv-cache .venv/bin/uv sync --group train --group notebook --group serve-ml
```

This workspace uses a separate git directory because the environment mounts `.git` as read-only:

```bash
git --git-dir=.repo/git --work-tree=. status
```

## Development Commands

```bash
UV_CACHE_DIR=.uv-cache .venv/bin/uv run pytest
UV_CACHE_DIR=.uv-cache .venv/bin/uv run python -m training.train --max-samples 64 --no-push
UV_CACHE_DIR=.uv-cache .venv/bin/uv run python -m training.evaluate --model-path artifacts/intent-model
UV_CACHE_DIR=.uv-cache .venv/bin/uv run uvicorn app.main:app --reload
```

## Training

The primary full training path is the Colab notebook at `notebooks/training_walkthrough.ipynb` using a T4 GPU runtime.

Local smoke test:

```bash
UV_CACHE_DIR=.uv-cache .venv/bin/uv run python -m training.train --max-samples 64 --epochs 1 --train-batch-size 8 --eval-batch-size 16 --no-push
```

Full training:

```bash
UV_CACHE_DIR=.uv-cache .venv/bin/uv run python -m training.train
```

Generated artifacts:

- `artifacts/intent-model/`: saved tokenizer and sequence classification model.
- `reports/test_metrics.json`: accuracy, macro F1, and weighted F1.
- `reports/classification_report.json`: per-class precision, recall, and F1.

## Model Configuration

- `INTENT_MODEL_ID`: local artifact path or HuggingFace model id for inference.
- `HF_MODEL_ID`: HuggingFace destination repo used by training when pushing.
- `HF_TOKEN`: HuggingFace token for authenticated push operations.

Publishing example:

```bash
export HF_MODEL_ID="your-username/distilbert-customer-support-intent"
export HF_TOKEN="hf_..."
UV_CACHE_DIR=.uv-cache .venv/bin/uv run python -m training.train --push-to-hub
```

## API Usage

Start the server after training or after setting `INTENT_MODEL_ID` to a compatible HuggingFace model id:

```bash
UV_CACHE_DIR=.uv-cache .venv/bin/uv run uvicorn app.main:app --reload
```

Health:

```bash
curl http://127.0.0.1:8000/health
```

Single prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I want to cancel my order from yesterday"}'
```

Batch prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["Where is my refund?","Do you offer express delivery?"]}'
```

## Evaluation Targets

The PRD targets are:

| Metric | Target | Current |
| --- | ---: | --- |
| Accuracy | >= 92% | Pending full training |
| Macro F1 | >= 0.90 | Pending full training |
| Weighted F1 | >= 0.92 | Pending full training |

## Limitations

- The model is trained for single-label customer support intents and is not a general dialogue classifier.
- Confidence scores depend on the trained model calibration and should be validated before automated routing.
- Full metrics and the HuggingFace model link are pending the first Colab T4 training run.

## Git Notes

This environment mounts `.git` as read-only, so repository metadata is stored in `.repo/git`. Use:

```bash
git --git-dir=.repo/git --work-tree=. status
git --git-dir=.repo/git --work-tree=. log --oneline
```
