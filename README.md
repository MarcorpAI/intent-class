# Customer Support Intent Classifier

Fine-tuned DistilBERT classifier for routing customer support messages to intent categories, with a reproducible training pipeline and a FastAPI inference service.

## Setup

```bash
uv sync --all-groups
```

This workspace uses a separate git directory because the environment mounts `.git` as read-only:

```bash
git --git-dir=.repo/git --work-tree=. status
```

## Development Commands

```bash
uv run pytest
uv run python -m training.train --max-samples 64 --no-push
uv run python -m training.evaluate --model-path artifacts/intent-model
uv run uvicorn app.main:app --reload
```

## Model Configuration

- `INTENT_MODEL_ID`: local artifact path or HuggingFace model id for inference.
- `HF_MODEL_ID`: HuggingFace destination repo used by training when pushing.
- `HF_TOKEN`: HuggingFace token for authenticated push operations.

See `project-2-intent-classifier.md` for the product requirements.

