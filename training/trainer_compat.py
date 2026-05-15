from __future__ import annotations

import inspect
from typing import Any

from transformers import Trainer


def trainer_tokenizer_kwargs(tokenizer: Any) -> dict[str, Any]:
    """Return Trainer kwargs for tokenizer/processor across transformers versions."""
    signature = inspect.signature(Trainer)
    if "processing_class" in signature.parameters:
        return {"processing_class": tokenizer}
    if "tokenizer" in signature.parameters:
        return {"tokenizer": tokenizer}
    return {}

