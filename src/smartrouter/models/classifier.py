from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from smartrouter.core import config
from smartrouter.core.logger import logger


class ModelNotFoundError(FileNotFoundError):
    """Raised when the local fine-tuned checkpoint is missing."""


class RoBERTaClassifier:
    """
    Wrapper around the fine-tuned klue/roberta-base difficulty classifier (submitted V8).
    Loads local weights only: no hub download and no silent fallback to an untrained head.
    """

    def __init__(self, model_path: Optional[Path] = None, device: Optional[str] = None):
        self.path = Path(model_path or config.MODEL_PATH)
        if not (self.path / "config.json").exists():
            raise ModelNotFoundError(
                f"Checkpoint not found at {self.path}. Set ROUTER_MODEL_PATH or pass --model-dir."
            )

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info("Loading classifier from %s on %s", self.path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.path, local_files_only=True, use_safetensors=True
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, formatted_inputs: Sequence[str], batch_size: int = 16) -> list[float]:
        """Return P(label == Hard) for each formatted input."""
        probs: list[float] = []
        for start in range(0, len(formatted_inputs), batch_size):
            batch = list(formatted_inputs[start : start + batch_size])
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=config.MAX_LENGTH,
                padding=True,
            ).to(self.device)
            logits = self.model(**encoded).logits
            probs.extend(F.softmax(logits, dim=-1)[:, config.LABEL_HARD].tolist())
        return probs

    def predict_probability(self, formatted_input: str) -> float:
        return self.predict_proba([formatted_input])[0]
