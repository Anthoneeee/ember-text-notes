from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Iterable, List

import joblib
from torch import nn


def _load_utils_module():
    """Load utility module by absolute path for robust runtime imports."""
    utils_path = Path(__file__).resolve().parent / "news_b_utils.py"
    if not utils_path.exists():
        raise FileNotFoundError(f"Utility module not found: {utils_path}")
    spec = importlib.util.spec_from_file_location("news_b_utils_local", utils_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load utility module: {utils_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["news_b_utils_local"] = module
    spec.loader.exec_module(module)
    return module


_utils = _load_utils_module()
normalize_text = _utils.normalize_text
url_to_pseudo_headline = _utils.url_to_pseudo_headline


class Model(nn.Module):
    """
    Project B inference wrapper.
    Notes:
    - Loads a trained sklearn Pipeline via joblib
    - Keeps compatibility with the evaluator's `predict(batch)` interface
    """

    def __init__(self, weights_path: str | None = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pipeline = self._load_pipeline(weights_path)

    @staticmethod
    def _default_model_path() -> Path:
        return Path(__file__).resolve().parent / "Newsheadlines" / "artifacts" / "news_b_tfidf_lr.joblib"

    def _resolve_model_path(self, weights_path: str | None) -> Path:
        # Evaluator may pass "__no_weights__.pth"; ignore that placeholder.
        if weights_path and weights_path != "__no_weights__.pth":
            candidate = Path(weights_path)
            if candidate.suffix.lower() in {".joblib", ".pkl"} and candidate.exists():
                return candidate
        return self._default_model_path()

    def _load_pipeline(self, weights_path: str | None):
        model_path = self._resolve_model_path(weights_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Run train_news_b_v1.py first."
            )
        payload = joblib.load(model_path)
        if isinstance(payload, dict) and "pipeline" in payload:
            return payload["pipeline"]
        return payload

    @staticmethod
    def _coerce_item_to_text(item: Any) -> str:
        """
        Normalize one batch item to text.
        - URL input -> pseudo-headline text
        - dict input -> prefer headline/title/text/url fields
        - otherwise -> string conversion
        """
        if isinstance(item, dict):
            for key in ["headline", "title", "text", "content"]:
                if key in item and item[key]:
                    return normalize_text(str(item[key]))
            if "url" in item and item["url"]:
                return normalize_text(url_to_pseudo_headline(str(item["url"])))
            return normalize_text(str(item))

        text = str(item)
        if text.startswith("http://") or text.startswith("https://"):
            return normalize_text(url_to_pseudo_headline(text))
        return normalize_text(text)

    def eval(self):
        # Keep nn.Module semantics; return self for chain compatibility.
        return super().eval()

    def predict(self, batch: Iterable[Any]) -> List[int]:
        texts = [self._coerce_item_to_text(x) for x in batch]
        preds = self.pipeline.predict(texts)
        return [int(p) for p in preds]


def get_model() -> Model:
    """Factory function expected by the evaluator."""
    return Model()
