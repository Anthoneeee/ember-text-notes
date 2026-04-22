from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List, Tuple


def _load_utils_module():
    """Load utility module by absolute path to avoid dynamic import issues."""
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


prepare_dataset_from_csv = _load_utils_module().prepare_dataset_from_csv


def prepare_data(path: str) -> Tuple[List[str], List[int]]:
    """
    Project B preprocessing entrypoint.
    Returns:
    - X: list of text inputs
    - y: list of binary labels (Fox=0, NBC=1)
    """
    prepared = prepare_dataset_from_csv(path)
    return prepared.texts, prepared.labels
