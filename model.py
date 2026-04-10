from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Iterable, List

import joblib
from torch import nn


def _load_utils_module():
    """按绝对路径加载工具模块，避免动态评测时找不到本地依赖。"""
    utils_path = Path(__file__).resolve().parent / "news_b_utils.py"
    if not utils_path.exists():
        raise FileNotFoundError(f"找不到工具模块: {utils_path}")
    spec = importlib.util.spec_from_file_location("news_b_utils_local", utils_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载工具模块: {utils_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["news_b_utils_local"] = module
    spec.loader.exec_module(module)
    return module


_utils = _load_utils_module()
normalize_text = _utils.normalize_text
url_to_pseudo_headline = _utils.url_to_pseudo_headline


class Model(nn.Module):
    """
    B 问第一版推理模型。
    说明：
    - 通过 joblib 加载已训练的 sklearn Pipeline
    - 保持与课程评测脚本约定的 `predict(batch)` 接口兼容
    """

    def __init__(self, weights_path: str | None = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pipeline = self._load_pipeline(weights_path)

    @staticmethod
    def _default_model_path() -> Path:
        return Path(__file__).resolve().parent / "Newsheadlines" / "artifacts" / "news_b_tfidf_lr.joblib"

    def _resolve_model_path(self, weights_path: str | None) -> Path:
        # 评测脚本会传入 "__no_weights__.pth"，这里需要忽略该占位参数
        if weights_path and weights_path != "__no_weights__.pth":
            candidate = Path(weights_path)
            if candidate.suffix.lower() in {".joblib", ".pkl"} and candidate.exists():
                return candidate
        return self._default_model_path()

    def _load_pipeline(self, weights_path: str | None):
        model_path = self._resolve_model_path(weights_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"未找到模型文件: {model_path}。请先运行 train_news_b_v1.py 训练并导出模型。"
            )
        payload = joblib.load(model_path)
        if isinstance(payload, dict) and "pipeline" in payload:
            return payload["pipeline"]
        return payload

    @staticmethod
    def _coerce_item_to_text(item: Any) -> str:
        """
        把 batch 中的单条样本统一转成文本。
        - 若输入是 URL，则转成伪标题文本
        - 若输入是字典，优先读 headline/title/text/url
        - 其他类型统一转字符串
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
        # 这里保持 nn.Module 语义，返回 self 以兼容调用链
        return super().eval()

    def predict(self, batch: Iterable[Any]) -> List[int]:
        texts = [self._coerce_item_to_text(x) for x in batch]
        preds = self.pipeline.predict(texts)
        return [int(p) for p in preds]


def get_model() -> Model:
    """工厂函数：与评测器约定一致。"""
    return Model()
