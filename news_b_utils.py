from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import pandas as pd


_LABEL_COL_CANDIDATES = ["label", "labels", "source", "outlet", "news_source", "publisher"]
_TEXT_COL_CANDIDATES = ["headline", "title", "text", "content", "news_title"]
_URL_COL_CANDIDATES = ["url", "link", "article_url"]


@dataclass
class PreparedData:
    """保存预处理后的文本与标签。"""

    texts: List[str]
    labels: List[int]


def _find_col_case_insensitive(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    """在列名中做不区分大小写匹配，找到第一个候选列名。"""
    lookup = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    return None


def canonicalize_label(raw_label: object) -> Optional[int]:
    """把多种标签表示统一成二分类整数：Fox=0, NBC=1。"""
    if raw_label is None:
        return None
    text = str(raw_label).strip().lower()
    if text == "":
        return None

    if text in {"0", "fox", "foxnews", "fox news"}:
        return 0
    if text in {"1", "nbc", "nbcnews", "nbc news"}:
        return 1
    return None


def infer_label_from_url(url: str) -> Optional[int]:
    """如果没有显式标签，则从 URL 域名推断标签。"""
    host = urlparse(str(url)).netloc.lower()
    if "foxnews.com" in host:
        return 0
    if "nbcnews.com" in host:
        return 1
    return None


def url_to_pseudo_headline(url: str) -> str:
    """
    当没有 headline 文本时，把 URL 路径转成“伪标题”。
    说明：第一版先保证流程可跑，后续可以替换成真实抓取标题。
    """
    parsed = urlparse(str(url))
    path = unquote(parsed.path or "")
    path = path.replace(".print", " ")
    path = path.replace(".html", " ")
    path = path.replace("/", " ")
    path = path.replace("-", " ")
    path = path.replace("_", " ")
    path = html.unescape(path)
    path = re.sub(r"\s+", " ", path).strip()
    return path


def normalize_text(text: str) -> str:
    """做最小标准化，减少噪声字符。"""
    text = html.unescape(str(text))
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_dataset_from_csv(csv_path: str) -> PreparedData:
    """
    从 CSV 构建训练/评估输入。
    规则：
    1) 文本优先使用 headline/title 等列。
    2) 没有文本列时，用 URL 路径生成伪标题。
    3) 标签优先使用 label/source 等列；没有则从 URL 推断。
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")

    df = pd.read_csv(csv_file)
    if df.empty:
        raise ValueError(f"CSV 为空: {csv_path}")

    text_col = _find_col_case_insensitive(df.columns, _TEXT_COL_CANDIDATES)
    label_col = _find_col_case_insensitive(df.columns, _LABEL_COL_CANDIDATES)
    url_col = _find_col_case_insensitive(df.columns, _URL_COL_CANDIDATES)

    texts: List[str] = []
    labels: List[int] = []

    for _, row in df.iterrows():
        # 先确定文本
        text_val: Optional[str] = None
        if text_col is not None and pd.notna(row[text_col]):
            text_val = normalize_text(str(row[text_col]))

        url_val = ""
        if url_col is not None and pd.notna(row[url_col]):
            url_val = str(row[url_col]).strip()
        elif "url" in df.columns and pd.notna(row["url"]):
            url_val = str(row["url"]).strip()

        if not text_val:
            if not url_val:
                continue
            text_val = normalize_text(url_to_pseudo_headline(url_val))

        # 再确定标签
        label_val: Optional[int] = None
        if label_col is not None and pd.notna(row[label_col]):
            label_val = canonicalize_label(row[label_col])

        if label_val is None and url_val:
            label_val = infer_label_from_url(url_val)

        # 第一版里若无法确定标签就跳过，确保 X/y 严格对齐
        if label_val is None:
            continue

        if text_val == "":
            continue

        texts.append(text_val)
        labels.append(label_val)

    if not texts:
        raise ValueError(
            "未能从 CSV 构建任何样本。请检查列名是否包含 headline/title/url 与 label/source。"
        )

    return PreparedData(texts=texts, labels=labels)
