from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from news_b_utils import prepare_dataset_from_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B 问第一版训练脚本（TF-IDF + LR）")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="Newsheadlines/url_only_data.csv",
        help="输入 CSV 路径（支持 url 或 headline/title 列）",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="Newsheadlines/artifacts/news_b_tfidf_lr.joblib",
        help="导出的模型文件路径",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    parser.add_argument("--max-features", type=int, default=30000, help="TF-IDF 最大特征数")
    parser.add_argument("--ngram-max", type=int, default=2, help="TF-IDF ngram 上限")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 第一步：读取并构建文本样本与标签
    prepared = prepare_dataset_from_csv(args.input_csv)
    X = prepared.texts
    y = np.asarray(prepared.labels, dtype=np.int64)

    # 第二步：分层划分训练/验证，保证类别比例一致
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # 第三步：第一版模型，先用稳定可解释的 TF-IDF + 逻辑回归
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, args.ngram_max),
                    min_df=2,
                    max_df=0.98,
                    max_features=args.max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="liblinear",
                    C=2.0,
                    class_weight="balanced",
                    random_state=args.random_state,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)

    # 第四步：在验证集上输出核心指标，作为第一版基线
    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    report = classification_report(y_val, y_pred, digits=4)

    print(f"num_samples: {len(X)}")
    print(f"train_samples: {len(X_train)}")
    print(f"val_samples: {len(X_val)}")
    print(f"val_accuracy: {acc:.6f}")
    print(f"val_macro_f1: {macro_f1:.6f}")
    print("classification_report:")
    print(report)

    # 第五步：导出模型与基础元信息，供 model.py 推理加载
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline": pipeline,
        "meta": {
            "model": "tfidf+logistic_regression",
            "label_map": {"0": "FoxNews", "1": "NBC"},
            "num_samples": len(X),
            "random_state": args.random_state,
        },
    }
    joblib.dump(payload, output_path)
    print(f"saved_model: {output_path}")
    print("saved_meta_json:")
    print(json.dumps(payload["meta"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
