from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from news_b_utils import normalize_text, url_to_pseudo_headline


@dataclass
class EvalResult:
    section: str
    scenario: str
    train_representation: str
    test_representation: str
    train_rows: int
    test_rows: int
    accuracy: float
    macro_f1: float
    avg_infer_ms: float
    notes: str


def build_pipeline(random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=30000,
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
                    random_state=random_state,
                ),
            ),
        ]
    )


def clean_dataframe(df: pd.DataFrame, strict: bool) -> pd.DataFrame:
    out = df.copy()
    out["headline"] = out["headline"].fillna("").astype(str).map(normalize_text)
    out["url"] = out["url"].fillna("").astype(str).str.strip()
    out["label"] = out["label"].astype(int)

    out = out[out["headline"].str.len() > 0].copy()
    if strict:
        out = out[out["headline"].str.len() >= 8].copy()
        out = out[out["headline"].map(lambda x: any(ch.isalnum() for ch in x))].copy()
        out["url_key"] = out["url"].str.lower()
        out["headline_key"] = out["headline"].str.lower()
        out = out.drop_duplicates(subset=["url_key"], keep="first").copy()
        out = out.drop_duplicates(subset=["headline_key"], keep="first").copy()
        out = out.drop(columns=["url_key", "headline_key"])
    return out.reset_index(drop=True)


def add_label_conflict_stress(df_train: pd.DataFrame, frac: float, random_state: int) -> pd.DataFrame:
    flip = df_train.sample(frac=frac, replace=True, random_state=random_state).copy()
    flip["label"] = 1 - flip["label"].astype(int)
    flip["url"] = flip["url"].astype(str) + "#label_conflict_" + flip.index.astype(str)
    return pd.concat([df_train, flip], ignore_index=True)


def text_representation(df: pd.DataFrame, mode: str) -> pd.Series:
    headline = df["headline"].map(normalize_text)
    url_pseudo = df["url"].map(url_to_pseudo_headline).map(normalize_text)
    if mode == "headline_only":
        return headline
    if mode == "url_only":
        return url_pseudo
    if mode == "headline_plus_url":
        return headline + " [SEP] " + url_pseudo
    raise ValueError(f"Unsupported text representation: {mode}")


def run_one_eval(
    section: str,
    scenario: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_repr: str,
    test_repr: str,
    random_state: int,
    notes: str,
) -> EvalResult:
    x_train = text_representation(train_df, train_repr)
    y_train = train_df["label"].astype(int).to_numpy()
    x_test = text_representation(test_df, test_repr)
    y_test = test_df["label"].astype(int).to_numpy()

    model = build_pipeline(random_state=random_state)
    model.fit(x_train, y_train)
    tic = time.perf_counter()
    preds = model.predict(x_test)
    infer_s = time.perf_counter() - tic

    return EvalResult(
        section=section,
        scenario=scenario,
        train_representation=train_repr,
        test_representation=test_repr,
        train_rows=len(train_df),
        test_rows=len(test_df),
        accuracy=float(accuracy_score(y_test, preds)),
        macro_f1=float(f1_score(y_test, preds, average="macro")),
        avg_infer_ms=float((infer_s / max(1, len(test_df))) * 1000.0),
        notes=notes,
    )


def make_plots(results_df: pd.DataFrame, figure_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    clean_df = results_df[results_df["section"] == "cleaning_effect"]
    clean_order = [
        "real_data_loose_clean",
        "real_data_strict_clean",
        "stress_data_loose_clean",
        "stress_data_strict_clean",
    ]
    clean_df = clean_df.set_index("scenario").reindex(clean_order).reset_index()
    axes[0].bar(clean_df["scenario"], clean_df["accuracy"], color=["#d95f02", "#1b9e77", "#d95f02", "#1b9e77"])
    axes[0].set_title("Cleaning Strategy vs Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.45, 1.0)
    axes[0].tick_params(axis="x", rotation=30, labelsize=8)

    shortcut_df = results_df[results_df["section"] == "shortcut_risk"]
    short_order = [
        "headline_to_headline",
        "url_to_headline",
        "url_to_url",
        "headline_plus_url_to_headline_plus_url",
        "headline_plus_url_to_headline_only",
    ]
    shortcut_df = shortcut_df.set_index("scenario").reindex(short_order).reset_index()
    axes[1].bar(shortcut_df["scenario"], shortcut_df["accuracy"], color="#4c78a8")
    axes[1].set_title("Shortcut Risk Matrix (Accuracy)")
    axes[1].set_ylim(0.45, 1.0)
    axes[1].tick_params(axis="x", rotation=30, labelsize=8)

    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def build_summary_md(results_df: pd.DataFrame, metadata: Dict[str, object]) -> str:
    def _row(s: str) -> pd.Series:
        return results_df.loc[results_df["scenario"] == s].iloc[0]

    headline_to_headline = _row("headline_to_headline")
    url_to_headline = _row("url_to_headline")
    url_to_url = _row("url_to_url")
    hu_to_hu = _row("headline_plus_url_to_headline_plus_url")
    hu_to_h = _row("headline_plus_url_to_headline_only")

    real_loose = _row("real_data_loose_clean")
    real_strict = _row("real_data_strict_clean")
    stress_loose = _row("stress_data_loose_clean")
    stress_strict = _row("stress_data_strict_clean")

    md = []
    md.append("# Exploratory Component (Step 7)")
    md.append("")
    md.append("## Topic")
    md.append("Data Cleaning and Shortcut Risk: Effects on Headline Classification Generalization")
    md.append("")
    md.append("## Setup")
    md.append(f"- Run date (local): {metadata['run_date_local']}")
    md.append(f"- Input dataset: `{metadata['input_csv']}`")
    md.append(f"- Split: stratified holdout, test_size={metadata['test_size']}, random_state={metadata['random_state']}")
    md.append("")
    md.append("## Key Findings")
    md.append(
        f"- Shortcut risk is clear: URL-only training reaches {url_to_url['accuracy']:.4f} on URL-style test "
        f"but only {url_to_headline['accuracy']:.4f} on real headlines."
    )
    md.append(
        f"- Headline-first training is more reliable for the actual task: "
        f"{headline_to_headline['accuracy']:.4f} on headline test."
    )
    md.append(
        f"- A model trained on headline+URL can become URL-dependent: "
        f"{hu_to_hu['accuracy']:.4f} with full input vs {hu_to_h['accuracy']:.4f} when URL features are absent."
    )
    md.append(
        f"- On this crawl, real-data cleaning deltas are small "
        f"({real_loose['accuracy']:.4f} -> {real_strict['accuracy']:.4f}) because the dataset is already clean."
    )
    md.append(
        f"- In stress test with injected conflicting duplicates, strict cleaning protects generalization "
        f"({stress_loose['accuracy']:.4f} -> {stress_strict['accuracy']:.4f})."
    )
    md.append("")
    md.append("## Recommendation")
    md.append("- Keep headline-only as the final training target for Project B.")
    md.append("- Keep strict cleaning enabled (dedup URLs/headlines + invalid headline filters).")
    md.append("- Use URL-based features only as diagnostic experiments, not as primary training signal.")
    md.append("")
    md.append("## Output Files")
    md.append("- `deliverables/report/exploratory_results_step7.csv`")
    md.append("- `deliverables/report/exploratory_summary_step7.md`")
    md.append("- `deliverables/manifests/exploratory_metadata_step7.json`")
    md.append("- `deliverables/figures/exploratory_cleaning_shortcut_step7.png`")
    md.append("")
    return "\n".join(md)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run exploratory component for Step 7.")
    p.add_argument(
        "--input-csv",
        type=str,
        default="Newsheadlines/scraped_headlines_clean.csv",
        help="Input cleaned headline CSV.",
    )
    p.add_argument("--test-size", type=float, default=0.2, help="Holdout split ratio.")
    p.add_argument("--random-state", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--stress-frac",
        type=float,
        default=0.5,
        help="Fraction of training rows duplicated with flipped labels for stress test.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    input_csv = (root / args.input_csv).resolve() if not Path(args.input_csv).is_absolute() else Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    report_dir = root / "deliverables" / "report"
    figure_dir = root / "deliverables" / "figures"
    manifest_dir = root / "deliverables" / "manifests"
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    required_cols = {"url", "headline", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input CSV missing required columns: {required_cols}")
    df = df[list(required_cols)].copy()
    df["label"] = df["label"].astype(int)

    train_raw, test_raw = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["label"],
    )
    train_raw = train_raw.reset_index(drop=True)
    test_raw = test_raw.reset_index(drop=True)

    # Cleaning-effect branch
    train_loose = clean_dataframe(train_raw, strict=False)
    train_strict = clean_dataframe(train_raw, strict=True)
    test_strict = clean_dataframe(test_raw, strict=True)

    stress_raw = add_label_conflict_stress(train_raw, frac=args.stress_frac, random_state=123)
    stress_loose = clean_dataframe(stress_raw, strict=False)
    stress_strict = clean_dataframe(stress_raw, strict=True)

    results: List[EvalResult] = []
    results.append(
        run_one_eval(
            section="cleaning_effect",
            scenario="real_data_loose_clean",
            train_df=train_loose,
            test_df=test_strict,
            train_repr="headline_only",
            test_repr="headline_only",
            random_state=args.random_state,
            notes="Baseline real-data training without strict dedup/filters.",
        )
    )
    results.append(
        run_one_eval(
            section="cleaning_effect",
            scenario="real_data_strict_clean",
            train_df=train_strict,
            test_df=test_strict,
            train_repr="headline_only",
            test_repr="headline_only",
            random_state=args.random_state,
            notes="Real-data training with strict dedup and invalid-headline filters.",
        )
    )
    results.append(
        run_one_eval(
            section="cleaning_effect",
            scenario="stress_data_loose_clean",
            train_df=stress_loose,
            test_df=test_strict,
            train_repr="headline_only",
            test_repr="headline_only",
            random_state=args.random_state,
            notes="Stress train set with conflicting duplicates, no strict cleaning.",
        )
    )
    results.append(
        run_one_eval(
            section="cleaning_effect",
            scenario="stress_data_strict_clean",
            train_df=stress_strict,
            test_df=test_strict,
            train_repr="headline_only",
            test_repr="headline_only",
            random_state=args.random_state,
            notes="Stress train set with conflicting duplicates, strict cleaning enabled.",
        )
    )

    # Shortcut-risk branch (all on strict-clean train/test)
    results.append(
        run_one_eval(
            section="shortcut_risk",
            scenario="headline_to_headline",
            train_df=train_strict,
            test_df=test_strict,
            train_repr="headline_only",
            test_repr="headline_only",
            random_state=args.random_state,
            notes="Task-aligned setting: headline train and headline inference.",
        )
    )
    results.append(
        run_one_eval(
            section="shortcut_risk",
            scenario="url_to_headline",
            train_df=train_strict,
            test_df=test_strict,
            train_repr="url_only",
            test_repr="headline_only",
            random_state=args.random_state,
            notes="Shortcut mismatch: URL-train model tested on real headline text.",
        )
    )
    results.append(
        run_one_eval(
            section="shortcut_risk",
            scenario="url_to_url",
            train_df=train_strict,
            test_df=test_strict,
            train_repr="url_only",
            test_repr="url_only",
            random_state=args.random_state,
            notes="Shortcut-aligned proxy setting using URL pseudo text for both train and test.",
        )
    )
    results.append(
        run_one_eval(
            section="shortcut_risk",
            scenario="headline_plus_url_to_headline_plus_url",
            train_df=train_strict,
            test_df=test_strict,
            train_repr="headline_plus_url",
            test_repr="headline_plus_url",
            random_state=args.random_state,
            notes="Combined-feature model with both fields available at inference.",
        )
    )
    results.append(
        run_one_eval(
            section="shortcut_risk",
            scenario="headline_plus_url_to_headline_only",
            train_df=train_strict,
            test_df=test_strict,
            train_repr="headline_plus_url",
            test_repr="headline_only",
            random_state=args.random_state,
            notes="Combined-feature model evaluated when URL signal is missing at inference.",
        )
    )

    results_df = pd.DataFrame([r.__dict__ for r in results])
    results_csv = report_dir / "exploratory_results_step7.csv"
    results_df.to_csv(results_csv, index=False)

    figure_path = figure_dir / "exploratory_cleaning_shortcut_step7.png"
    make_plots(results_df, figure_path)

    metadata: Dict[str, object] = {
        "run_date_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_csv": str(input_csv),
        "random_state": args.random_state,
        "test_size": args.test_size,
        "stress_frac": args.stress_frac,
        "train_rows_raw": int(len(train_raw)),
        "test_rows_raw": int(len(test_raw)),
        "train_rows_strict": int(len(train_strict)),
        "test_rows_strict": int(len(test_strict)),
        "stress_rows_raw": int(len(stress_raw)),
        "stress_rows_loose": int(len(stress_loose)),
        "stress_rows_strict": int(len(stress_strict)),
    }
    meta_path = manifest_dir / "exploratory_metadata_step7.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_md = build_summary_md(results_df, metadata)
    summary_path = report_dir / "exploratory_summary_step7.md"
    summary_path.write_text(summary_md, encoding="utf-8")

    print(f"saved_results_csv: {results_csv.resolve()}")
    print(f"saved_figure: {figure_path.resolve()}")
    print(f"saved_summary_md: {summary_path.resolve()}")
    print(f"saved_metadata_json: {meta_path.resolve()}")
    print("----- metrics preview -----")
    print(results_df[["section", "scenario", "accuracy", "macro_f1", "avg_infer_ms", "train_rows", "test_rows"]].to_string(index=False))


if __name__ == "__main__":
    main()
