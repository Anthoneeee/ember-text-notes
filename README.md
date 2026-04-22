# CIS 4190/5190 Project B: News Source Classification

This project builds a binary text classifier to predict whether a headline is from Fox News (`0`) or NBC (`1`).

## Overview

The pipeline has three stages:

1. Scrape headlines from URL lists.
2. Train a TF-IDF + Logistic Regression model.
3. Evaluate the model with the local Project B evaluator.

## Environment Setup

Use Python `3.10` (recommended with `conda`):

```bash
conda create -n cis5450 python=3.10 -y
conda activate cis5450
pip install pandas numpy scikit-learn requests beautifulsoup4 joblib torch
```

All commands below assume you run from the project root (the folder containing `model.py`).

## Key Files

- `Newsheadlines/scrape_headlines.py`: headline scraping and cleaning pipeline
- `Newsheadlines/url_only_data.csv`: starter URL list
- `Newsheadlines/scraped_headlines_raw.csv`: raw scrape output with metadata
- `Newsheadlines/scraped_headlines_clean.csv`: cleaned dataset for training
- `train_news_b_v1.py`: training script
- `model.py`: inference wrapper used by evaluator
- `preprocess.py`: preprocessing entry for evaluator
- `Newsheadlines/eval_project_b.py`: local evaluation script

## Recommended Workflow

### 1. Scrape Headlines

```bash
python -u Newsheadlines/scrape_headlines.py \
  --input-csv Newsheadlines/url_only_data.csv \
  --output-raw Newsheadlines/scraped_headlines_raw.csv \
  --output-clean Newsheadlines/scraped_headlines_clean.csv \
  --max-workers 4 \
  --timeout 10 \
  --retries 1 \
  --min-delay 0.15 \
  --max-delay 0.55 \
  --allow-url-fallback
```

Check the summary at the end, especially:

- `headline_method_counts`
- `saved_raw`
- `saved_clean`

### 2. Train Model

```bash
python train_news_b_v1.py \
  --input-csv Newsheadlines/scraped_headlines_clean.csv \
  --output-model Newsheadlines/artifacts/news_b_tfidf_lr.joblib
```

Important training behavior:

- Default input is `Newsheadlines/scraped_headlines_clean.csv`.
- By default, the script rejects URL-only CSVs without headline/title/text columns.
- Validation metrics are computed on a split first, then the final exported model is retrained on the full dataset.

If you intentionally want URL pseudo-text experiments:

```bash
python train_news_b_v1.py \
  --input-csv Newsheadlines/url_only_data.csv \
  --allow-url-pseudo-text
```

### 3. Evaluate Locally

```bash
python Newsheadlines/eval_project_b.py \
  --model model.py \
  --preprocess preprocess.py \
  --csv Newsheadlines/scraped_headlines_clean.csv \
  --batch-size 64
```

## Quick Evaluation (No Retraining)

If `Newsheadlines/artifacts/news_b_tfidf_lr.joblib` already exists:

```bash
python Newsheadlines/eval_project_b.py \
  --model model.py \
  --preprocess preprocess.py \
  --csv Newsheadlines/scraped_headlines_clean.csv \
  --batch-size 64
```

## Final Code + Data Artifacts

Current minimal set:

- `model.py`
- `preprocess.py`
- `news_b_utils.py`
- `train_news_b_v1.py`
- `Newsheadlines/scrape_headlines.py`
- `Newsheadlines/artifacts/news_b_tfidf_lr.joblib`
- `deliverables/dataset/scraped_headlines_clean_final.csv`

## Troubleshooting

### `No module named torch`

Install in current environment:

```bash
pip install torch
```

### Missing evaluator arguments

`eval_project_b.py` requires:

- `--model`
- `--preprocess`
- `--csv`

### Frequent `403/406` during scraping

Current scraper already includes:

- request delays
- retry logic
- URL variants
- Wayback fallback
- Jina mirror fallback
- optional URL-slug fallback

To be more conservative, reduce concurrency (for example `--max-workers 2`).
