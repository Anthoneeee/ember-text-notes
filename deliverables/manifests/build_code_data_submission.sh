#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

OUT="deliverables/manifests/code_data_submission_$(date +"%Y%m%d_%H%M%S").tar.gz"

# Build from explicit whitelist (code + data only)
 tar -czf "$OUT" \
  model.py \
  preprocess.py \
  news_b_utils.py \
  train_news_b_v1.py \
  Newsheadlines/scrape_headlines.py \
  Newsheadlines/artifacts/news_b_tfidf_lr.joblib \
  deliverables/dataset/scraped_headlines_clean_final.csv \
  README.md \
  Newsheadlines/url_only_data.csv \
  Newsheadlines/scraped_headlines_raw.csv \
  Newsheadlines/scraped_headlines_clean.csv

 echo "Created: $OUT"
 tar -tzf "$OUT"
