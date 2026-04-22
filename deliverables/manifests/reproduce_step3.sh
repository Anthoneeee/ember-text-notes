#!/usr/bin/env bash
set -euo pipefail

# Reproduce training + evaluation for Project B final model.
# Expected environment: conda env `cis5450`

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

TS="$(date +"%Y%m%d_%H%M%S")"

conda run --no-capture-output -n cis5450 python train_news_b_v1.py \
  --input-csv Newsheadlines/scraped_headlines_clean.csv \
  --output-model deliverables/temp/retrain_${TS}.joblib \
  2>&1 | tee deliverables/logs/retrain_${TS}.log

conda run --no-capture-output -n cis5450 python Newsheadlines/eval_project_b.py \
  --model model.py \
  --preprocess preprocess.py \
  --csv Newsheadlines/scraped_headlines_clean.csv \
  --batch-size 64 \
  2>&1 | tee deliverables/logs/eval_${TS}.log

shasum -a 256 Newsheadlines/scraped_headlines_clean.csv \
  Newsheadlines/artifacts/news_b_tfidf_lr.joblib \
  deliverables/temp/retrain_${TS}.joblib \
  > deliverables/manifests/checksums_${TS}.sha256

echo "Reproduction run completed: ${TS}"
