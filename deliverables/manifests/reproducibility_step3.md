# Reproducibility Manifest (Step 3)

## Purpose

Freeze the reproducible evidence for Project B after Step 2 model update.

## Repro Commands

- Script: `deliverables/manifests/reproduce_step3.sh`
- Training command:
  - `conda run --no-capture-output -n cis5450 python train_news_b_v1.py --input-csv Newsheadlines/scraped_headlines_clean.csv --output-model deliverables/temp/retrain_step3_20260422_012254.joblib`
- Evaluation command:
  - `conda run --no-capture-output -n cis5450 python Newsheadlines/eval_project_b.py --model model.py --preprocess preprocess.py --csv Newsheadlines/scraped_headlines_clean.csv --batch-size 64`

## Artifact Integrity (SHA256)

- `Newsheadlines/scraped_headlines_clean.csv`
  - `f43c90e3ccd88e3206ccb188599ac0ffea5a1ba438b12ea0eb91b0b6970d1922`
- `Newsheadlines/artifacts/news_b_tfidf_lr.joblib`
  - `fe418c6636ce619a609230b1181de76148299294300b0a9198e85cfc9d54019e`
- `deliverables/temp/retrain_step3_20260422_012254.joblib`
  - `2ec35183540f2f17740678e1b4da07f987e5e9409f0b5b64294e1d089a34e4e1`

## Metrics Snapshot

- Train validation accuracy: `0.800263`
- Train validation macro F1: `0.799055`
- Post-update local evaluation accuracy: `0.933474`
- Evaluation examples: `3803`
- Average inference latency: `0.020 ms/sample`

## Baseline Check

- Course baseline accuracy: `0.6649`
- Current evaluation accuracy: `0.933474`
- Margin above baseline: `+0.268574`
- Status: `PASS`

## Evidence Files

- `deliverables/logs/retrain_step3_20260422_012254.log`
- `deliverables/logs/eval_step3_20260422_012254.log`
- `deliverables/manifests/environment_step3.txt`
- `deliverables/manifests/checksums_step3.sha256`
- `deliverables/manifests/reproducibility_step3.json`
