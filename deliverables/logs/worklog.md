# Work Log

## 2026-04-22

- Initialized deliverables workspace.
- Created subdirectories: dataset, model, report, figures, logs, manifests, notes, temp.
- Added workspace README and submission checklist template.

## 2026-04-22 (Step 2)

- Retrained model with `train_news_b_v1.py` using `Newsheadlines/scraped_headlines_clean.csv`.
- Overwrote final model path: `Newsheadlines/artifacts/news_b_tfidf_lr.joblib`.
- Validation metrics during training: `val_accuracy=0.800263`, `val_macro_f1=0.799055`.
- Post-update local evaluation accuracy: `0.933474` (on `scraped_headlines_clean.csv`).
- Backed up model to `deliverables/model/news_b_tfidf_lr_20260422_011945.joblib`.

## 2026-04-22 (Step 3)

- Frozen reproducibility evidence with commands, logs, checksums, and environment versions.
- Re-ran training to temp artifact and re-ran evaluation for verification.
- Verified baseline check: accuracy 0.933474 > 0.6649 (PASS).
- Added `reproduce_step3.sh`, `checksums_step3.sha256`, and structured manifest files.

## 2026-04-22 (Step 4)

- Exported metrics summary artifacts for report usage.
- Added structured metrics JSON, markdown summary, and CSV metric table.
- Metrics covered: accuracy, macro/weighted F1, class-level PRF, dataset scale, baseline gap, and training parameters.

## 2026-04-22 (Step 5)

- Generated baseline vs current model comparison figures for report.
- Added accuracy-only line chart and multi-metric line chart.
- Added source table and interpretation notes for report writing.

## 2026-04-22 (Step 6)

- Generated dataset statistics tables and plots (data flow, class distribution).
- Exported dataset profile JSON + report notes for direct use in project report.
- Copied final clean dataset into deliverables/dataset and recorded checksum.

## 2026-04-22 (Model Update after P1 fix)

- Retrained and overwrote `Newsheadlines/artifacts/news_b_tfidf_lr.joblib` using updated two-stage logic.
- Validation stage metrics: val_accuracy=0.800263, val_macro_f1=0.799055.
- Final export mode: full_dataset_retrain_after_validation.
- Post-update local evaluator accuracy on clean dataset: 0.963187.
- Backed up model to `deliverables/model/news_b_tfidf_lr_full_retrain_20260422_015936.joblib`.
