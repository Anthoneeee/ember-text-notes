# Model Manifest (Full Retrain Update)

- Timestamp: 2026-04-22 01:59:36 (America/New_York)
- Trigger: P1 fix (validation-first + full-dataset retrain before export)
- Train script: `train_news_b_v1.py`
- Input dataset: `Newsheadlines/scraped_headlines_clean.csv`
- Final model path (overwritten): `Newsheadlines/artifacts/news_b_tfidf_lr.joblib`
- Backup model path: `deliverables/model/news_b_tfidf_lr_full_retrain_20260422_015936.joblib`
- SHA256 (both files): `8c92cd05685dbbbfd83d8286dbe82bc6f8b7b4fd0723f2272b9c7b3615b073ae`

## Validation Stage Metrics

- num_samples: 3803
- train_samples: 3042
- val_samples: 761
- val_accuracy: 0.8002628120893561
- val_macro_f1: 0.7990549649086234

## Export Mode

- final_train_mode: full_dataset_retrain_after_validation
- final_retrain_on_full: true

## Post-update Evaluation Snapshot

- Evaluator: `Newsheadlines/eval_project_b.py`
- Accuracy: 0.963187
- num_examples: 3803
- avg_infer_ms: 0.020

## Logs

- `deliverables/logs/train_full_retrain_update_20260422_015936.log`
- `deliverables/logs/eval_after_full_retrain_update_20260422_015936.log`
