# Model Manifest (Step 2)

- Timestamp: 2026-04-22 01:19:45 (America/New_York)
- Train script: `train_news_b_v1.py`
- Input dataset: `Newsheadlines/scraped_headlines_clean.csv`
- Final model path (overwritten): `Newsheadlines/artifacts/news_b_tfidf_lr.joblib`
- Backup model path: `deliverables/model/news_b_tfidf_lr_20260422_011945.joblib`
- SHA256 (both files): `fe418c6636ce619a609230b1181de76148299294300b0a9198e85cfc9d54019e`

## Training Metrics

- num_samples: 3803
- train_samples: 3042
- val_samples: 761
- val_accuracy: 0.800263
- val_macro_f1: 0.799055

## Post-update Evaluation

- Evaluator: `Newsheadlines/eval_project_b.py`
- Accuracy: 0.933474
- num_examples: 3803
- avg_infer_ms: 0.021

## Logs

- `deliverables/logs/train_step2_20260422_011945.log`
- `deliverables/logs/eval_after_step2_20260422_011945.log`
