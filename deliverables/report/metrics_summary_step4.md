# Metrics Summary (Step 4)

## Core Metrics

- Full-dataset accuracy: `0.933474`
- Full-dataset macro F1: `0.933266`
- Full-dataset weighted F1: `0.933459`
- Validation accuracy (training split): `0.800263`
- Validation macro F1 (training split): `0.799055`

## Dataset Scale

- Dataset: `Newsheadlines/scraped_headlines_clean.csv`
- Total samples: `3803`
- Class distribution:
  - FoxNews (0): `2000`
  - NBC (1): `1803`

## Class-level Metrics (Full Dataset)

- FoxNews (0): precision `0.933499`, recall `0.940500`, f1 `0.936986`, support `2000`
- NBC (1): precision `0.933445`, recall `0.925679`, f1 `0.929546`, support `1803`

## Baseline Comparison

- Course baseline accuracy: `0.664900`
- Current accuracy: `0.933474`
- Margin: `+0.268574`

## Training Parameters Snapshot

- test_size: `0.2`
- random_state: `42`
- max_features: `30000`
- ngram_max: `2`
- TF-IDF: `lowercase=True`, `strip_accents=unicode`, `ngram_range=(1,2)`, `min_df=2`, `max_df=0.98`, `sublinear_tf=True`
- LogisticRegression: `max_iter=2000`, `solver=liblinear`, `C=2.0`, `class_weight=balanced`, `random_state=42`

## Source Files

- Training log: `deliverables/logs/train_step2_20260422_011945.log`
- Evaluation log: `deliverables/logs/eval_step3_20260422_012254.log`
- Structured summary: `deliverables/manifests/metrics_summary_step4.json`
