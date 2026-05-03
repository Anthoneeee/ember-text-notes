# CIS 4190/5190 Project B: News Headline Classifier

This repository contains the Project B submission for classifying whether a news headline
comes from Fox News (`0`) or NBC News (`1`).

## Current Best Submission

Observed Hugging Face leaderboard accuracy:

- `0.8741666666666666`

Current best variant:

- `cur_nb_sgdcal_t5525`
- decision threshold: `0.5525`
- headline-only ensemble using TF-IDF/LR style branches plus ComplementNB and calibrated SGD

The active root submission files are:

- `model.py`
- `preprocess.py`
- `model.pt`

These are identical to the copies in:

- `submission_hf_urlfetch/`
- `experiments/submission_hf_best_8742_cur_nb_sgdcal_t5525/`

Use the root files or `submission_hf_urlfetch/` for Hugging Face upload. The `experiments/`
copy is kept as a stable backup of the same best version.

## Compliance Notes

Project B is a headline classification task. URL handling is limited to obtaining the
article headline text when the evaluator provides URL-only rows.

The submitted feature text does not include URL domains, URL paths, article ids, or source
leaking tokens. Outlet names are also masked during normalization. If a URL-only CSV does
not include labels, `preprocess.py` may infer the target `y` from the URL host so the
evaluator can compute accuracy, but that URL information is not returned as model input.

## Runtime Requirements

The model artifact was serialized with:

- Python 3.x
- `torch`
- `pandas`
- `scikit-learn==1.7.2`
- `joblib`

Use the `cis5450` conda environment locally if available:

```bash
conda run -n cis5450 python -m py_compile model.py preprocess.py
conda run -n cis5450 python -c "import model; m=model.get_model(); print(m.decision_threshold, m.decision_threshold_kind)"
```

Using an older scikit-learn version such as `1.1.1` can fail when loading `model.pt`,
because the packed estimator was built with `1.7.2`.

## Main Files

- `model.py`: HF-compatible model wrapper with `get_model()`.
- `preprocess.py`: HF-compatible `prepare_data(csv_path)` implementation.
- `model.pt`: packed trained model artifact.
- `submission_hf_urlfetch/`: clean upload copy of the current best three files.
- `experiments/submission_hf_best_8742_cur_nb_sgdcal_t5525/`: stable backup of the current best three files.
- `Newsheadlines/scrape_headlines.py`: headline scraping utility.
- `Newsheadlines/eval_project_b.py`: local evaluator utility.
- `deliverables/dataset/`: cleaned dataset artifacts for the final report/submission.
- `deliverables/external_headlines/`: collected external headline data and raw scrape outputs.
- `deliverables/figures/`: report figure assets.
- `deliverables/report/`: report tables and notes.
- `deliverables/manifests/`: reproducibility notes and final packaging helpers.

## Final Deliverables Still Needed

Per the project documents, the final course submission should include:

- collected dataset
- trained model files
- 5-page project report
- Hugging Face Dataset link in the report
- model metric line chart(s), including the baseline metric
- clear explanation of data collection, cleaning, modeling, evaluation, and leaderboard selection

The Hugging Face model upload itself only needs:

- `model.py`
- `preprocess.py`
- `model.pt`
