# Final Submission Whitelist (Code + Data Artifacts)

This whitelist is for a **code + data + model** package.
It intentionally excludes local caches, virtual environments, old logs, and assignment PDFs.

## Include (Required)

- `model.py`
- `preprocess.py`
- `model.pt`
- `news_b_utils.py`
- `train_news_b_v1.py`
- `train_char_lr_final.py`
- `build_compliant_external.py`
- `Newsheadlines/scrape_headlines.py`
- `deliverables/dataset/scraped_headlines_clean_final.csv`
- `deliverables/dataset/scraped_headlines_clean_headline_only.csv`
- `deliverables/external_headlines/processed/external_headlines_compliant.csv`

## Include (Recommended for reproducibility)

- `README.md`
- `submission_hf_urlfetch/model.py`
- `submission_hf_urlfetch/preprocess.py`
- `submission_hf_urlfetch/model.pt`
- `experiments/submission_hf_best_8742_cur_nb_sgdcal_t5525/model.py`
- `experiments/submission_hf_best_8742_cur_nb_sgdcal_t5525/preprocess.py`
- `experiments/submission_hf_best_8742_cur_nb_sgdcal_t5525/model.pt`
- `Newsheadlines/url_only_data.csv`
- `Newsheadlines/scraped_headlines_raw.csv`
- `Newsheadlines/scraped_headlines_clean.csv`

## Optional (only if staff asks)

- `Newsheadlines/eval_project_b.py`

## Exclude (Do NOT submit)

- `.venv/`
- `__pycache__/`
- `Newsheadlines/__pycache__/`
- `.DS_Store`
- `deliverables/temp/`
- `deliverables/logs/`
- `deliverables/report/`
- `deliverables/figures/`
- `final_project_descriptions_extracted.txt`
- `CIS 4190_5190 Final Project Descriptions.pdf`
- `Project_Submission.pdf`

## Notes

- Keep exactly one active final model artifact (`model.pt`) with matching `model.py` and `preprocess.py`.
- Keep exactly one final dataset artifact (`scraped_headlines_clean_final.csv`).
- `final_project_descriptions_extracted_clean.txt` and the two assignment PDFs are retained locally for reference, but should not be part of the course submission package unless explicitly requested.
