# Final Submission Whitelist (Code + Data Artifacts)

This whitelist is for a **code + data only** package.
It intentionally excludes report/figures/logs/temp files.

## Include (Required)

- `model.py`
- `preprocess.py`
- `news_b_utils.py`
- `train_news_b_v1.py`
- `Newsheadlines/scrape_headlines.py`
- `Newsheadlines/artifacts/news_b_tfidf_lr.joblib`
- `deliverables/dataset/scraped_headlines_clean_final.csv`

## Include (Recommended for reproducibility)

- `README.md`
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
- `final_project_descriptions_extracted_clean.txt`
- `CIS 4190_5190 Final Project Descriptions.pdf`
- `Newsheadlines/_jina_check.py`

## Notes

- Keep exactly one final model artifact (`news_b_tfidf_lr.joblib`).
- Keep exactly one final dataset artifact (`scraped_headlines_clean_final.csv`).
- If submission size is limited, prioritize Required section first.
