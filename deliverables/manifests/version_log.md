# Project B Model Version Log

## Restored HF 0.8425 Version - 2026-05-02

- Status: restored as the active root submission and copied to `submission_hf_urlfetch/`.
- Files to upload: `model.py`, `preprocess.py`, `model.pt`.
- Observed Hugging Face accuracy: `0.8425`.
- Model: `vote_word_char_dlr`.
- Validation used during restore: hidden-like headline split accuracy `0.848939`.
- Training data: base cleaned real headlines plus curated external real headlines from `external_headlines_clean.csv`.
- Compliance note: headline-only classifier; URL is only used by `preprocess.py` to fetch a real page headline. No URL slug/path/domain/source-derived text is used as a model feature.
- Label handling note: `preprocess.py` prefers explicit labels/source columns. For the Hugging Face URL-only `url_val` format, it derives only the target `y` from the URL host so the evaluator can score predictions; the URL host/path is never returned as an input feature.
- Rollback reason: the later NB + broader extractor version performed worse on Hugging Face.

## SVM Stack Experiment - 2026-05-02

- Status: experimental candidate only; active root files and `submission_hf_urlfetch/` remain the restored HF `0.8425` baseline.
- Candidate files: `experiments/submission_hf_svmstack/model.py`, `experiments/submission_hf_svmstack/preprocess.py`, `experiments/submission_hf_svmstack/model.pt`.
- Model: `svmstack_nb_probe`.
- Idea: add calibrated LinearSVC and a small ComplementNB branch to the restored LR/DLR headline-only ensemble.
- Hidden-like validation: `0.855181`, compared with restored baseline validation `0.848939`.
- Full clean local evaluation: `0.962135`; this is not the main selection signal because the final model trains on the full base set.
- Model size: about `38M`.
- Compliance note: headline-only classifier; no URL slug/path/domain/source-derived text. For URL-only evaluator CSVs, URL host is used only to build target `y`, not model input `X`.
- Recommendation: upload this from `experiments/submission_hf_svmstack/` only as an A/B HF test. Keep `submission_hf_urlfetch/` as the fallback unless HF confirms improvement over `0.8425`.

## BERT-Tiny Transformer Experiment - 2026-05-02

- Status: experimental only; not recommended for Hugging Face upload unless we intentionally want a transformer smoke test.
- Candidate files: `experiments/submission_hf_berttiny/model.py`, `experiments/submission_hf_berttiny/preprocess.py`, `experiments/submission_hf_berttiny/model.pt`.
- Stable fallback preserved: `submission_hf_urlfetch/` remains the observed HF `0.8425` version.
- Pretrained model: `prajjwal1/bert-tiny`, implemented with local PyTorch-only BERT layers so the submission does not require the `transformers` package at evaluation time.
- Validation signal: BERT-tiny alone reached about `0.815231` on the hidden-like split, below the linear LR/DLR baseline.
- Final packed behavior: includes the restored LR/DLR baseline artifact and BERT-tiny weights; tuned transformer mix weight is `0.0`, so this artifact effectively falls back to the stable baseline while adding transformer overhead.
- Local clean evaluation: `0.964765`; this is not evidence of HF improvement because the final model includes the full base training set.
- Model size: about `53M`, larger than the stable `36M`.
- Recommendation: do not upload this version for score. A larger transformer such as BERT-mini/DistilBERT may be more useful, but would increase model size and HF runtime substantially.

## URL Fetch LR-Tuned Submission - 2026-05-02

- Status: rejected after Hugging Face A/B test; no longer active.
- Stable fallback copy: `experiments/submission_hf_urlfetch_stable_8425/`.
- Experiment source: `experiments/submission_hf_urlfetch_lr_tuned/`.
- Model: `vote_no_dlr_curated`, a headline-only `word_char LogisticRegression + char LogisticRegression` soft vote.
- Change from HF `0.8425` baseline: removed the DLR/SVD/RBF branch after multi-split validation showed it was not helping enough and increased variance/runtime.
- Hugging Face observed accuracy: `0.8208333333333333`, below the `0.8425` stable baseline.
- Postmortem: local multi-split validation was misleading for the hidden URL validation set; the DLR/SVD/RBF branch appears important for the HF distribution.
- Multi-split validation: mean accuracy `0.859461`, std `0.004943`, min `0.851436` over seeds `[11, 23, 42, 57, 71, 89, 101]`.
- Baseline comparison in the same search: `stable_vote_curated` mean `0.856965`, min `0.848939`.
- Decision threshold: median split threshold `0.45`.
- Local clean evaluation after final training: `0.964765`.
- Model size: about `1.2M`, compared with `36M` for the previous stable artifact.
- Compliance note: headline-only classifier; no URL slug/path/domain/source-derived text. For URL-only evaluator CSVs, URL host is used only to build target `y`, not model input `X`.

## DLR Gamma Enhancement Experiment - 2026-05-02

- Status: experimental candidate only; `submission_hf_urlfetch/` has been restored to the observed HF `0.8425` stable baseline.
- Candidate files: `experiments/submission_hf_dlr_enhanced/model.py`, `experiments/submission_hf_dlr_enhanced/preprocess.py`, `experiments/submission_hf_dlr_enhanced/model.pt`.
- Model: `dlr_gamma04_curated`.
- Change from stable baseline: keep the same `word_char + char + dlr_rbf` structure, keep the stable threshold `0.4825`, change the DLR RBF gamma from `0.8` to `0.4`, and slightly increase DLR vote weight to `[2.0, 1.0, 1.2]`.
- Search constraint: no SVM, no BERT, no NB, no URL/path/domain features; only DLR weight/gamma variants and DLR-preserving votes were compared.
- Multi-split validation: candidate mean accuracy `0.855930`, min `0.850187`; same-run stable baseline mean `0.854682`, min `0.848939`.
- Local clean evaluation: `0.961872`.
- Model size: about `36M`, similar to the stable baseline.
- Hugging Face observed accuracy: `0.8442`, the current best observed score for our compliant headline-only line.
- Frozen copy: `experiments/submission_hf_dlr_enhanced_8442/`.

## DLR Refined SVD Experiment - 2026-05-02

- Status: experimental candidate only; does not overwrite the HF `0.8442` copy.
- Candidate files: `experiments/submission_hf_dlr_refined/model.py`, `experiments/submission_hf_dlr_refined/preprocess.py`, `experiments/submission_hf_dlr_refined/model.pt`.
- Model: `gamma04_svd180`.
- Change from HF `0.8442` model: keep `gamma=0.4`, vote weights `[2.0, 1.0, 1.2]`, threshold `0.4825`, and increase the DLR SVD dimension from `160` to `180`.
- Search constraint: only small local refinements around the HF-observed `0.8442` model; no URL/path/domain features, no SVM/BERT/NB, no scraper changes.
- Multi-split validation: reference `hf8442_gamma04_w12` mean `0.858213`, min `0.850187`; refined `gamma04_svd180` mean `0.860175`, min `0.851436`.
- Local clean evaluation: `0.964239`.
- Model size: about `40M`, compared with `36M` for the `0.8442` model.
- Recommendation: upload this as the next A/B test. If it does not beat `0.8442`, use `experiments/submission_hf_dlr_enhanced_8442/` as the best known version.

## DLR Quote/Outlet Mask Experiment - 2026-05-03

- Status: Hugging Face confirmed improvement and copied to active `submission_hf_urlfetch/`.
- Candidate files: `experiments/submission_hf_dlr_quote_outlet_mask/model.py`, `experiments/submission_hf_dlr_quote_outlet_mask/preprocess.py`, `experiments/submission_hf_dlr_quote_outlet_mask/model.pt`.
- Model: `dlr_gamma04_curated_quote_outlet_mask`.
- Change from HF `0.8442` model: keep `word_char + char + dlr_rbf`, vote weights `[2.0, 1.0, 1.2]`, and threshold `0.4825`; only normalize curly quotes and mask explicit `Fox News`, `NBC News`, `MSNBC`, and all-caps `TODAY` mentions to `the outlet`.
- Search constraint: no architecture/capacity/threshold/data-source changes; only headline text preprocessing changed in both training and inference.
- Multi-split validation: mean accuracy `0.844694`, min `0.838951`, mean macro F1 `0.836740` over seeds `[11, 23, 42, 57, 71]`.
- Local clean evaluation: `0.964502`.
- Model size: about `36M`, similar to the HF `0.8442` model.
- Hugging Face observed accuracy: `0.8458`, the current best known score.
- Frozen copy: `experiments/submission_hf_dlr_quote_outlet_mask_8458/`.

## DLR Extended Outlet Mask Experiment - 2026-05-03

- Status: experimental next A/B candidate; does not overwrite the HF `0.8458` copy.
- Candidate files: `experiments/submission_hf_dlr_next/model.py`, `experiments/submission_hf_dlr_next/preprocess.py`, `experiments/submission_hf_dlr_next/model.pt`.
- Model: `quote_outlet_extended`.
- Change from HF `0.8458` model: keep the same `word_char + char + dlr_rbf` structure, weights `[2.0, 1.0, 1.2]`, and threshold `0.4825`; extend the outlet-name neutralization to other named media outlets such as `CNN`, `ABC News`, `CBS News`, `Newsmax`, `NewsNation`, `AP`, `Reuters`, `New York Times`, `Washington Post`, and `Wall Street Journal`.
- Search result: `quote_outlet_extended` mean `0.845194`, min `0.841448`, mean macro F1 `0.837227` over seeds `[11, 23, 42, 57, 71]`; HF `0.8458` reference mean `0.844694`, min `0.838951`.
- Negative result: hard-topic row duplication and limited external hard-topic augmentation both hurt validation, so they were not selected.
- Local clean evaluation: `0.963187`.
- Model size: about `36M`.
- Recommendation: upload `experiments/submission_hf_dlr_next/` as the next low-risk A/B test. If it does not beat `0.8458`, keep `experiments/submission_hf_dlr_quote_outlet_mask_8458/` and active `submission_hf_urlfetch/`.
