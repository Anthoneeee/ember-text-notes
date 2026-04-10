# Ember Text Notes

This repository contains a text-source classification workflow and supporting data processing scripts.

## Structure
- `model.py`, `preprocess.py`: inference/evaluation entry points
- `news_b_utils.py`, `train_news_b_v1.py`: data prep and training utilities
- `Newsheadlines/`: evaluator, datasets, scraping pipeline, and artifacts

## Notes
- The dataset build script prioritizes direct webpage extraction and includes robust fallbacks.
- This repo is configured as public for team collaboration.
