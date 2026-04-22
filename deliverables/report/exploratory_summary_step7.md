# Exploratory Component (Step 7)

## Topic
Data Cleaning and Shortcut Risk: Effects on Headline Classification Generalization

## Setup
- Run date (local): 2026-04-22 18:47:28
- Input dataset: `/Users/anthony/PycharmProjects/pythonProject1/CIS5190/Newsheadlines/scraped_headlines_clean.csv`
- Split: stratified holdout, test_size=0.2, random_state=42

## Key Findings
- Shortcut risk is clear: URL-only training reaches 0.9671 on URL-style test but only 0.5440 on real headlines.
- Headline-first training is more reliable for the actual task: 0.8003 on headline test.
- A model trained on headline+URL can become URL-dependent: 0.9566 with full input vs 0.6531 when URL features are absent.
- On this crawl, real-data cleaning deltas are small (0.8003 -> 0.8003) because the dataset is already clean.
- In stress test with injected conflicting duplicates, strict cleaning protects generalization (0.7595 -> 0.8003).

## Recommendation
- Keep headline-only as the final training target for Project B.
- Keep strict cleaning enabled (dedup URLs/headlines + invalid headline filters).
- Use URL-based features only as diagnostic experiments, not as primary training signal.

## Output Files
- `deliverables/report/exploratory_results_step7.csv`
- `deliverables/report/exploratory_summary_step7.md`
- `deliverables/manifests/exploratory_metadata_step7.json`
- `deliverables/figures/exploratory_cleaning_shortcut_step7.png`
