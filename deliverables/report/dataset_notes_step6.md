# Dataset Notes (Step 6)

## Core Statistics

- Input URL rows: `3805`
- Raw rows: `3805`
- Clean rows: `3803`
- Removed from raw to clean: `2`
- Removed duplicates (`headline+label`): `2`
- Clean empty headlines: `0`
- Clean missing labels: `0`
- Clean duplicate (`headline+label`): `0`
- Source-label mismatch (clean): `0`

## Class Distribution (Clean)

- FoxNews (0): `2000`
- NBC (1): `1803`

## Headline Extraction Method (Raw)

- {'direct_html': 3801, 'wayback_html': 2, 'direct_html_variant': 1, 'url_slug_fallback': 1}

## Output Files

- `deliverables/report/dataset_statistics_step6.csv`
- `deliverables/report/dataset_class_distribution_step6.csv`
- `deliverables/figures/dataset_flow_step6.png`
- `deliverables/figures/dataset_class_distribution_step6.png`
- `deliverables/manifests/dataset_statistics_step6.json`
