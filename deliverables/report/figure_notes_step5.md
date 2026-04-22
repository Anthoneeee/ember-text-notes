# Figure Notes (Step 5)

- Accuracy chart: `deliverables/figures/baseline_vs_current_accuracy_step5.png`
- Multi-metric chart: `deliverables/figures/baseline_vs_current_multimetric_step5.png`
- Data table: `deliverables/report/baseline_vs_current_step5.csv`

## Interpretation

1. Relative to the course baseline (`0.6649`), the current model improves to `0.8003` on a comparable validation split.
2. Full-dataset evaluation reaches `0.9335`, but this setting is optimistic and should be interpreted separately from hidden-test generalization.
3. Report text should explicitly distinguish validation-split metrics vs in-sample/full-dataset snapshot metrics.
