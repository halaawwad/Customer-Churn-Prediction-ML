# Merged Updates

This version adds the strongest feature-analysis and presentation parts:

- Correlation matrix between encoded dataset columns and the Churn target.
- `correlation_with_churn.csv` to rank features by relationship with churn.
- SelectKBest feature selection using ANOVA F-test.
- `selected_features.csv` with feature scores, p-values, and selected status.
- Model comparison bar chart.
- Radar chart comparing Accuracy, Precision, Recall, and F1-score.
- Dashboard updated to show the new generated charts after each run.

Run:

```bash
pip install -r requirements.txt
python src/churn_prediction.py
```

Then open:

```text
dashboard/index.html
```

## Latest Finalization Update
- Added PCA before/after comparison using Accuracy, feature count, training time, and prediction time.
- Added new PCA output files:
  - `results/pca_time_comparison.csv`
  - `results/figures/pca_training_time_comparison.png`
  - `results/figures/pca_prediction_time_comparison.png`
- Updated the dashboard layout with a dedicated PCA Impact section, improved visual ordering, and descriptions for every figure.
- Updated the final report by inserting a new PCA Impact and Computational Time page after Feature Analysis. The report remains under 12 pages.
- Optimized K-Means silhouette score computation using a representative sample to avoid long runtime.
