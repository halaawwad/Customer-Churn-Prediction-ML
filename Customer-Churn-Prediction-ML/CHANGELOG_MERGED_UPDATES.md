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
