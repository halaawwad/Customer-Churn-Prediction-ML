# HTML Dashboard

After each run of:

```bash
python src/churn_prediction.py
```

open:

```text
dashboard/index.html
```

The page reads the latest exported results from `dashboard/latest_results.js` and displays:

- Best model
- Accuracy, Precision, Recall, F1-score
- Model comparison table
- Generated charts from `results/figures`

You can double-click `index.html`. No server is required.
