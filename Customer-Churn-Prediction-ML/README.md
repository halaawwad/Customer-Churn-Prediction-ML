# Customer Churn Prediction Using Machine Learning

This package contains a complete final project solution for the Data Mining and Machine Learning course.

## Project Goal
Predict whether a telecommunication customer will churn using supervised machine learning models.

## Models Included
- Decision Tree Classifier
- Support Vector Machine (SVM)
- K-Means clustering for customer segmentation

## Main Pipeline
1. Data understanding
2. Data cleaning and preprocessing
3. Train/test split using 70/30 sampling
4. Feature analysis and dimensionality discussion
5. Model training
6. Model evaluation
7. Clustering analysis
8. Report and presentation deliverables

## How to Run
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Put the dataset file here:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

3. Run the main script:

```bash
python src/churn_prediction.py
```

4. Check outputs:

```text
results/model_comparison.csv
results/figures/
results/*.txt
```

## Dataset
Official dataset page:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Deliverables Included
- Source code in `src/`
- Notebook in `notebooks/`
- Final report in `report/`
- Presentation in `presentation/`
- Arabic explanation file
- Requirements file

## Added Feature Analysis Outputs

After running the project, check these new outputs:

- `results/correlation_matrix.csv`
- `results/correlation_with_churn.csv`
- `results/selected_features.csv`
- `results/figures/correlation_matrix.png`
- `results/figures/selectkbest_top_features.png`
- `results/figures/model_comparison_chart.png`
- `results/figures/radar_chart.png`

