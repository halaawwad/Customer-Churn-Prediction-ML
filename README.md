# Customer Churn Prediction Using Machine Learning

This repository contains a complete final project solution for the Data Mining and Machine Learning course.

## Project Goal
Predict whether a telecommunication customer will churn using supervised machine learning models, then analyze the dataset using feature analysis, PCA dimensionality reduction, and K-Means clustering.

## Models Included
- Decision Tree Classifier
- Support Vector Machine (SVM)
- K-Means clustering for customer segmentation

## Main Pipeline
1. Data understanding and exploratory visualizations
2. Data cleaning and preprocessing
3. Stratified 70/30 train-test split
4. Feature analysis and dimensionality discussion
5. PCA before/after comparison using accuracy, feature count, training time, and prediction time
6. Decision Tree and SVM model training
7. Model evaluation using accuracy, precision, recall, F1-score, and confusion matrices
8. K-Means clustering analysis
9. Dashboard, report, and presentation deliverables

## How to Run
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure the dataset file exists here:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

3. Run the main script:

```bash
python src/churn_prediction.py
```

4. Open the dashboard:

```text
dashboard/index.html
```

## Key Outputs

### Text and CSV outputs
- `results/eda_summary.txt`
- `results/model_comparison.csv`
- `results/decision_tree_classification_report.txt`
- `results/svm_classification_report.txt`
- `results/correlation_matrix.csv`
- `results/correlation_with_churn.csv`
- `results/selected_features.csv`
- `results/pca_accuracy_comparison.csv`
- `results/pca_feature_count_comparison.csv`
- `results/pca_time_comparison.csv`
- `results/pca_comparison_summary.txt`
- `results/kmeans_summary.txt`

### Figure outputs
- `results/figures/churn_distribution.png`
- `results/figures/contract_vs_churn.png`
- `results/figures/tenure_distribution.png`
- `results/figures/monthly_charges_distribution.png`
- `results/figures/tenure_by_churn_boxplot.png`
- `results/figures/payment_method_vs_churn.png`
- `results/figures/correlation_matrix.png`
- `results/figures/selectkbest_top_features.png`
- `results/figures/feature_importance_top15.png`
- `results/figures/decision_tree_structure.png`
- `results/figures/pca_accuracy_comparison.png`
- `results/figures/pca_feature_count_comparison.png`
- `results/figures/pca_training_time_comparison.png`
- `results/figures/pca_prediction_time_comparison.png`
- `results/figures/model_comparison_chart.png`
- `results/figures/radar_chart.png`
- `results/figures/confusion_matrix_decision_tree.png`
- `results/figures/confusion_matrix_svm.png`
- `results/figures/kmeans_pca_clusters.png`

## Dataset
Official dataset page:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Deliverables Included
- Source code in `src/`
- Notebook in `notebooks/`
- Final report in `report/`
- Presentation in `presentation/`
- Dashboard in `dashboard/`
- Arabic explanation file
- Requirements file

## Notes
PCA is evaluated by comparing feature count, accuracy, training time, and prediction time before and after dimensionality reduction. The goal is not only to improve accuracy, but also to reduce feature complexity and computational cost while preserving useful information.
