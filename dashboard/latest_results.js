window.CHURN_DASHBOARD_DATA = {
  "generated_at": "2026-05-12 10:24:37",
  "metrics": [
    {
      "Model": "Decision Tree",
      "Accuracy": 0.71557,
      "Precision": 0.478214,
      "Recall": 0.782531,
      "F1_Score": 0.593644
    },
    {
      "Model": "SVM",
      "Accuracy": 0.745386,
      "Precision": 0.513326,
      "Recall": 0.789661,
      "F1_Score": 0.622191
    }
  ],
  "figures": [
    {
      "title": "Churn Distribution",
      "path": "../results/figures/churn_distribution.png"
    },
    {
      "title": "Contract vs Churn",
      "path": "../results/figures/contract_vs_churn.png"
    },
    {
      "title": "Tenure Distribution",
      "path": "../results/figures/tenure_distribution.png"
    },
    {
      "title": "Monthly Charges Distribution",
      "path": "../results/figures/monthly_charges_distribution.png"
    },
    {
      "title": "Tenure by Churn",
      "path": "../results/figures/tenure_by_churn_boxplot.png"
    },
    {
      "title": "Payment Method vs Churn",
      "path": "../results/figures/payment_method_vs_churn.png"
    },
    {
      "title": "Correlation Matrix",
      "path": "../results/figures/correlation_matrix.png"
    },
    {
      "title": "SelectKBest Top Features",
      "path": "../results/figures/selectkbest_top_features.png"
    },
    {
      "title": "PCA Accuracy Comparison",
      "path": "../results/figures/pca_accuracy_comparison.png"
    },
    {
      "title": "PCA Feature Count Comparison",
      "path": "../results/figures/pca_feature_count_comparison.png"
    },
    {
      "title": "PCA Training Time Comparison",
      "path": "../results/figures/pca_training_time_comparison.png"
    },
    {
      "title": "PCA Prediction Time Comparison",
      "path": "../results/figures/pca_prediction_time_comparison.png"
    },
    {
      "title": "Model Comparison Chart",
      "path": "../results/figures/model_comparison_chart.png"
    },
    {
      "title": "Radar Chart",
      "path": "../results/figures/radar_chart.png"
    },
    {
      "title": "Decision Tree Confusion Matrix",
      "path": "../results/figures/confusion_matrix_decision_tree.png"
    },
    {
      "title": "SVM Confusion Matrix",
      "path": "../results/figures/confusion_matrix_svm.png"
    },
    {
      "title": "Top Feature Importances",
      "path": "../results/figures/feature_importance_top15.png"
    },
    {
      "title": "Decision Tree Structure",
      "path": "../results/figures/decision_tree_structure.png"
    },
    {
      "title": "K-Means PCA Clusters",
      "path": "../results/figures/kmeans_pca_clusters.png"
    }
  ],
  "pca_accuracy": [
    {
      "Model": "Decision Tree",
      "Accuracy_Before_PCA": 0.71557,
      "Accuracy_After_PCA": 0.727875,
      "Accuracy_Change": 0.012305
    },
    {
      "Model": "SVM",
      "Accuracy_Before_PCA": 0.745386,
      "Accuracy_After_PCA": 0.743493,
      "Accuracy_Change": -0.001893
    }
  ],
  "pca_feature_counts": [
    {
      "Model": "Decision Tree",
      "Features_Before_PCA": 30,
      "Components_After_PCA": 17,
      "Reduced_By": 13
    },
    {
      "Model": "SVM",
      "Features_Before_PCA": 30,
      "Components_After_PCA": 17,
      "Reduced_By": 13
    }
  ],
  "pca_times": [
    {
      "Model": "Decision Tree",
      "Train_Time_Before_PCA_Seconds": 0.07593,
      "Train_Time_After_PCA_Seconds": 0.122187,
      "Train_Time_Change_Seconds": 0.046257,
      "Predict_Time_Before_PCA_Seconds": 0.021218,
      "Predict_Time_After_PCA_Seconds": 0.085611,
      "Predict_Time_Change_Seconds": 0.064394
    },
    {
      "Model": "SVM",
      "Train_Time_Before_PCA_Seconds": 2.499715,
      "Train_Time_After_PCA_Seconds": 1.828337,
      "Train_Time_Change_Seconds": -0.671379,
      "Predict_Time_Before_PCA_Seconds": 0.529085,
      "Predict_Time_After_PCA_Seconds": 0.459804,
      "Predict_Time_Change_Seconds": -0.069281
    }
  ]
};
