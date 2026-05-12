from __future__ import annotations

import os
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
DATA_FILE = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

PUBLIC_DATA_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/telco-customer-churn.csv",
]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FBFCFE",
    "axes.edgecolor": "#D0D7DE",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "font.size": 10,
    "axes.titleweight": "bold",
})

PALETTE = {
    "teal": "#1B998B",
    "coral": "#E85D75",
    "navy": "#2D3047",
    "gold": "#F4A261",
    "sky": "#3A86FF",
    "purple": "#8338EC",
    "green": "#6A994E",
    "gray": "#6C757D",
}

MODEL_COLORS = {
    "Decision Tree": PALETTE["gold"],
    "SVM": PALETTE["teal"],
}


def _save_figure(filename: str) -> None:
    """Save a figure with consistent spacing and resolution."""
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close()


def _to_dense(X):
    """Convert sparse matrix to dense because PCA requires dense input."""
    return X.toarray() if hasattr(X, "toarray") else X


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)


def try_download_dataset() -> None:
    """Try to download a public copy of the dataset if the local file is missing."""
    if DATA_FILE.exists():
        return

    print("Dataset not found locally. Trying to download a public copy...")

    for url in PUBLIC_DATA_URLS:
        try:
            urllib.request.urlretrieve(url, DATA_FILE)
            print(f"Downloaded dataset to {DATA_FILE}")
            return
        except Exception as exc:
            print(f"Download failed from {url}: {exc}")

    raise FileNotFoundError(
        f"Dataset file not found: {DATA_FILE}\n"
        "Please download it from Kaggle and place it in the data folder."
    )


def load_data() -> pd.DataFrame:
    ensure_directories()
    try_download_dataset()
    df = pd.read_csv(DATA_FILE)
    return df


def save_basic_eda(df: pd.DataFrame) -> None:
    """Save Stage 1 Data Understanding outputs and visualizations."""
    summary = []

    summary.append("Stage 1 - Data Understanding")
    summary.append("=" * 35)

    summary.append(f"\nRows: {df.shape[0]}")
    summary.append(f"Columns: {df.shape[1]}")

    summary.append("\nColumn names:\n" + str(df.columns.tolist()))
    summary.append("\nColumn types:\n" + str(df.dtypes))

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    summary.append("\nNumeric columns:\n" + str(numeric_cols))
    summary.append("\nCategorical columns:\n" + str(categorical_cols))

    summary.append("\nNumeric description:\n" + str(df.describe()))

    if categorical_cols:
        summary.append("\nCategorical description:\n" + str(df[categorical_cols].describe()))

    summary.append("\nMissing values:\n" + str(df.isna().sum()))

    if "TotalCharges" in df.columns:
        total_charges_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")
        summary.append(
            "\nTotalCharges values converted to missing because they are blank/non-numeric:\n"
            + str(total_charges_numeric.isna().sum())
        )

    if "Churn" in df.columns:
        summary.append("\nChurn distribution:\n" + str(df["Churn"].value_counts()))
        summary.append(
            "\nChurn distribution percentage:\n"
            + str(df["Churn"].value_counts(normalize=True) * 100)
        )

        churn_percent = df["Churn"].value_counts(normalize=True) * 100
        summary.append(
            "\nClass balance note:\n"
            + f"No Churn: {churn_percent.get('No', 0):.2f}% | "
            + f"Churn: {churn_percent.get('Yes', 0):.2f}%"
        )

    (RESULTS_DIR / "eda_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    if "Churn" in df.columns:
        counts = df["Churn"].value_counts().reindex(["No", "Yes"])
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        bars = ax.bar(
            counts.index,
            counts.values,
            color=[PALETTE["teal"], PALETTE["coral"]],
            edgecolor="white",
            linewidth=1.5,
        )

        ax.set_title("Customer Churn Distribution")
        ax.set_xlabel("Churn Status")
        ax.set_ylabel("Number of Customers")
        ax.set_ylim(0, counts.max() * 1.15)

        for bar, value in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + counts.max() * 0.02,
                f"{value}\n({value / counts.sum() * 100:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        _save_figure("churn_distribution.png")

    if {"Contract", "Churn"}.issubset(df.columns):
        contract_order = ["Month-to-month", "One year", "Two year"]
        ct = (
            pd.crosstab(df["Contract"], df["Churn"], normalize="index")
            .reindex(contract_order)[["No", "Yes"]]
        )

        ax = ct.plot(
            kind="bar",
            stacked=True,
            figsize=(8, 4.8),
            color=[PALETTE["teal"], PALETTE["coral"]],
            edgecolor="white",
            linewidth=1.2,
        )

        ax.set_title("Churn Proportion by Contract Type")
        ax.set_xlabel("Contract Type")
        ax.set_ylabel("Customer Proportion")
        ax.set_ylim(0, 1)
        ax.legend(title="Churn", loc="upper right")
        plt.xticks(rotation=20, ha="right")

        _save_figure("contract_vs_churn.png")

    if "tenure" in df.columns:
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.hist(
            df["tenure"],
            bins=28,
            color=PALETTE["sky"],
            edgecolor="white",
            linewidth=0.7,
        )

        ax.set_title("Customer Tenure Distribution")
        ax.set_xlabel("Tenure in Months")
        ax.set_ylabel("Frequency")

        _save_figure("tenure_distribution.png")

    if "MonthlyCharges" in df.columns:
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.hist(
            df["MonthlyCharges"],
            bins=28,
            color=PALETTE["purple"],
            edgecolor="white",
            linewidth=0.7,
        )

        ax.set_title("Monthly Charges Distribution")
        ax.set_xlabel("Monthly Charges")
        ax.set_ylabel("Frequency")

        _save_figure("monthly_charges_distribution.png")

    if {"tenure", "Churn"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        data_to_plot = [
            df.loc[df["Churn"] == label, "tenure"]
            for label in ["No", "Yes"]
        ]

        bp = ax.boxplot(
            data_to_plot,
            tick_labels=["No", "Yes"],
            patch_artist=True,
        )

        for patch, color in zip(bp["boxes"], [PALETTE["teal"], PALETTE["coral"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_title("Tenure by Churn Status")
        ax.set_xlabel("Churn Status")
        ax.set_ylabel("Tenure in Months")

        _save_figure("tenure_by_churn_boxplot.png")

    if {"PaymentMethod", "Churn"}.issubset(df.columns):
        pay = pd.crosstab(
            df["PaymentMethod"],
            df["Churn"],
            normalize="index",
        )[["No", "Yes"]]

        pay = pay.sort_values("Yes", ascending=False)

        ax = pay.plot(
            kind="barh",
            stacked=True,
            figsize=(8.5, 4.8),
            color=[PALETTE["teal"], PALETTE["coral"]],
            edgecolor="white",
            linewidth=1.0,
        )

        ax.set_title("Churn Proportion by Payment Method")
        ax.set_xlabel("Customer Proportion")
        ax.set_ylabel("Payment Method")
        ax.legend(title="Churn", loc="lower right")

        _save_figure("payment_method_vs_churn.png")


def clean_target_and_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean raw dataframe and split into X and y."""
    data = df.copy()

    if "customerID" in data.columns:
        data = data.drop(columns=["customerID"])

    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    if "Churn" not in data.columns:
        raise ValueError("The dataset must contain a target column named 'Churn'.")

    y = data["Churn"].map({"No": 0, "Yes": 1})
    X = data.drop(columns=["Churn"])

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def save_correlation_and_feature_selection(X: pd.DataFrame, y: pd.Series) -> None:
    """Create correlation matrix and SelectKBest feature-selection outputs."""
    analysis_data = X.copy()

    if "TotalCharges" in analysis_data.columns:
        analysis_data["TotalCharges"] = pd.to_numeric(
            analysis_data["TotalCharges"],
            errors="coerce",
        )

    encoded = pd.get_dummies(analysis_data, drop_first=True)
    encoded = encoded.replace([np.inf, -np.inf], np.nan)
    encoded = encoded.fillna(encoded.median(numeric_only=True))

    encoded_with_target = encoded.copy()
    encoded_with_target["Churn"] = y.values

    corr = encoded_with_target.corr(numeric_only=True)
    corr.to_csv(RESULTS_DIR / "correlation_matrix.csv")

    churn_corr = corr["Churn"].drop("Churn").sort_values(
        key=lambda col: col.abs(),
        ascending=False,
    )

    churn_corr.rename("Correlation_With_Churn").to_csv(
        RESULTS_DIR / "correlation_with_churn.csv"
    )

    top_features = churn_corr.abs().head(12).index.tolist()
    heatmap_cols = top_features + ["Churn"]
    heatmap_corr = encoded_with_target[heatmap_cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    im = ax.imshow(
        heatmap_corr.values,
        aspect="auto",
        cmap="RdYlBu_r",
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(range(len(heatmap_cols)))
    ax.set_yticks(range(len(heatmap_cols)))
    ax.set_xticklabels(heatmap_cols, rotation=55, ha="right", fontsize=8)
    ax.set_yticklabels(heatmap_cols, fontsize=8)
    ax.set_title("Correlation Heatmap: Top Features Related to Churn")

    for i in range(len(heatmap_cols)):
        for j in range(len(heatmap_cols)):
            ax.text(
                j,
                i,
                f"{heatmap_corr.values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color=PALETTE["navy"],
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_figure("correlation_matrix.png")

    k = min(15, encoded.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(encoded, y)

    selected = pd.DataFrame({
        "Feature": encoded.columns,
        "Score": selector.scores_,
        "P_Value": selector.pvalues_,
        "Selected": selector.get_support(),
    }).sort_values("Score", ascending=False)

    selected.to_csv(RESULTS_DIR / "selected_features.csv", index=False)

    top_selected = selected.head(15).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    colors = [
        PALETTE["sky"] if i % 2 == 0 else PALETTE["purple"]
        for i in range(len(top_selected))
    ]

    ax.barh(
        top_selected["Feature"],
        top_selected["Score"],
        color=colors,
        edgecolor="white",
    )

    ax.set_title("Top 15 Features Selected by ANOVA F-score")
    ax.set_xlabel("ANOVA F-score")

    _save_figure("selectkbest_top_features.png")


def compare_accuracy_before_after_pca(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    X_full: pd.DataFrame,
) -> None:
    """
    Compare models before and after PCA using accuracy, feature count,
    training time, and prediction time.
    """

    comparison_results = []
    feature_count_results = []
    time_results = []

    feature_counter = build_preprocessor(X_full)
    X_train_processed = feature_counter.fit_transform(X_train)
    features_before_pca = X_train_processed.shape[1]

    pca_variance = 0.95

    model_configs = [
        {
            "Model": "Decision Tree",
            "Classifier": DecisionTreeClassifier(
                max_depth=5,
                random_state=42,
                class_weight="balanced",
            ),
        },
        {
            "Model": "SVM",
            "Classifier": SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                random_state=42,
            ),
        },
    ]

    for config in model_configs:
        model_name = config["Model"]

        before_pca_model = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_full)),
                ("classifier", config["Classifier"]),
            ]
        )

        start_train = time.perf_counter()
        before_pca_model.fit(X_train, y_train)
        train_time_before = time.perf_counter() - start_train

        start_predict = time.perf_counter()
        before_pred = before_pca_model.predict(X_test)
        predict_time_before = time.perf_counter() - start_predict

        accuracy_before = accuracy_score(y_test, before_pred)

        after_pca_model = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_full)),
                ("to_dense", FunctionTransformer(_to_dense, accept_sparse=True)),
                ("pca", PCA(n_components=pca_variance, random_state=42)),
                ("classifier", config["Classifier"]),
            ]
        )

        start_train = time.perf_counter()
        after_pca_model.fit(X_train, y_train)
        train_time_after = time.perf_counter() - start_train

        start_predict = time.perf_counter()
        after_pred = after_pca_model.predict(X_test)
        predict_time_after = time.perf_counter() - start_predict

        accuracy_after = accuracy_score(y_test, after_pred)
        pca_components = after_pca_model.named_steps["pca"].n_components_

        comparison_results.append({
            "Model": model_name,
            "Accuracy_Before_PCA": accuracy_before,
            "Accuracy_After_PCA": accuracy_after,
            "Accuracy_Change": accuracy_after - accuracy_before,
        })

        feature_count_results.append({
            "Model": model_name,
            "Features_Before_PCA": features_before_pca,
            "Components_After_PCA": pca_components,
            "Reduced_By": features_before_pca - pca_components,
        })

        time_results.append({
            "Model": model_name,
            "Train_Time_Before_PCA_Seconds": train_time_before,
            "Train_Time_After_PCA_Seconds": train_time_after,
            "Train_Time_Change_Seconds": train_time_after - train_time_before,
            "Predict_Time_Before_PCA_Seconds": predict_time_before,
            "Predict_Time_After_PCA_Seconds": predict_time_after,
            "Predict_Time_Change_Seconds": predict_time_after - predict_time_before,
        })

    accuracy_df = pd.DataFrame(comparison_results)
    feature_count_df = pd.DataFrame(feature_count_results)
    time_df = pd.DataFrame(time_results)

    accuracy_df.to_csv(RESULTS_DIR / "pca_accuracy_comparison.csv", index=False)
    feature_count_df.to_csv(RESULTS_DIR / "pca_feature_count_comparison.csv", index=False)
    time_df.to_csv(RESULTS_DIR / "pca_time_comparison.csv", index=False)

    summary_lines = []
    summary_lines.append("PCA Before vs After Comparison")
    summary_lines.append("=" * 40)
    summary_lines.append("\nAccuracy Comparison:\n")
    summary_lines.append(accuracy_df.to_string(index=False))
    summary_lines.append("\n\nFeature Count Comparison:\n")
    summary_lines.append(feature_count_df.to_string(index=False))
    summary_lines.append("\n\nComputational Time Comparison:\n")
    summary_lines.append(time_df.to_string(index=False))
    summary_lines.append(
        "\n\nInterpretation:\n"
        "PCA was evaluated using accuracy, number of features/components, "
        "training time, and prediction time. The purpose is not only to improve "
        "accuracy, but also to reduce dimensionality and computational cost while "
        "preserving useful information."
    )

    (RESULTS_DIR / "pca_comparison_summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\nPCA Accuracy Comparison:")
    print(accuracy_df)
    print("\nPCA Feature Count Comparison:")
    print(feature_count_df)
    print("\nPCA Time Comparison:")
    print(time_df)

    plot_accuracy = accuracy_df.set_index("Model")[[
        "Accuracy_Before_PCA", "Accuracy_After_PCA"
    ]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plot_accuracy.plot(kind="bar", ax=ax, color=[PALETTE["gray"], PALETTE["sky"]], edgecolor="white")
    ax.set_title("Accuracy Before and After PCA")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(["Before PCA", "After PCA"], title="Feature Space")
    plt.xticks(rotation=0)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9, padding=3)
    _save_figure("pca_accuracy_comparison.png")

    count_plot_df = feature_count_df.set_index("Model")[[
        "Features_Before_PCA", "Components_After_PCA"
    ]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    count_plot_df.plot(kind="bar", ax=ax, color=[PALETTE["gray"], PALETTE["purple"]], edgecolor="white")
    ax.set_title("Number of Features Before and After PCA")
    ax.set_xlabel("Model")
    ax.set_ylabel("Number of Features / Components")
    ax.legend(["Before PCA", "After PCA"], title="Feature Space")
    plt.xticks(rotation=0)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=9, padding=3)
    _save_figure("pca_feature_count_comparison.png")

    train_time_plot = time_df.set_index("Model")[[
        "Train_Time_Before_PCA_Seconds", "Train_Time_After_PCA_Seconds"
    ]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    train_time_plot.plot(kind="bar", ax=ax, color=[PALETTE["gray"], PALETTE["green"]], edgecolor="white")
    ax.set_title("Training Time Before and After PCA")
    ax.set_xlabel("Model")
    ax.set_ylabel("Seconds")
    ax.legend(["Before PCA", "After PCA"], title="Feature Space")
    plt.xticks(rotation=0)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9, padding=3)
    _save_figure("pca_training_time_comparison.png")

    predict_time_plot = time_df.set_index("Model")[[
        "Predict_Time_Before_PCA_Seconds", "Predict_Time_After_PCA_Seconds"
    ]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    predict_time_plot.plot(kind="bar", ax=ax, color=[PALETTE["gray"], PALETTE["gold"]], edgecolor="white")
    ax.set_title("Prediction Time Before and After PCA")
    ax.set_xlabel("Model")
    ax.set_ylabel("Seconds")
    ax.legend(["Before PCA", "After PCA"], title="Feature Space")
    plt.xticks(rotation=0)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9, padding=3)
    _save_figure("pca_prediction_time_comparison.png")


def evaluate_model(
    name: str,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    y_pred = model.predict(X_test)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1_Score": f1_score(y_test, y_pred, zero_division=0),
    }

    report = classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"],
    )

    (RESULTS_DIR / f"{name.lower().replace(' ', '_')}_classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_tick_labels=["No", "Yes"],
    )

    disp.plot(values_format="d", cmap="YlGnBu", colorbar=False)
    plt.title(f"Confusion Matrix - {name}")
    plt.grid(False)

    _save_figure(f"confusion_matrix_{name.lower().replace(' ', '_')}.png")

    return metrics


def save_feature_importance(tree_pipeline: Pipeline, X: pd.DataFrame) -> None:
    """Save Decision Tree feature importance values and a tree visualization."""
    preprocessor = tree_pipeline.named_steps["preprocessor"]
    model = tree_pipeline.named_steps["classifier"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = np.array([
            f"feature_{i}"
            for i in range(len(model.feature_importances_))
        ])

    importances = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    importances.to_csv(
        RESULTS_DIR / "decision_tree_feature_importance.csv",
        index=False,
    )

    top = importances.head(15).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 6.5))

    ax.barh(
        top["Feature"],
        top["Importance"],
        color=PALETTE["gold"],
        edgecolor="white",
    )

    ax.set_title("Decision Tree: Top 15 Feature Importances")
    ax.set_xlabel("Importance")

    _save_figure("feature_importance_top15.png")

    plt.figure(figsize=(21, 10.5))

    plot_tree(
        model,
        max_depth=3,
        feature_names=feature_names,
        class_names=["No", "Yes"],
        filled=True,
        rounded=True,
        fontsize=8,
    )

    plt.title("Decision Tree Structure - First Levels")

    _save_figure("decision_tree_structure.png")


def run_kmeans(preprocessor: ColumnTransformer, X_train: pd.DataFrame) -> None:
    """Run K-Means clustering and save a PCA visualization."""
    X_train_processed = preprocessor.fit_transform(X_train)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_train_processed)

    sil = silhouette_score(
        X_train_processed,
        clusters,
        sample_size=min(1000, X_train_processed.shape[0]),
        random_state=42,
    )

    (RESULTS_DIR / "kmeans_summary.txt").write_text(
        f"K-Means clustering with k=3\nSilhouette score: {sil:.4f}\n",
        encoding="utf-8",
    )

    pca = PCA(n_components=2, random_state=42)

    reduced = pca.fit_transform(
        X_train_processed.toarray()
        if hasattr(X_train_processed, "toarray")
        else X_train_processed
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    cluster_colors = [
        PALETTE["teal"],
        PALETTE["coral"],
        PALETTE["gold"],
    ]

    for cluster_id in sorted(np.unique(clusters)):
        mask = clusters == cluster_id

        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            s=16,
            alpha=0.75,
            color=cluster_colors[int(cluster_id) % len(cluster_colors)],
            label=f"Cluster {cluster_id}",
        )

    ax.set_title("K-Means Customer Segments - PCA 2D View")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(loc="best")

    _save_figure("kmeans_pca_clusters.png")


def save_model_comparison_charts(metrics_df: pd.DataFrame) -> None:
    """Save bar and radar charts comparing model performance."""
    metric_cols = ["Accuracy", "Precision", "Recall", "F1_Score"]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    y_pos = np.arange(len(metric_cols))
    bar_height = 0.34

    for idx, (_, row) in enumerate(metrics_df.iterrows()):
        offset = (idx - 0.5) * bar_height
        values = [float(row[col]) for col in metric_cols]

        bars = ax.barh(
            y_pos + offset,
            values,
            height=bar_height,
            label=row["Model"],
            color=MODEL_COLORS.get(row["Model"], PALETTE["gray"]),
            edgecolor="white",
        )

        for bar, value in zip(bars, values):
            ax.text(
                value + 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                fontsize=9,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([col.replace("_", " ") for col in metric_cols])
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend(loc="lower right")

    _save_figure("model_comparison_chart.png")

    labels = [col.replace("_", " ") for col in metric_cols]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7.5, 7.5))
    ax = plt.subplot(111, polar=True)

    for _, row in metrics_df.iterrows():
        values = [float(row[col]) for col in metric_cols]
        values += values[:1]

        color = MODEL_COLORS.get(row["Model"], PALETTE["gray"])

        ax.plot(
            angles,
            values,
            linewidth=2.5,
            label=row["Model"],
            color=color,
        )

        ax.fill(
            angles,
            values,
            alpha=0.16,
            color=color,
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Radar Chart - Model Metrics")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))

    _save_figure("radar_chart.png")


def export_dashboard_data(metrics_df: pd.DataFrame) -> None:
    """Export latest run data for the HTML/CSS/JavaScript dashboard."""
    pca_accuracy_path = RESULTS_DIR / "pca_accuracy_comparison.csv"
    pca_feature_path = RESULTS_DIR / "pca_feature_count_comparison.csv"
    pca_time_path = RESULTS_DIR / "pca_time_comparison.csv"

    dashboard_data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics_df.round(6).to_dict(orient="records"),
        "pca_accuracy": pd.read_csv(pca_accuracy_path).round(6).to_dict(orient="records") if pca_accuracy_path.exists() else [],
        "pca_feature_counts": pd.read_csv(pca_feature_path).to_dict(orient="records") if pca_feature_path.exists() else [],
        "pca_times": pd.read_csv(pca_time_path).round(6).to_dict(orient="records") if pca_time_path.exists() else [],
        "figures": [
            {"title": "Churn Distribution", "path": "../results/figures/churn_distribution.png"},
            {"title": "Contract vs Churn", "path": "../results/figures/contract_vs_churn.png"},
            {"title": "Tenure Distribution", "path": "../results/figures/tenure_distribution.png"},
            {"title": "Monthly Charges Distribution", "path": "../results/figures/monthly_charges_distribution.png"},
            {"title": "Tenure by Churn", "path": "../results/figures/tenure_by_churn_boxplot.png"},
            {"title": "Payment Method vs Churn", "path": "../results/figures/payment_method_vs_churn.png"},
            {"title": "Correlation Matrix", "path": "../results/figures/correlation_matrix.png"},
            {"title": "SelectKBest Top Features", "path": "../results/figures/selectkbest_top_features.png"},
            {"title": "PCA Accuracy Comparison", "path": "../results/figures/pca_accuracy_comparison.png"},
            {"title": "PCA Feature Count Comparison", "path": "../results/figures/pca_feature_count_comparison.png"},
            {"title": "PCA Training Time Comparison", "path": "../results/figures/pca_training_time_comparison.png"},
            {"title": "PCA Prediction Time Comparison", "path": "../results/figures/pca_prediction_time_comparison.png"},
            {"title": "Model Comparison Chart", "path": "../results/figures/model_comparison_chart.png"},
            {"title": "Radar Chart", "path": "../results/figures/radar_chart.png"},
            {"title": "Decision Tree Confusion Matrix", "path": "../results/figures/confusion_matrix_decision_tree.png"},
            {"title": "SVM Confusion Matrix", "path": "../results/figures/confusion_matrix_svm.png"},
            {"title": "Top Feature Importances", "path": "../results/figures/feature_importance_top15.png"},
            {"title": "Decision Tree Structure", "path": "../results/figures/decision_tree_structure.png"},
            {"title": "K-Means PCA Clusters", "path": "../results/figures/kmeans_pca_clusters.png"},
        ],
    }

    (RESULTS_DIR / "dashboard_data.json").write_text(
        json.dumps(dashboard_data, indent=2),
        encoding="utf-8",
    )

    (DASHBOARD_DIR / "latest_results.js").write_text(
        "window.CHURN_DASHBOARD_DATA = "
        + json.dumps(dashboard_data, indent=2)
        + ";\n",
        encoding="utf-8",
    )


def main() -> None:
    ensure_directories()

    df = load_data()

    # Stage 1 - Data Understanding
    save_basic_eda(df)

    # Stage 2 - Data Cleaning
    X, y = clean_target_and_features(df)

    # Stage 4 - Feature Analysis
    save_correlation_and_feature_selection(X, y)

    # Stage 3 - Sampling / Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    # Stage 4 - PCA Before vs After Comparison
    compare_accuracy_before_after_pca(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_full=X,
    )

    preprocessor = build_preprocessor(X)

    # Stage 5 - Decision Tree Classifier
    decision_tree = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=5,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # Stage 5 - SVM Classifier
    svm = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    decision_tree.fit(X_train, y_train)
    svm.fit(X_train, y_train)

    # Stage 6 - Model Evaluation
    metrics = []
    metrics.append(evaluate_model("Decision Tree", decision_tree, X_test, y_test))
    metrics.append(evaluate_model("SVM", svm, X_test, y_test))

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    print("\nMain Model Comparison:")
    print(metrics_df)

    save_model_comparison_charts(metrics_df)

    save_feature_importance(decision_tree, X)

    run_kmeans(build_preprocessor(X), X_train)

    export_dashboard_data(metrics_df)

    print("Project finished successfully.")
    print(f"Results saved in: {RESULTS_DIR}")
    print(f"Dashboard updated: {DASHBOARD_DIR / 'index.html'}")


if __name__ == "__main__":
    main()