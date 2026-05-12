function pct(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function num(value, digits = 3) {
  return Number(value).toFixed(digits);
}

function signedPct(value) {
  const n = Number(value);
  return `${n >= 0 ? "+" : ""}${pct(n)}`;
}

function getBestModel(metrics) {
  return [...metrics].sort((a, b) => Number(b.F1_Score) - Number(a.F1_Score))[0];
}

const categoryNotes = {
  "Data Understanding": "Explore the dataset shape and customer behavior patterns before modeling.",
  "Feature Insight": "These charts explain which customer attributes are visually related to churn.",
  "Feature Analysis": "This group supports dimensionality and feature relevance discussion.",
  "PCA Impact": "Use these charts to explain how PCA affected accuracy, feature count, training time, and prediction time.",
  "Model Evaluation": "Open these charts to compare Decision Tree and SVM after training.",
  "Decision Tree Interpretation": "These visuals explain how the Decision Tree makes decisions and which features matter most.",
  "Clustering": "This chart shows unsupervised customer groups using K-Means with PCA for visualization.",
  "All Charts": "All generated charts are listed here. Open only the ones you want to discuss."
};

const figureDescriptions = {
  "Churn Distribution": {
    group: "Data Understanding",
    text: "Shows the class distribution between customers who stayed and customers who churned. This justifies using stratified sampling and balanced class weights."
  },
  "Contract vs Churn": {
    group: "Feature Insight",
    text: "Shows churn proportions by contract type. Month-to-month contracts have a higher churn proportion, making Contract an important feature."
  },
  "Tenure Distribution": {
    group: "Feature Insight",
    text: "Shows how long customers have stayed with the company. Tenure is useful because churn risk is often higher for newer customers."
  },
  "Monthly Charges Distribution": {
    group: "Feature Insight",
    text: "Shows the spread of monthly charges. This helps understand whether service cost may contribute to churn behavior."
  },
  "Tenure by Churn": {
    group: "Feature Insight",
    text: "A boxplot comparing tenure for churn and non-churn customers. It helps show that churn customers often have lower tenure."
  },
  "Payment Method vs Churn": {
    group: "Feature Insight",
    text: "Shows churn proportion across payment methods. It highlights payment categories that may be associated with higher churn."
  },
  "Correlation Matrix": {
    group: "Feature Analysis",
    text: "Shows correlations after encoding the data. It helps identify features that are strongly related to the target variable."
  },
  "SelectKBest Top Features": {
    group: "Feature Analysis",
    text: "Ranks features using ANOVA F-score. This supports the feature selection and dimensionality analysis part of the project."
  },
  "PCA Accuracy Comparison": {
    group: "PCA Impact",
    text: "Compares accuracy before and after PCA. The small changes indicate that PCA preserved most predictive information."
  },
  "PCA Feature Count Comparison": {
    group: "PCA Impact",
    text: "Shows the direct PCA effect: reducing the preprocessed feature space from many features into fewer principal components."
  },
  "PCA Training Time Comparison": {
    group: "PCA Impact",
    text: "Compares training time before and after PCA. SVM benefits more because it is more sensitive to feature dimensionality."
  },
  "PCA Prediction Time Comparison": {
    group: "PCA Impact",
    text: "Compares prediction time before and after PCA. It measures whether reduced dimensionality improves inference speed."
  },
  "Model Comparison Chart": {
    group: "Model Evaluation",
    text: "Compares Decision Tree and SVM across accuracy, precision, recall, and F1-score. It summarizes the main model results."
  },
  "Radar Chart": {
    group: "Model Evaluation",
    text: "Gives a quick visual comparison of the four classification metrics. A larger shape indicates better overall performance."
  },
  "Decision Tree Confusion Matrix": {
    group: "Model Evaluation",
    text: "Shows correct and incorrect predictions for Decision Tree. It helps explain precision and recall behavior."
  },
  "SVM Confusion Matrix": {
    group: "Model Evaluation",
    text: "Shows correct and incorrect predictions for SVM. It supports comparing SVM with Decision Tree."
  },
  "Top Feature Importances": {
    group: "Decision Tree Interpretation",
    text: "Shows the most important features according to the trained Decision Tree, making the model more interpretable."
  },
  "Decision Tree Structure": {
    group: "Decision Tree Interpretation",
    text: "Shows the first levels of the tree and the main split rules used to classify customers."
  },
  "K-Means PCA Clusters": {
    group: "Clustering",
    text: "Shows customer segments produced by K-Means, projected into two PCA components so the clusters can be visualized."
  }
};

const preferredOrder = [
  "Churn Distribution",
  "Tenure Distribution",
  "Monthly Charges Distribution",
  "Contract vs Churn",
  "Payment Method vs Churn",
  "Tenure by Churn",
  "Correlation Matrix",
  "SelectKBest Top Features",
  "Top Feature Importances",
  "Decision Tree Structure",
  "PCA Accuracy Comparison",
  "PCA Feature Count Comparison",
  "PCA Training Time Comparison",
  "PCA Prediction Time Comparison",
  "Model Comparison Chart",
  "Radar Chart",
  "Decision Tree Confusion Matrix",
  "SVM Confusion Matrix",
  "K-Means PCA Clusters"
];

const categories = [
  "Data Understanding",
  "Feature Insight",
  "Feature Analysis",
  "PCA Impact",
  "Model Evaluation",
  "Decision Tree Interpretation",
  "Clustering",
  "All Charts"
];

function sortFigures(figures) {
  const rank = new Map(preferredOrder.map((title, index) => [title, index]));
  return [...figures].sort((a, b) => (rank.get(a.title) ?? 999) - (rank.get(b.title) ?? 999));
}

function figureGroup(title) {
  return (figureDescriptions[title] || { group: "All Charts" }).group;
}

function renderMetricCards(data, best) {
  const pcaFeature = (data.pca_feature_counts || [])[0];
  const svmTime = (data.pca_times || []).find(row => row.Model === "SVM");
  const svmAccuracy = (data.pca_accuracy || []).find(row => row.Model === "SVM");

  document.getElementById("metricCards").innerHTML = `
    <article class="metric-card"><span>Best Model</span><strong>${best.Model}</strong><small>Chosen by F1-score</small></article>
    <article class="metric-card"><span>Best Accuracy</span><strong>${pct(best.Accuracy)}</strong><small>${best.Model} on test set</small></article>
    <article class="metric-card"><span>PCA Reduction</span><strong>${pcaFeature ? `${pcaFeature.Features_Before_PCA} → ${pcaFeature.Components_After_PCA}` : "-"}</strong><small>Features to components</small></article>
    <article class="metric-card"><span>SVM PCA Accuracy</span><strong>${svmAccuracy ? pct(svmAccuracy.Accuracy_After_PCA) : "-"}</strong><small>${svmTime ? `${Math.abs(Number(svmTime.Train_Time_Change_Seconds)).toFixed(3)}s train-time change` : "After PCA"}</small></article>
  `;
}

function renderTables(data) {
  document.getElementById("resultsTable").innerHTML = (data.metrics || []).map(row => `
    <tr><td><strong>${row.Model}</strong></td><td>${pct(row.Accuracy)}</td><td>${pct(row.Precision)}</td><td>${pct(row.Recall)}</td><td>${pct(row.F1_Score)}</td></tr>
  `).join("");

  document.getElementById("pcaAccuracyTable").innerHTML = (data.pca_accuracy || []).map(row => `
    <tr><td><strong>${row.Model}</strong></td><td>${pct(row.Accuracy_Before_PCA)}</td><td>${pct(row.Accuracy_After_PCA)}</td><td>${signedPct(row.Accuracy_Change)}</td></tr>
  `).join("");

  document.getElementById("pcaFeatureTable").innerHTML = (data.pca_feature_counts || []).map(row => `
    <tr><td><strong>${row.Model}</strong></td><td>${row.Features_Before_PCA}</td><td>${row.Components_After_PCA}</td><td>${row.Reduced_By}</td></tr>
  `).join("");

  document.getElementById("pcaTimeTable").innerHTML = (data.pca_times || []).map(row => `
    <tr><td><strong>${row.Model}</strong></td><td>${num(row.Train_Time_Before_PCA_Seconds)}s</td><td>${num(row.Train_Time_After_PCA_Seconds)}s</td><td>${num(row.Predict_Time_Before_PCA_Seconds)}s</td><td>${num(row.Predict_Time_After_PCA_Seconds)}s</td></tr>
  `).join("");
}

function renderPcaSummary(data) {
  const feature = (data.pca_feature_counts || [])[0];
  const dt = (data.pca_accuracy || []).find(row => row.Model === "Decision Tree");
  const svmTime = (data.pca_times || []).find(row => row.Model === "SVM");

  document.getElementById("pcaSummary").innerHTML = `
    <article class="summary-chip"><span>Feature Space</span><strong>${feature ? `${feature.Features_Before_PCA} → ${feature.Components_After_PCA}` : "-"}</strong></article>
    <article class="summary-chip"><span>Decision Tree Accuracy</span><strong>${dt ? `${pct(dt.Accuracy_Before_PCA)} → ${pct(dt.Accuracy_After_PCA)}` : "-"}</strong></article>
    <article class="summary-chip"><span>SVM Train Time</span><strong>${svmTime ? `${num(svmTime.Train_Time_Before_PCA_Seconds)}s → ${num(svmTime.Train_Time_After_PCA_Seconds)}s` : "-"}</strong></article>
  `;
}

function renderTabs(data) {
  const tabBar = document.getElementById("tabBar");
  tabBar.innerHTML = categories.map((cat, index) => `
    <button class="tab-btn ${index === 0 ? "active" : ""}" data-category="${cat}">${cat}</button>
  `).join("");

  tabBar.querySelectorAll(".tab-btn").forEach(button => {
    button.addEventListener("click", () => {
      tabBar.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
      button.classList.add("active");
      renderGallery(data, button.dataset.category);
    });
  });

  document.querySelectorAll("[data-open-category]").forEach(button => {
    button.addEventListener("click", () => {
      const category = button.dataset.openCategory;
      const targetButton = [...tabBar.querySelectorAll(".tab-btn")].find(btn => btn.dataset.category === category);
      if (targetButton) targetButton.click();
      document.getElementById("visuals").scrollIntoView({ behavior: "smooth" });
    });
  });

  document.querySelectorAll(".jump-btn").forEach(button => {
    button.addEventListener("click", () => {
      document.getElementById(button.dataset.target).scrollIntoView({ behavior: "smooth" });
    });
  });

  renderGallery(data, categories[0]);
}

function renderGallery(data, category) {
  const figures = sortFigures(data.figures || []).filter(fig => {
    return category === "All Charts" || figureGroup(fig.title) === category;
  });

  document.getElementById("activeCategoryNote").textContent = categoryNotes[category] || "Open a chart to view it.";

  document.getElementById("gallery").innerHTML = figures.map((fig, index) => {
    const meta = figureDescriptions[fig.title] || { group: "Result", text: "Generated by the Python analysis pipeline." };
    return `
      <article class="chart-card" data-title="${fig.title}">
        <div class="chart-top">
          <div>
            <span class="pill dark">${meta.group}</span>
            <h3 class="chart-title">${fig.title}</h3>
            <p class="chart-meta">${meta.text}</p>
          </div>
          <button class="open-chart-btn" type="button">${index === 0 ? "Open Chart" : "View"}</button>
        </div>
        <div class="chart-body">
          <img src="${fig.path}" alt="${fig.title}" onerror="this.closest('.chart-card').style.display='none'" />
          <p class="caption">${meta.text}</p>
        </div>
      </article>
    `;
  }).join("");

  document.querySelectorAll(".chart-card .open-chart-btn").forEach(button => {
    button.addEventListener("click", () => {
      const card = button.closest(".chart-card");
      card.classList.toggle("open");
      button.textContent = card.classList.contains("open") ? "Hide" : "View";
    });
  });
}

function renderDashboard(data) {
  const metrics = data.metrics || [];
  if (!metrics.length) throw new Error("No metric data found.");

  const best = getBestModel(metrics);
  document.getElementById("heroBestModel").textContent = best.Model;
  document.getElementById("heroBestText").textContent = `${best.Model} achieved ${pct(best.Accuracy)} accuracy and ${pct(best.F1_Score)} F1-score on the test set.`;

  renderMetricCards(data, best);
  renderTables(data);
  renderPcaSummary(data);

  document.getElementById("bestModel").innerHTML = `<strong>${best.Model}</strong> is the preferred predictive model in this run based on F1-score. Decision Tree remains useful for interpretation because it provides tree structure and feature importance.`;

  renderTabs(data);
}

try {
  if (window.CHURN_DASHBOARD_DATA && window.CHURN_DASHBOARD_DATA.metrics) {
    renderDashboard(window.CHURN_DASHBOARD_DATA);
  } else {
    throw new Error("Dashboard data not found.");
  }
} catch (error) {
  document.body.innerHTML = `<div class="error-box"><strong>Dashboard data could not be loaded.</strong><br>${error.message}<br>Run <code>python src/churn_prediction.py</code> first.</div>`;
}
