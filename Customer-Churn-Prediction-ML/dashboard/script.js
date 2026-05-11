function pct(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function getBestModel(metrics) {
  return [...metrics].sort((a, b) => Number(b.F1_Score) - Number(a.F1_Score))[0];
}

function renderDashboard(data) {
  document.getElementById("lastRun").textContent = data.generated_at || "Unknown";

  const best = getBestModel(data.metrics);
  const metricCards = document.getElementById("metricCards");
  metricCards.innerHTML = `
    <article class="card"><span>Best Model</span><strong>${best.Model}</strong></article>
    <article class="card"><span>Accuracy</span><strong>${pct(best.Accuracy)}</strong></article>
    <article class="card"><span>Recall</span><strong>${pct(best.Recall)}</strong></article>
    <article class="card"><span>F1 Score</span><strong>${pct(best.F1_Score)}</strong></article>
  `;

  const table = document.getElementById("resultsTable");
  table.innerHTML = data.metrics.map(row => `
    <tr>
      <td><strong>${row.Model}</strong></td>
      <td>${pct(row.Accuracy)}</td>
      <td>${pct(row.Precision)}</td>
      <td>${pct(row.Recall)}</td>
      <td>${pct(row.F1_Score)}</td>
    </tr>
  `).join("");

  document.getElementById("bestModel").innerHTML = `
    <strong>${best.Model}</strong> is the best model in this run based on F1-score.
    It achieved <strong>${pct(best.Accuracy)}</strong> accuracy and <strong>${pct(best.F1_Score)}</strong> F1-score.
  `;

  const gallery = document.getElementById("gallery");
  gallery.innerHTML = data.figures.map(fig => `
    <article class="figure-card">
      <h3>${fig.title}</h3>
      <img src="${fig.path}" alt="${fig.title}" onerror="this.parentElement.style.display='none'" />
    </article>
  `).join("");
}

if (window.CHURN_DASHBOARD_DATA) {
  renderDashboard(window.CHURN_DASHBOARD_DATA);
} else {
  document.body.innerHTML = `
    <main class="page">
      <div class="error">
        No dashboard data found. Run <strong>python src/churn_prediction.py</strong> first.
      </div>
    </main>
  `;
}
