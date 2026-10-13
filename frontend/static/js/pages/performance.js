import { $, api, bindTabs, tabs, esc, fmt, icon, tip, errorState, skeleton, cssVar } from "../core.js";
import { plot, themed, series } from "../charts.js";

const METRICS = [
  ["accuracy", "Accuracy", "Share of correct predictions. Misleading alone under class imbalance."],
  ["precision", "Precision", "Of the tools flagged as risky, the share that actually failed."],
  ["recall", "Recall", "Of the tools that actually failed, the share the model caught (missed failures are the costly error)."],
  ["f1", "F1", "Harmonic mean of precision and recall at the tuned threshold."],
  ["roc_auc", "ROC-AUC", "Threshold-free ranking quality: P(score of a failing tool > score of a successful one)."],
  ["pr_auc", "PR-AUC", "Average precision — primary selection metric; the no-skill level equals the failure base rate."],
];

export default async function performance(view) {
  view.innerHTML = `<div class="page-head"><div><div class="eyebrow">Benchmarking & comparative analysis</div><h1 class="mt8">Model Performance</h1>
    <p>Every number on this page is read from the latest pipeline run — 14 models across the experiment matrix and benchmark, evaluated on a leakage-safe grouped split.</p></div></div>
    ${skeleton(160)}<div class="mt16">${skeleton(360)}</div>`;
  let d;
  try { d = await api("/model-metrics"); } catch (e) { view.innerHTML = errorState(e); return; }
  const ev = d.evaluation, sel = d.results.find((r) => r.id === d.selection.selected_id), meta = d.model;
  const ci = ev.bootstrap_ci_95;
  let split = "test";

  view.innerHTML = `
  <div class="page-head"><div><div class="eyebrow">Benchmarking & comparative analysis</div><h1 class="mt8">Model Performance</h1>
    <p>Every number on this page is read from the latest pipeline run — ${d.results.length} models across the experiment matrix and benchmark, evaluated on a leakage-safe grouped split.</p></div></div>

  <div class="panel headline">
    <div class="eyebrow">${icon("trophy")} Selected production model</div>
    <h2 class="mt8" style="font-size:24px">${esc(sel.name)}</h2>
    <p class="text-2 mt8" style="max-width:900px">${esc(d.selection.reason)}</p>
    <p class="muted mt8" style="font-size:12.5px">Selection rule: ${esc(d.selection.rule)} · threshold ${fmt(meta.threshold)} · ${meta.n_features} features · MCC ${fmt(sel.test.mcc)} ${tip("Matthews correlation coefficient: balanced summary of the confusion matrix (−1…1).")}</p>
  </div>

  <div class="grid g6 mt16">${METRICS.map(([k, l, h]) => `<div class="panel kpi"><div class="label">${l}${tip(h)}</div><div class="value">${fmt(ev.test_metrics[k])}</div>
    <div class="bar"><span style="width:${ev.test_metrics[k] * 100}%"></span></div>
    <div class="foot">${ci[k] ? `95% CI ${fmt(ci[k].low, 3)}–${fmt(ci[k].high, 3)}` : `train ${fmt(sel.train[k])} · val ${fmt(sel.val[k])}`}</div></div>`).join("")}</div>

  <div class="panel mt24">
    <div class="panel-head"><div><h3>Model comparison table</h3><div class="sub">Sorted by PR-AUC on the selected split · ★ = selected production model · bars show relative score</div></div>
      <div class="seg" id="split-seg">${["train", "val", "test"].map((s) => `<button data-s="${s}" class="${s === split ? "on" : ""}">${s === "val" ? "Validation" : s[0].toUpperCase() + s.slice(1)}</button>`).join("")}</div></div>
    <div class="table-wrap" id="cmp"></div>
    <p class="muted mt12" style="font-size:12px">Train scores of tree ensembles are optimistic by construction; selection uses validation only and the test split is reported once.</p>
  </div>

  <div class="panel mt24">
    <div class="panel-head"><div><h3>Charts</h3><div class="sub">Test split · click legend items to toggle models</div></div>
      ${tabs("ch", [["curves", "ROC & PR curves"], ["cmp", "Model comparison"], ["diag", "Diagnostics"]].map(([k, l]) => [k, l, ""]), { bar: false }).head}</div>
    <div class="tab-pane" data-pane="ch:curves"><div class="grid g2">
      <div><h3 style="font-size:14px">ROC curves</h3><div class="chart lg" id="roc"></div></div>
      <div><h3 style="font-size:14px">Precision–recall curves <span class="muted" style="font-weight:500">· dashed = no-skill</span></h3><div class="chart lg" id="pr"></div></div></div></div>
    <div class="tab-pane" data-pane="ch:cmp" hidden><div class="grid g2">
      <div><h3 style="font-size:14px">F1 (validation-tuned threshold)</h3><div class="chart" id="f1c"></div></div>
      <div><h3 style="font-size:14px">ROC-AUC vs PR-AUC</h3><div class="chart" id="aucc"></div></div></div></div>
    <div class="tab-pane" data-pane="ch:diag" hidden><div class="grid g3">
      <div><h3 style="font-size:14px">Confusion matrix</h3><div class="chart" id="cm"></div></div>
      <div><h3 style="font-size:14px">Reliability (calibration)</h3><div class="chart" id="cal"></div></div>
      <div><h3 style="font-size:14px">Probability distribution <span class="muted" style="font-weight:500">· line = threshold</span></h3><div class="chart" id="pd"></div></div></div></div>
  </div>
  <div class="grid mt24">
    <div class="panel"><div class="panel-head"><div><h3>${icon("flask")} Statistical significance</h3><div class="sub">Paired bootstrap (1,000 resamples) of test PR-AUC: selected model − baseline</div></div></div>
      <div class="table-wrap"><table class="tbl"><thead><tr><th>Baseline</th><th class="num">ΔPR-AUC</th><th class="num">95% CI</th><th class="num">p (one-sided)</th><th>Verdict</th></tr></thead><tbody>
      ${ev.paired_tests.sort((a, b) => b.pr_auc.mean_delta - a.pr_auc.mean_delta).map((c) => { const sig = c.pr_auc.ci_low > 0; return `<tr><td><strong>${esc(c.vs)}</strong> <span class="muted">${esc(c.vs_name)}</span></td>
        <td class="num" style="color:${c.pr_auc.mean_delta > 0 ? "var(--good)" : "var(--critical)"}">${c.pr_auc.mean_delta > 0 ? "+" : ""}${fmt(c.pr_auc.mean_delta)}</td>
        <td class="num">[${fmt(c.pr_auc.ci_low)}, ${fmt(c.pr_auc.ci_high)}]</td><td class="num">${fmt(c.pr_auc.p_value_one_sided)}</td>
        <td>${sig ? `<span class="risk risk-LOW">${icon("check")}better</span>` : `<span class="chip">n.s.</span>`}</td></tr>`; }).join("")}
      </tbody></table></div>
      <p class="muted mt12" style="font-size:12px">"better" = 95% CI of the difference excludes 0. Non-significant differences are reported as such — several fused-feature models are statistically tied.</p></div>
    <div class="panel"><div class="panel-head"><div><h3>${icon("layers")} Ablation study</h3><div class="sub">XGBoost, 5-fold grouped CV on train+val · does feature fusion help?</div></div></div>
      <div class="chart" id="abl"></div>
      <p class="text-2 mt8">All features vs best single family (${esc(d.ablation.rows.find((r) => r.set === d.ablation.best_single_set).label)}): <b>${d.ablation.fusion_gain_pr_auc_vs_best_single > 0 ? "+" : ""}${fmt(d.ablation.fusion_gain_pr_auc_vs_best_single)}</b> CV PR-AUC.</p></div>
  </div>`;

  bindTabs(view, "ch");
  const cmp = () => {
    const rows = [...d.results].sort((a, b) => b[split].pr_auc - a[split].pr_auc);
    const max = Object.fromEntries(METRICS.map(([k]) => [k, Math.max(...rows.map((r) => r[split][k]))]));
    $("#cmp", view).innerHTML = `<table class="tbl"><thead><tr><th>ID</th><th>Model</th><th>Features</th>${METRICS.map(([k, l]) => `<th class="num">${l}</th>`).join("")}<th class="num">Train s</th></tr></thead><tbody>
      ${rows.map((r) => `<tr class="${r.id === sel.id ? "best" : ""}"><td><span class="chip">${r.id}</span></td><td><strong>${r.id === sel.id ? "★ " : ""}${esc(r.model)}</strong></td><td class="muted" style="font-size:12px">${esc(r.feature_set)}</td>
        ${METRICS.map(([k]) => `<td class="num metric-cell" style="${r[split][k] === max[k] ? "color:var(--text);font-weight:700" : ""}">${fmt(r[split][k])}<span class="mbar" style="width:${r[split][k] * 100}%"></span></td>`).join("")}
        <td class="num">${fmt(r.train_seconds, 2)}</td></tr>`).join("")}</tbody></table>`;
  };
  cmp();
  view.querySelectorAll("#split-seg button").forEach((b) => b.addEventListener("click", () => {
    split = b.dataset.s; view.querySelectorAll("#split-seg button").forEach((x) => x.classList.toggle("on", x === b)); cmp();
  }));

  themed(() => {
    const S = series();
    const ids = [sel.id, ...["B2", "B3", "B6", "E2", "E1", "E4", "B1"].filter((k) => k !== sel.id && d.curves.models[k])].slice(0, 8);
    const curve = (kind) => ids.map((id, i) => {
      const m = d.curves.models[id];
      return kind === "roc"
        ? { x: m.roc.fpr, y: m.roc.tpr, name: `${id} · ${m.roc_auc.toFixed(3)}`, mode: "lines", line: { color: S[i], width: id === sel.id ? 3.2 : 1.8, shape: "hv" }, hovertemplate: `${esc(m.name)}<br>FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>` }
        : { x: m.pr.recall, y: m.pr.precision, name: `${id} · ${m.pr_auc.toFixed(3)}`, mode: "lines", line: { color: S[i], width: id === sel.id ? 3.2 : 1.8 }, hovertemplate: `${esc(m.name)}<br>Recall %{x:.2f} · Precision %{y:.2f}<extra></extra>` };
    });
    const muted = cssVar("--muted");
    plot($("#roc", view), [...curve("roc"), { x: [0, 1], y: [0, 1], mode: "lines", line: { dash: "dash", color: muted, width: 1 }, showlegend: false, hoverinfo: "skip" }],
      { xaxis: { title: { text: "False positive rate" }, range: [0, 1] }, yaxis: { title: { text: "True positive rate" }, range: [0, 1.02] }, legend: { y: -0.22 } });
    const base = d.curves.y_test.reduce((a, b) => a + b, 0) / d.curves.y_test.length;
    plot($("#pr", view), [...curve("pr"), { x: [0, 1], y: [base, base], mode: "lines", line: { dash: "dash", color: muted, width: 1 }, showlegend: false, hoverinfo: "skip" }],
      { xaxis: { title: { text: "Recall" }, range: [0, 1] }, yaxis: { title: { text: "Precision" }, range: [0, 1.02] }, legend: { y: -0.22 } });

    const rows = d.results.filter((r) => r.model_key !== "dummy").sort((a, b) => a.test.f1 - b.test.f1);
    plot($("#f1c", view), [{ type: "bar", orientation: "h", y: rows.map((r) => r.id), x: rows.map((r) => r.test.f1), customdata: rows.map((r) => r.name),
      marker: { color: rows.map((r) => (r.id === sel.id ? S[1] : S[0])), cornerradius: 4 }, hovertemplate: "%{customdata}<br>F1 %{x:.3f}<extra></extra>",
      text: rows.map((r) => r.test.f1.toFixed(3)), textposition: "inside", insidetextanchor: "end", textfont: { color: "#fff", size: 10.5 } }],
    { xaxis: { range: [0, 1], title: { text: "F1 (test)" } }, margin: { l: 44 }, bargap: 0.3 });
    const r2 = d.results.filter((r) => r.model_key !== "dummy").sort((a, b) => b.test.pr_auc - a.test.pr_auc);
    plot($("#aucc", view), [
      { type: "bar", name: "PR-AUC", x: r2.map((r) => r.id), y: r2.map((r) => r.test.pr_auc), marker: { color: S[0], cornerradius: 3 }, customdata: r2.map((r) => r.name), hovertemplate: "%{customdata}<br>PR-AUC %{y:.3f}<extra></extra>" },
      { type: "bar", name: "ROC-AUC", x: r2.map((r) => r.id), y: r2.map((r) => r.test.roc_auc), marker: { color: S[2], cornerradius: 3 }, customdata: r2.map((r) => r.name), hovertemplate: "%{customdata}<br>ROC-AUC %{y:.3f}<extra></extra>" }],
    { barmode: "group", yaxis: { range: [0, 1], title: { text: "score (test)" } }, bargap: 0.25, bargroupgap: 0.08 });

    const cm = ev.confusion_matrix, tot = cm.flat().reduce((a, b) => a + b, 0);
    plot($("#cm", view), [{ type: "heatmap", z: [cm[1], cm[0]], x: ["Pred: success", "Pred: error"], y: ["Actual: error", "Actual: success"],
      colorscale: [[0, cssVar("--surface-2") || "#eee"], [1, S[0]]], showscale: false, xgap: 4, ygap: 4,
      text: [[`FN ${cm[1][0]}`, `TP ${cm[1][1]}`], [`TN ${cm[0][0]}`, `FP ${cm[0][1]}`]].map((row, i) => row.map((t, j) => `${t}<br>${(([cm[1], cm[0]][i][j] / tot) * 100).toFixed(1)}%`)),
      texttemplate: "%{text}", textfont: { size: 14, color: cssVar("--text") }, hovertemplate: "%{y} · %{x}: %{z}<extra></extra>" }],
    { margin: { l: 96, t: 10, b: 40 }, xaxis: { showgrid: false }, yaxis: { showgrid: false } });
    plot($("#cal", view), [
      { x: [0, 1], y: [0, 1], mode: "lines", line: { dash: "dash", color: muted, width: 1 }, name: "perfect", hoverinfo: "skip" },
      { x: ev.calibration.mean_predicted, y: ev.calibration.fraction_positive, mode: "lines+markers", name: "model", line: { color: S[0], width: 2.4 }, marker: { size: 8, color: S[0], line: { color: cssVar("--surface-solid"), width: 2 } },
        hovertemplate: "predicted %{x:.2f}<br>observed %{y:.2f}<extra></extra>" }],
    { xaxis: { title: { text: "mean predicted probability" }, range: [0, 1] }, yaxis: { title: { text: "observed error rate" }, range: [0, 1] } });
    const pd = ev.probability_distribution, ctr = pd.bins.slice(0, -1).map((b, i) => (b + pd.bins[i + 1]) / 2);
    plot($("#pd", view), [
      { type: "bar", x: ctr, y: pd.negative, name: "actual success", marker: { color: S[0], cornerradius: 2 }, hovertemplate: "p≈%{x:.2f}: %{y}<extra>success</extra>" },
      { type: "bar", x: ctr, y: pd.positive, name: "actual error", marker: { color: S[1], cornerradius: 2 }, hovertemplate: "p≈%{x:.2f}: %{y}<extra>error</extra>" }],
    { barmode: "group", bargap: 0.15, xaxis: { title: { text: "P(agent error)" }, range: [0, 1] }, yaxis: { title: { text: "tools" } },
      shapes: [{ type: "line", x0: ev.threshold, x1: ev.threshold, y0: 0, y1: 1, yref: "paper", line: { color: cssVar("--text"), dash: "dot", width: 1.4 } }] });
    const ab = d.ablation.rows;
    plot($("#abl", view), [{ type: "bar", orientation: "h", y: ab.map((r) => `${r.set[0]} · ${r.label}`).reverse(), x: ab.map((r) => r.cv.pr_auc.mean).reverse(),
      error_x: { type: "data", array: ab.map((r) => r.cv.pr_auc.std).reverse(), color: muted, thickness: 1.2, width: 4 },
      marker: { color: ab.map((r) => (r.set.startsWith("F") ? S[1] : S[0])).reverse(), cornerradius: 4 },
      customdata: ab.map((r) => [r.n_features, r.test.pr_auc]).reverse(), hovertemplate: "%{y}<br>CV PR-AUC %{x:.3f}<br>%{customdata[0]} features · test %{customdata[1]:.3f}<extra></extra>" }],
    { xaxis: { range: [0, 1], title: { text: "PR-AUC (mean ± sd)" } }, margin: { l: 210 }, bargap: 0.3 });
  });
}
