import { $, api, esc, fmt, icon, openDrawer, errorState, skeleton, cssVar } from "../core.js";
import { plot, themed } from "../charts.js";

const STAGES = { all: "All runs", matrix: "Experiment matrix", benchmark: "Benchmark", optimization: "Optimisation" };

export default async function experiments(view) {
  view.innerHTML = `<div class="page-head"><div><div class="eyebrow">Experiment tracking</div><h1 class="mt8">Experiments</h1></div></div>${skeleton(420)}`;
  let d;
  try { d = await api("/experiments"); } catch (e) { view.innerHTML = errorState(e); return; }
  let stage = "all";
  const runs = d.runs;

  view.innerHTML = `
  <div class="page-head"><div><div class="eyebrow">Experiment tracking</div><h1 class="mt8">Experiments</h1>
    <p>Every training run is logged twice: to MLflow (SQLite backend) and to an append-only JSONL log that powers this page. Hyper-parameters were optimised with Optuna (TPE sampler + median pruning) on grouped CV inside the training split.</p></div></div>
  <div class="grid g4">
    <div class="panel kpi"><div class="kpi-icon">${icon("flask")}</div><div class="label">Tracked runs</div><div class="value">${runs.length}</div><div class="foot">latest run group ${esc((runs[0] || {}).run_group || "—")}</div></div>
    ${["xgboost", "lightgbm"].map((m) => { const b = d.optuna[m].best; return `<div class="panel kpi"><div class="kpi-icon">${icon("spark")}</div><div class="label">Optuna · ${m}</div>
      <div class="value">${b ? fmt(b.best_value_cv_pr_auc) : "—"}</div><div class="foot">${b ? `best CV PR-AUC · ${b.n_trials} trials (${b.n_pruned} pruned) · ${b.search_seconds}s` : "not run"}</div></div>`; }).join("")}
    <div class="panel kpi"><div class="kpi-icon">${icon("database")}</div><div class="label">MLflow</div><div class="value" style="font-size:20px">${d.mlflow.enabled ? "enabled" : "disabled"}</div><div class="foot mono" style="font-size:11px">${esc(d.mlflow.backend)}</div></div>
  </div>

  <div class="panel mt16">
    <div class="panel-head"><div><h3>Run log</h3><div class="sub">Click a run to inspect parameters and all logged metrics</div></div>
      <div class="seg" id="stage-seg">${Object.entries(STAGES).map(([k, l]) => `<button data-s="${k}" class="${k === stage ? "on" : ""}">${l}</button>`).join("")}</div></div>
    <div class="table-wrap" id="runs"></div>
  </div>

  <div class="grid g2 mt16">
    ${["xgboost", "lightgbm"].map((m) => `<div class="panel"><div class="panel-head"><div><h3>Optuna history · ${m}</h3><div class="sub">Trial objective (mean grouped-CV PR-AUC) and best-so-far</div></div></div><div class="chart" id="oh-${m}"></div></div>`).join("")}
  </div>
  <div class="grid g2 mt16">
    ${["xgboost", "lightgbm"].map((m) => { const b = d.optuna[m].best; return `<div class="panel"><div class="panel-head"><div><h3>Best hyper-parameters · ${m}</h3><div class="sub">${b ? `trial #${b.best_trial} · ${esc(b.objective)}` : ""}</div></div></div>
      ${b ? `<div class="feature-grid">${Object.entries(b.best_params).map(([k, v]) => `<div><span>${esc(k)}</span><b>${typeof v === "number" ? (Number.isInteger(v) ? v : v.toPrecision(3)) : esc(v)}</b></div>`).join("")}</div>` : '<p class="muted">Not available.</p>'}</div>`; }).join("")}
  </div>
  <div class="panel mt16"><div class="panel-head"><div><h3>Learning-rate vs objective (XGBoost)</h3><div class="sub">Each point is a completed trial · colour = max_depth</div></div></div><div class="chart" id="lr"></div></div>
  <div class="callout mt16">${icon("info")}<div>Open the full MLflow UI locally with <code>${esc(d.mlflow.ui_command)}</code>, then browse to <code>http://127.0.0.1:5000</code>.</div></div>`;

  const drawRuns = () => {
    const rows = runs.filter((r) => stage === "all" || r.stage === stage);
    $("#runs", view).innerHTML = `<table class="tbl"><thead><tr><th>Experiment ID</th><th>Name</th><th>Stage</th><th>Model</th><th>Feature set</th><th class="num">F1</th><th class="num">ROC-AUC</th><th class="num">PR-AUC</th><th class="num">Train s</th><th>Timestamp</th></tr></thead><tbody>
      ${rows.map((r, i) => { const m = r.metrics || {}; const f1 = m.test_f1, roc = m.test_roc_auc, pr = m.test_pr_auc ?? m.cv_pr_auc; return `<tr class="clickable" data-i="${runs.indexOf(r)}">
        <td class="mono" style="font-size:12px">${esc(r.experiment_id)}</td><td><strong>${esc(r.name)}</strong></td><td><span class="chip">${esc(r.stage)}</span></td><td>${esc(r.model)}</td>
        <td class="muted" style="font-size:12px">${esc(r.feature_set)}</td><td class="num">${fmt(f1)}</td><td class="num">${fmt(roc)}</td><td class="num">${fmt(pr)}${m.cv_pr_auc !== undefined ? ' <span class="muted">cv</span>' : ""}</td>
        <td class="num">${fmt(r.training_time_s, 2)}</td><td class="muted mono" style="font-size:11.5px">${esc((r.timestamp || "").replace("T", " ").replace("+00:00", ""))}</td></tr>`; }).join("")}</tbody></table>`;
    $("#runs", view).querySelectorAll("tr[data-i]").forEach((tr) => tr.addEventListener("click", () => {
      const r = runs[+tr.dataset.i];
      openDrawer(r.name, `<div class="row"><span class="chip">${esc(r.experiment_id)}</span><span class="chip">${esc(r.stage)}</span><span class="chip">dataset ${esc(r.dataset_version)}</span></div>
        <div><h3 style="margin-bottom:8px">Parameters</h3><pre class="code">${esc(JSON.stringify(r.params, null, 2))}</pre></div>
        <div><h3 style="margin-bottom:8px">Metrics</h3><pre class="code">${esc(JSON.stringify(r.metrics, null, 2))}</pre></div>
        <p class="muted">Model: ${esc(r.model)} · features: ${esc(r.feature_set)} · logged ${esc(r.timestamp)}</p>`);
    }));
  };
  drawRuns();
  view.querySelectorAll("#stage-seg button").forEach((b) => b.addEventListener("click", () => {
    stage = b.dataset.s; view.querySelectorAll("#stage-seg button").forEach((x) => x.classList.toggle("on", x === b)); drawRuns();
  }));

  themed(() => {
    for (const m of ["xgboost", "lightgbm"]) {
      const t = d.optuna[m].trials;
      const done = t.filter((x) => x.state === "COMPLETE");
      let best = -1;
      const bestLine = done.map((x) => (best = Math.max(best, x.value)));
      const pruned = t.filter((x) => x.state === "PRUNED");
      plot($(`#oh-${m}`, view), [
        { x: done.map((x) => x.trial), y: done.map((x) => x.value), mode: "markers", name: "completed trial", marker: { size: 8, color: cssVar("--s1"), line: { color: cssVar("--surface-solid"), width: 1.5 } }, hovertemplate: "trial %{x}<br>CV PR-AUC %{y:.4f}<extra></extra>" },
        { x: pruned.map((x) => x.trial), y: pruned.map(() => Math.min(...done.map((x) => x.value))), mode: "markers", name: "pruned", marker: { size: 7, symbol: "x-thin", line: { width: 2, color: cssVar("--muted") } }, hovertemplate: "trial %{x} pruned<extra></extra>" },
        { x: done.map((x) => x.trial), y: bestLine, mode: "lines", name: "best so far", line: { color: cssVar("--s2"), width: 2.4, shape: "hv" }, hoverinfo: "skip" }],
      { xaxis: { title: { text: "trial" } }, yaxis: { title: { text: "CV PR-AUC" } }, legend: { y: 1.1, x: 0 }, margin: { t: 30 } });
    }
    const t = d.optuna.xgboost.trials.filter((x) => x.state === "COMPLETE");
    plot($("#lr", view), [{ x: t.map((x) => x.learning_rate), y: t.map((x) => x.value), mode: "markers", marker: { size: 10, color: t.map((x) => x.max_depth), colorscale: [[0, cssVar("--s3")], [1, cssVar("--s7")]],
      colorbar: { title: { text: "max_depth", side: "right" }, thickness: 10, outlinewidth: 0 }, line: { color: cssVar("--surface-solid"), width: 1.5 } },
      customdata: t.map((x) => [x.trial, x.n_estimators]), hovertemplate: "trial %{customdata[0]}<br>lr %{x:.3f} · %{customdata[1]} trees<br>CV PR-AUC %{y:.4f}<extra></extra>" }],
    { xaxis: { type: "log", title: { text: "learning_rate (log)" } }, yaxis: { title: { text: "CV PR-AUC" } } });
  });
}
