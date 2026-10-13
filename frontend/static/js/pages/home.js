import { api, esc, fmt, icon, int, pct, riskBadge, riskColor, tip, errorState, skeleton } from "../core.js";
import { plot, themed, series } from "../charts.js";
import { cssVar } from "../core.js";

const kpi = (label, value, foot, help, bar) => `
  <div class="panel kpi">
    <div class="label">${label}${help ? tip(help) : ""}</div>
    <div class="value">${value}</div>
    ${bar !== undefined ? `<div class="bar"><span style="width:${Math.max(2, Math.min(100, bar * 100))}%"></span></div>` : ""}
    <div class="foot">${foot}</div>
  </div>`;

const example = (t) => !t ? "" : `<a class="ex" href="#/explain" title="Open in Explainability">
  <span class="tname">${esc(t.tool_name)}</span>
  <span class="row" style="gap:10px">${riskBadge(t.risk_level)}<span class="p" style="color:${riskColor(t.proba)}">${(t.proba * 100).toFixed(1)}%</span></span>
  <span class="tdesc">${esc(t.tool_description || "")}</span></a>`;

export default async function home(view) {
  view.innerHTML = `
  <section class="hero">
    <div class="hero-grid">
      <div>
        <span class="tagline"><b>MCPGuard</b>Predict. Explain. Prevent agent failures.</span>
        <h1 class="mt16">Find the MCP tools that will <span class="grad-text">break your AI agent</span></h1>
        <p class="lede mt16">Paste a tool definition and get its failure risk, the reasons behind it, and concrete fixes — before an agent ever calls it.</p>
        <div class="cta">
          <a class="btn btn-primary btn-lg" href="#/analyze">${icon("scan")}Analyze a tool</a>
          <a class="btn btn-lg" href="#/performance">${icon("chart")}Model performance</a>
        </div>
      </div>
      <div class="examples" id="hero-visual">${skeleton(78, 2)}</div>
    </div>
  </section>

  <div class="grid g4 mt24" id="kpis">${skeleton(118, 4)}</div>

  <div class="grid g2 mt24">
    <div class="panel" id="model-status">${skeleton(200)}</div>
    <div class="panel" id="dataset-status">${skeleton(200)}</div>
  </div>

  <div class="grid g2 mt24">
    <div class="panel"><div class="panel-head"><div><h3>Risk across all tools</h3><div class="sub">How many dataset tools fall in each predicted risk band</div></div></div><div class="chart sm" id="risk-chart"></div></div>
    <div class="panel"><div class="panel-head"><div><h3>Best models (test PR-AUC)</h3><div class="sub">Orange = selected production model</div></div><a class="btn btn-sm btn-ghost" href="#/performance">All models ${icon("arrow")}</a></div><div class="chart sm" id="model-chart"></div></div>
  </div>

  <div class="panel mt24">
    <div class="panel-head"><div><h3>How it works</h3><div class="sub">From raw tool definitions to an explained risk score</div></div><a class="btn btn-sm btn-ghost" href="#/about">Methodology ${icon("arrow")}</a></div>
    <div class="steps">${[["Collect", "BFCL v3 + MCP server tools"], ["Label", "controlled agent benchmark"], ["Featurize", "TF-IDF · MiniLM · schema"], ["Train", "Optuna-tuned XGBoost"], ["Explain", "SHAP + fix suggestions"], ["Serve", "FastAPI + Docker"]]
      .map(([a, b]) => `<div><b>${a}</b><br>${b}</div>`).join("")}</div>
  </div>`;

  let stats, perf, ds;
  try {
    [stats, perf, ds] = await Promise.all([api("/stats"), api("/model-metrics"), api("/dataset-summary")]);
  } catch (e) {
    view.querySelector("#kpis").innerHTML = `<div class="span3" style="grid-column:1/-1">${errorState(e)}</div>`;
    return;
  }
  const ev = perf.evaluation;
  const meta = perf.model;
  view.querySelector("#kpis").innerHTML = [
    kpi("PR-AUC", fmt(stats.test_pr_auc), `no-skill baseline ${pct(stats.positive_rate, 0)}`, "Primary metric (average precision) on the held-out test split. A random model scores the failure base rate.", stats.test_pr_auc),
    kpi("ROC-AUC", fmt(stats.test_roc_auc), "ranking quality · test split", "Probability that a random failing tool is ranked above a random successful one.", stats.test_roc_auc),
    kpi("F1 score", fmt(stats.test_f1), `95% CI ${fmt(ev.bootstrap_ci_95.f1.low, 2)}–${fmt(ev.bootstrap_ci_95.f1.high, 2)}`, "Harmonic mean of precision and recall at the validation-tuned threshold.", stats.test_f1),
    kpi("Tools in dataset", int(stats.tools_analyzed), `${int(stats.servers)} servers · ${int(stats.high_risk_tools)} high-risk`, "Clean, de-duplicated MCP-format tool definitions scored by the production model."),
  ].join("");

  // hero preview: the riskiest and safest real test-set tools, with their SHAP drivers
  try {
    const [hi, lo] = await Promise.all([
      api("/dataset/tools?split=test&sort=proba&desc=true&page_size=1"),
      api("/dataset/tools?split=test&sort=proba&desc=false&page_size=1"),
    ]);
    const [a, b] = await Promise.all([api(`/tools/${hi.rows[0].tool_id}`), api(`/tools/${lo.rows[0].tool_id}`)]);
    view.querySelector("#hero-visual").innerHTML = `<div class="cap">Live examples from the test split</div>` + example(a) + example(b);
  } catch (e) { view.querySelector("#hero-visual").innerHTML = ""; }

  const sel = perf.results.find((r) => r.id === perf.selection.selected_id);
  view.querySelector("#model-status").innerHTML = `
    <div class="row" style="justify-content:space-between"><div class="eyebrow">Production model</div><span class="pill pill-brand">${icon("check")}selected on validation</span></div>
    <h2 class="mt12">${esc(meta.model_name)}</h2>
    <p class="text-2 mt8">${esc(meta.experiment_name)} — chosen by validation PR-AUC, then F1, then ROC-AUC.</p>
    <div class="kv mt16" style="grid-template-columns:repeat(3,minmax(0,1fr))">
      <div><div class="k">Features</div><div class="v">${meta.n_features}</div></div>
      <div><div class="k">Threshold ${tip("Decision threshold that maximises F1 on the validation split.")}</div><div class="v">${fmt(meta.threshold, 3)}</div></div>
      <div><div class="k">Val PR-AUC</div><div class="v">${fmt(sel.val.pr_auc)}</div></div>
    </div>
    <p class="muted mt16" style="font-size:12px">Trained ${esc(meta.trained_at.slice(0, 10))} · dataset ${esc(meta.dataset_version)}</p>`;

  const s = ds.summary;
  const prov = s.by_source;
  const tot = Object.values(prov).reduce((a, b) => a + b, 0);
  const pcol = { bfcl_v3: "--s1", mcp_reference_servers: "--s3", augmented: "--s2" };
  const pname = { bfcl_v3: "BFCL v3", mcp_reference_servers: "MCP servers", augmented: "Augmented" };
  view.querySelector("#dataset-status").innerHTML = `
    <div class="row" style="justify-content:space-between"><div class="eyebrow">Dataset</div><a class="pill pill-link" href="#/dataset">${icon("database")}Explore</a></div>
    <h2 class="mt12">${int(ds.quality.output_records)} clean tool records</h2>
    <p class="text-2 mt8">${pct(s.positive_rate)} cause agent errors · labels from a controlled agent benchmark.</p>
    <div class="family-split mt16" style="height:10px">${Object.entries(prov).map(([k, v]) => `<span style="width:${(v / tot) * 100}%;background:var(${pcol[k] || "--s8"})" title="${esc(pname[k] || k)}: ${v}"></span>`).join("")}</div>
    <div class="legend-row mt8">${Object.entries(prov).map(([k, v]) => `<span><i class="sw" style="background:var(${pcol[k] || "--s8"})"></i>${esc(pname[k] || k)} <b class="mono">${int(v)}</b></span>`).join("")}</div>
    <p class="muted mt16" style="font-size:12px">Split ${["train", "val", "test"].map((k) => `${k} ${int(ds.split[k].n)}`).join(" · ")} · grouped by tool lineage, zero overlap</p>`;

  themed(() => {
    const rc = ev.risk_level_counts_all_tools;
    const names = Object.keys(rc);
    const colors = { LOW: cssVar("--good"), MEDIUM: cssVar("--warning"), HIGH: cssVar("--serious"), CRITICAL: cssVar("--critical") };
    plot(view.querySelector("#risk-chart"), [{ type: "bar", x: names, y: names.map((n) => rc[n]), marker: { color: names.map((n) => colors[n]), cornerradius: 4 },
      text: names.map((n) => rc[n]), textposition: "outside", cliponaxis: false, textfont: { color: cssVar("--text-2") }, hovertemplate: "%{x}: %{y} tools<extra></extra>" }],
    { margin: { t: 20, b: 36, l: 44 }, yaxis: { title: { text: "tools" }, range: [0, Math.max(...names.map((n) => rc[n])) * 1.18] }, bargap: 0.45 });
    const rows = perf.results.filter((r) => !["dummy"].includes(r.model_key)).sort((a, b) => b.test.pr_auc - a.test.pr_auc).slice(0, 7).reverse();
    plot(view.querySelector("#model-chart"), [{ type: "bar", orientation: "h", y: rows.map((r) => r.id), x: rows.map((r) => r.test.pr_auc),
      marker: { color: rows.map((r) => (r.id === perf.selection.selected_id ? series()[1] : series()[0])), cornerradius: 4 },
      customdata: rows.map((r) => r.name), text: rows.map((r) => r.test.pr_auc.toFixed(3)), textposition: "inside", insidetextanchor: "end",
      textfont: { color: "#fff", size: 11 }, hovertemplate: "%{customdata}<br>PR-AUC %{x:.3f}<extra></extra>" }],
    { margin: { t: 10, b: 36, l: 44 }, xaxis: { range: [0, 1], title: { text: "test PR-AUC" } }, bargap: 0.35 });
  });
}
