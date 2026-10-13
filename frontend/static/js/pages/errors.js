import { $, api, esc, fmt, icon, openDrawer, outcomeTag, prettyType, probCell, shapBars, errorState, skeleton, cssVar } from "../core.js";
import { plot, themed } from "../charts.js";

const QUAD = {
  FN: { t: "False negatives", d: "Failed, but predicted safe — dangerous misses" },
  FP: { t: "False positives", d: "Predicted risky, but succeeded" },
  TP: { t: "True positives", d: "Correctly flagged failures" },
  TN: { t: "True negatives", d: "Correctly cleared tools" },
};
const SLICES = { error_type: "Failure mode", source: "Source", desc_len_bucket: "Description length", param_bucket: "Parameter count", bfcl_category: "BFCL category", error_severity: "Severity" };

function detail(r) {
  let schema = r.input_schema;
  try { schema = JSON.stringify(JSON.parse(r.input_schema), null, 2); } catch (e) { /* malformed – show raw */ }
  const counts = Object.entries(r.error_type_counts || {}).sort((a, b) => b[1] - a[1]);
  const tot = counts.reduce((a, [, v]) => a + v, 0) || 1;
  return `
  <div class="row">${outcomeTag(r.actual && r.predicted ? "TP" : !r.actual && !r.predicted ? "TN" : r.predicted ? "FP" : "FN")}<span class="chip">${esc(r.source)}</span><span class="chip">${esc(r.server_name)}</span></div>
  <div class="kv"><div><div class="k">Actual</div><div class="v">${r.actual ? "error" : "success"}</div></div><div><div class="k">Predicted</div><div class="v">${r.predicted ? "error" : "success"}</div></div>
    <div><div class="k">Probability</div><div class="v">${(r.probability * 100).toFixed(1)}%</div></div><div><div class="k">Failure rate</div><div class="v">${(r.failure_rate * 100).toFixed(0)}%</div></div></div>
  <div class="callout">${icon("bulb")}<div><strong>Possible reason.</strong> ${esc(r.possible_reason)}</div></div>
  <div><h3 style="margin-bottom:8px">Tool description</h3><p class="text-2">${esc(r.description)}</p></div>
  <div><h3 style="margin-bottom:8px">Benchmark episodes (${tot})</h3>
    <div class="family-split" style="height:14px">${counts.map(([k, v], i) => `<span title="${esc(k)}: ${v}" style="width:${(v / tot) * 100}%;background:${k === "NONE" ? "var(--good)" : `var(--s${(i % 7) + 2})`}"></span>`).join("")}</div>
    <div class="legend-row mt8">${counts.map(([k, v], i) => `<span><i class="sw" style="background:${k === "NONE" ? "var(--good)" : `var(--s${(i % 7) + 2})`}"></i>${esc(prettyType(k))} <b class="mono">${v}</b></span>`).join("")}</div></div>
  <div><h3 style="margin-bottom:10px">Important features (SHAP)</h3>${shapBars(r.important_features)}</div>
  <div><h3 style="margin-bottom:8px">Input schema</h3><pre class="code">${esc(schema)}</pre></div>`;
}

export default async function errors(view) {
  view.innerHTML = `<div class="page-head"><div><div class="eyebrow">Rigorous error analysis</div><h1 class="mt8">Error Analysis</h1></div></div>${skeleton(400)}`;
  let ea;
  try { ea = await api("/error-analysis"); } catch (e) { view.innerHTML = errorState(e); return; }
  let kind = "FN", slice = "error_type";
  const c = ea.counts, n = Object.values(c).reduce((a, b) => a + b, 0);

  view.innerHTML = `
  <div class="page-head"><div><div class="eyebrow">Rigorous error analysis</div><h1 class="mt8">Error Analysis</h1>
    <p>Where and why the production model is wrong on the held-out test split. False negatives are prioritised — a missed failure ships a broken tool to agents.</p></div></div>
  <div class="headline">
    <div class="eyebrow">${icon("target")} Most common source of model mistakes</div>
    <h3 class="mt8">${esc(ea.summary.headline)}</h3>
    <ul>${ea.summary.points.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>
  </div>
  <div class="quad mt16">${["FN", "FP", "TP", "TN"].map((k) => `<button data-k="${k}" class="${k === kind ? "on" : ""}">
    <span class="q-k">${QUAD[k].t} ${outcomeTag(k)}</span><span class="q-v">${c[k] || 0}</span><span class="q-d">${QUAD[k].d} · ${(((c[k] || 0) / n) * 100).toFixed(1)}%</span></button>`).join("")}</div>
  <div class="grid g3 mt16" style="align-items:start">
    <div class="panel span2"><div class="panel-head"><div><h3 id="ex-title"></h3><div class="sub">Most confident cases first · click a row for the full explanation</div></div></div><div class="table-wrap" id="ex"></div></div>
    <div class="stack">
      <div class="panel"><div class="panel-head"><div><h3>Failure modes behind false negatives</h3><div class="sub">Dominant benchmark error type of missed failures</div></div></div><div class="chart sm" id="fnm"></div></div>
      <div class="panel"><div class="panel-head"><div><h3>FP vs TN profile</h3><div class="sub">Mean feature values</div></div></div>
        <table class="tbl"><thead><tr><th>Feature</th><th class="num">FP</th><th class="num">TN</th></tr></thead><tbody>
        ${Object.entries(ea.summary.fp_vs_tn_profile).map(([k, v]) => `<tr><td>${esc(k.replaceAll("_", " "))}</td><td class="num">${fmt(v.fp_mean, 2)}</td><td class="num">${fmt(v.tn_mean, 2)}</td></tr>`).join("")}</tbody></table></div>
    </div>
  </div>
  <div class="panel mt16"><div class="panel-head"><div><h3>Slice analysis</h3><div class="sub">Misclassification rate and FN / FP counts per data slice (test split)</div></div>
    <div class="seg" id="slice-seg">${Object.entries(SLICES).map(([k, l]) => `<button data-s="${k}" class="${k === slice ? "on" : ""}">${l}</button>`).join("")}</div></div>
    <div class="grid g2"><div class="chart" id="sl"></div><div class="table-wrap" id="sl-t" style="max-height:340px"></div></div></div>`;

  const showEx = () => {
    $("#ex-title", view).innerHTML = `${outcomeTag(kind)} ${QUAD[kind].t}`;
    const rows = ea.examples[kind] || [];
    $("#ex", view).innerHTML = rows.length ? `<table class="tbl"><thead><tr><th>Tool</th><th>Failure mode</th><th class="num">P(error)</th><th>Top feature</th></tr></thead><tbody>
      ${rows.map((r, i) => `<tr class="clickable" data-i="${i}"><td><strong class="mono">${esc(r.tool_name)}</strong><div class="cell-desc">${esc(r.description)}</div></td>
        <td><span class="chip">${esc(prettyType(r.error_type))}</span></td><td class="num">${probCell(r.probability)}</td>
        <td class="text-2" style="font-size:12px">${esc((r.important_features[0] || {}).display_name || "—")}</td></tr>`).join("")}</tbody></table>`
      : '<div class="empty">No examples.</div>';
    $("#ex", view).querySelectorAll("tr[data-i]").forEach((tr) => tr.addEventListener("click", () => {
      const r = rows[+tr.dataset.i];
      openDrawer(r.tool_name, detail(r));
    }));
  };
  showEx();
  view.querySelectorAll(".quad button").forEach((b) => b.addEventListener("click", () => {
    kind = b.dataset.k; view.querySelectorAll(".quad button").forEach((x) => x.classList.toggle("on", x === b)); showEx();
  }));

  const drawSlice = () => {
    const rows = ea.slices[slice] || [];
    plot($("#sl", view), [
      { type: "bar", name: "False negatives", x: rows.map((r) => prettyType(r.slice)), y: rows.map((r) => r.fn), marker: { color: cssVar("--s7"), cornerradius: 3 }, hovertemplate: "%{x}<br>FN %{y}<extra></extra>" },
      { type: "bar", name: "False positives", x: rows.map((r) => prettyType(r.slice)), y: rows.map((r) => r.fp), marker: { color: cssVar("--s4"), cornerradius: 3 }, hovertemplate: "%{x}<br>FP %{y}<extra></extra>" }],
    { barmode: "stack", yaxis: { title: { text: "misclassified tools" } }, xaxis: { tickangle: -30, automargin: true }, bargap: 0.35, legend: { y: 1.12, x: 0 }, margin: { t: 34, b: 20 } });
    $("#sl-t", view).innerHTML = `<table class="tbl"><thead><tr><th>${SLICES[slice]}</th><th class="num">n</th><th class="num">Error rate</th><th class="num">Positive rate</th><th class="num">FN</th><th class="num">FP</th></tr></thead><tbody>
      ${rows.map((r) => `<tr><td><strong>${esc(prettyType(r.slice))}</strong></td><td class="num">${r.n}</td><td class="num">${(r.error_rate * 100).toFixed(1)}%</td><td class="num">${(r.positive_rate * 100).toFixed(0)}%</td><td class="num">${r.fn}</td><td class="num">${r.fp}</td></tr>`).join("")}</tbody></table>`;
  };
  themed(() => {
    drawSlice();
    const fm = Object.entries(ea.summary.fn_failure_modes);
    plot($("#fnm", view), [{ type: "bar", orientation: "h", y: fm.map(([k]) => prettyType(k)).reverse(), x: fm.map(([, v]) => v).reverse(),
      marker: { color: cssVar("--s7"), cornerradius: 4 }, text: fm.map(([, v]) => `${(v * 100).toFixed(0)}%`).reverse(), textposition: "outside", textfont: { color: cssVar("--text-2") },
      hovertemplate: "%{y}: %{x:.0%}<extra></extra>" }], { margin: { l: 160, t: 6 }, xaxis: { tickformat: ".0%", range: [0, Math.max(...fm.map(([, v]) => v)) * 1.25] }, bargap: 0.35 });
  });
  view.querySelectorAll("#slice-seg button").forEach((b) => b.addEventListener("click", () => {
    slice = b.dataset.s; view.querySelectorAll("#slice-seg button").forEach((x) => x.classList.toggle("on", x === b)); drawSlice();
  }));
}
