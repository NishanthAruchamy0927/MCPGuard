import { $, api, bindTabs, tabs, esc, fmt, icon, int, openDrawer, pct, prettyType, probCell, riskBadge, shapBars, tip, errorState, skeleton, cssVar } from "../core.js";
import { plot, themed, series } from "../charts.js";

const SRC = { bfcl_v3: "BFCL v3 (real)", mcp_reference_servers: "MCP reference (real)", augmented: "Augmented (synthetic)" };

async function toolDrawer(id) {
  const body = openDrawer("Loading…", skeleton(420));
  const t = await api(`/tools/${encodeURIComponent(id)}`);
  document.querySelector("#drawer-title").textContent = t.tool_name;
  const e = t.explanation || {};
  body.innerHTML = `
    <div class="row">${riskBadge(t.risk_level)}<span class="chip">${esc(SRC[t.source] || t.source)}</span><span class="chip">${esc(t.server_name)}</span><span class="chip">${esc(t.split)} split</span></div>
    <div class="kv"><div><div class="k">P(error)</div><div class="v">${t.proba != null ? (t.proba * 100).toFixed(1) + "%" : "—"}</div></div><div><div class="k">Actual</div><div class="v">${t.error_occurred ? "error" : "success"}</div></div>
      <div><div class="k">Severity</div><div class="v" style="font-size:14px">${esc(t.error_severity)}</div></div><div><div class="k">Episodes</div><div class="v">${t.execution_count}</div></div></div>
    <div><h3 style="margin-bottom:6px">Description</h3><p class="text-2">${esc(t.tool_description)}</p></div>
    <div class="callout">${icon("info")}<div><strong>Provenance:</strong> ${esc(t.provenance)} · labels ${esc(t.label_provenance)} (${esc(t.episode_origin)}). ${esc(t.provenance_note)}</div></div>
    <div><h3 style="margin-bottom:10px">SHAP explanation</h3>${shapBars(e.top)}</div>
    <div><h3 style="margin-bottom:6px">Benchmark outcome counts</h3><pre class="code">${esc(JSON.stringify(t.error_type_counts, null, 2))}</pre></div>
    <div><h3 style="margin-bottom:6px">Input schema</h3><pre class="code">${esc(typeof t.input_schema === "string" ? t.input_schema : JSON.stringify(t.input_schema, null, 2))}</pre></div>
    <a class="btn" href="#/explain?tool=${encodeURIComponent(t.tool_id)}">${icon("spark")}Open in Explainability</a>`;
}

export default async function dataset(view) {
  view.innerHTML = `<div class="page-head"><div><div class="eyebrow">Data engineering</div><h1 class="mt8">Dataset Explorer</h1></div></div>${skeleton(400)}`;
  let d;
  try { d = await api("/dataset-summary"); } catch (e) { view.innerHTML = errorState(e); return; }
  const s = d.summary, q = d.quality;
  const st = { q: "", risk_level: "", error_type: "", server: "", source: "", split: "", actual: "", predicted: "", sort: "proba", desc: true, page: 1 };

  view.innerHTML = `
  <div class="page-head"><div><div class="eyebrow">Data engineering</div><h1 class="mt8">Dataset Explorer</h1>
    <p>${int(s.records)} MCP-format tool records from real tool catalogues plus documented augmentation. Every label is measured by the controlled agent benchmark — nothing is assigned by hand.</p></div>
    <span class="pill pill-brand">${icon("database")}version ${esc(s.dataset_version)} · built ${esc(s.built_at.slice(0, 10))}</span></div>

  <div class="grid g3">
    ${[["Total records", int(q.output_records), `${int(s.records)} before cleaning`, "Records after exact-duplicate removal."],
       ["Agent-error rate", pct(s.positive_rate), `imbalance ${q.imbalance_ratio} : 1`, "Share of tools with failure rate ≥ 30% across benchmark episodes."],
       ["Servers / sources", int(s.servers), `${int(s.lineages)} lineages`, "Distinct MCP servers or BFCL function sets; lineages group near-duplicates and augmented variants."],
       ["Malformed schemas", int(q.malformed_schemas), "kept & flagged", "Schemas that are not valid JSON-schema objects are kept as a risk signal, not dropped."],
       ["Duplicates removed", int(q.exact_duplicates_removed), `${int(q.near_duplicate_groups)} near-dup groups`, "Exact duplicates removed; near-duplicates share a lineage so they never straddle splits."],
       ["Missing values", int(Object.values(d.missing_values || {}).reduce((a, b) => a + b, 0)), "metadata fields only", "Missing values appear only in optional metadata (e.g. parent id of seed tools)."]]
      .map(([l, v, f, h]) => `<div class="panel kpi"><div class="label">${l}${tip(h)}</div><div class="value">${v}</div><div class="foot">${f}</div></div>`).join("")}
  </div>

  <div class="panel mt24"><div class="panel-head"><div><h3>${icon("shield")} Data provenance</h3><div class="sub">Clearly separated real observations, controlled benchmark results and synthetic data</div></div></div>
    <div class="prov">
      <div><span class="tag" style="background:var(--good-soft);color:var(--good)">REAL_TOOL_DEFINITION</span><div class="n">${int(s.by_provenance.REAL_TOOL_DEFINITION || 0)}</div><p class="text-2">Tool definitions from BFCL v3 (Apache-2.0, ${int(s.by_source.bfcl_v3 || 0)}) and MCP reference servers (${int(s.by_source.mcp_reference_servers || 0)}).</p></div>
      <div><span class="tag" style="background:var(--brand-soft);color:var(--brand-ink)">CONTROLLED_BENCHMARK</span><div class="n">${int(q.output_records)}</div><p class="text-2">All labels: ${s.episodes_per_tool} simulated agent episodes per tool, validated BFCL-style against gold calls (${int(s.by_episode_origin.bfcl_real || 0)} tools use real BFCL user queries).</p></div>
      <div><span class="tag" style="background:var(--warning-soft);color:var(--warning)">SYNTHETIC_AUGMENTED</span><div class="n">${int(s.by_provenance.SYNTHETIC_AUGMENTED || 0)}</div><p class="text-2">Variants created by documented operators (degradation & enrichment); kept in the same lineage as their seed tool.</p></div>
    </div></div>

  <div class="panel mt24">
    <div class="panel-head"><div><h3>Charts</h3><div class="sub">Labels, input distributions and engineered risk scores</div></div>
      ${tabs("dc", [["lab", "Labels"], ["dist", "Distributions"], ["risk", "Risk scores"], ["stats", "Feature statistics"]].map(([k, l]) => [k, l, ""]), { bar: false }).head}</div>
    <div class="tab-pane" data-pane="dc:lab"><div class="grid g3">
      <div><h3 style="font-size:14px">Class balance <span class="muted" style="font-weight:500">· error_occurred</span></h3><div class="chart sm" id="cls"></div></div>
      <div><h3 style="font-size:14px">Dominant failure mode</h3><div class="chart sm" id="et"></div></div>
      <div><h3 style="font-size:14px">Severity by source</h3><div class="chart sm" id="sev"></div></div></div></div>
    <div class="tab-pane" data-pane="dc:dist" hidden><div class="grid g3">
      ${[["description_length", "Description length (chars)"], ["parameter_count", "Parameter count"], ["failure_rate", "Benchmark failure rate"]].map(([k, l]) => `<div><h3 style="font-size:14px">${l}</h3><div class="chart sm" id="h-${k}"></div></div>`).join("")}</div>
      <p class="muted mt12" style="font-size:12px">Histograms are clipped at the 98th percentile.</p></div>
    <div class="tab-pane" data-pane="dc:risk" hidden><h3 style="font-size:14px">Mean engineered risk score · successful vs failing tools</h3><div class="chart" id="rs"></div></div>
    <div class="tab-pane" data-pane="dc:stats" hidden><div class="table-wrap">
      <table class="tbl"><thead><tr><th>Feature</th><th class="num">mean</th><th class="num">std</th><th class="num">min</th><th class="num">median</th><th class="num">max</th></tr></thead><tbody>
      ${d.feature_statistics.map((r) => `<tr><td><strong>${esc(r.feature.replaceAll("_", " "))}</strong></td><td class="num">${fmt(r.mean, 2)}</td><td class="num">${fmt(r.std, 2)}</td><td class="num">${fmt(r.min, 2)}</td><td class="num">${fmt(r["50%"], 2)}</td><td class="num">${fmt(r.max, 2)}</td></tr>`).join("")}</tbody></table></div></div>
  </div>

  <div class="panel mt24">
    <div class="panel-head"><div><h3>${icon("filter")} Browse all tools</h3><div class="sub" id="tbl-count">—</div></div></div>
    <div class="grid g4" style="gap:10px">
      <input class="input" id="f-q" placeholder="Search name or description…" />
      <select class="select" id="f-risk_level"><option value="">Any risk level</option>${["LOW", "MEDIUM", "HIGH", "CRITICAL"].map((x) => `<option>${x}</option>`).join("")}</select>
      <select class="select" id="f-error_type"><option value="">Any error type</option></select>
      <select class="select" id="f-server"><option value="">Any server</option></select>
      <select class="select" id="f-source"><option value="">Any source</option>${Object.entries(SRC).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}</select>
      <select class="select" id="f-split"><option value="">Any split</option><option>train</option><option>val</option><option>test</option></select>
      <select class="select" id="f-actual"><option value="">Actual: any</option><option value="1">Actual: error</option><option value="0">Actual: success</option></select>
      <select class="select" id="f-predicted"><option value="">Predicted: any</option><option value="1">Predicted: error</option><option value="0">Predicted: success</option></select>
    </div>
    <div class="table-wrap mt12" id="tbl"></div>
    <div class="row mt12" style="justify-content:space-between"><span class="muted" id="pg-info"></span><div class="row"><button class="btn btn-sm" id="prev">Previous</button><button class="btn btn-sm" id="next">Next</button></div></div>
  </div>`;

  bindTabs(view, "dc");
  themed(() => {
    const S = series();
    const cb = q.class_balance;
    plot($("#cls", view), [{ type: "pie", hole: 0.62, labels: ["Success (0)", "Agent error (1)"], values: [cb["0"], cb["1"]], sort: false,
      marker: { colors: [S[0], S[1]], line: { color: cssVar("--surface-solid"), width: 2 } }, textinfo: "value+percent", textfont: { color: "#fff" }, hovertemplate: "%{label}: %{value}<extra></extra>" }],
    { margin: { l: 10, r: 10, t: 10, b: 10 }, legend: { orientation: "h", y: -0.05 } });
    const et = Object.entries(s.error_type).sort((a, b) => a[1] - b[1]);
    plot($("#et", view), [{ type: "bar", orientation: "h", y: et.map(([k]) => prettyType(k)), x: et.map(([, v]) => v), marker: { color: et.map(([k]) => (k === "NONE" ? cssVar("--good") : S[0])), cornerradius: 4 },
      hovertemplate: "%{y}: %{x}<extra></extra>" }], { margin: { l: 150, t: 6 }, bargap: 0.3 });
    const sev = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"], srcs = Object.keys(d.severity_by_source);
    const sevC = { NONE: S[0], LOW: cssVar("--good"), MEDIUM: cssVar("--warning"), HIGH: cssVar("--serious"), CRITICAL: cssVar("--critical") };
    const SHORT = { bfcl_v3: "BFCL", mcp_reference_servers: "MCP", augmented: "Synthetic" };
    plot($("#sev", view), sev.map((k) => ({ type: "bar", name: k, x: srcs.map((x) => SHORT[x] || x), y: srcs.map((x) => d.severity_by_source[x][k] || 0), marker: { color: sevC[k] }, hovertemplate: `%{x}<br>${k}: %{y}<extra></extra>` })),
      { barmode: "stack", bargap: 0.4, margin: { t: 10, b: 30, r: 90 }, xaxis: { tickangle: 0 }, legend: { orientation: "v", x: 1.02, y: 1, font: { size: 11 } } });
    for (const k of ["description_length", "parameter_count", "failure_rate"]) {
      const h = d.histograms[k];
      const ctr = h.edges.slice(0, -1).map((e, i) => (e + h.edges[i + 1]) / 2);
      plot($(`#h-${k}`, view), [{ type: "bar", x: ctr, y: h.counts, marker: { color: S[0], cornerradius: 2 }, hovertemplate: "≈%{x:.2f}: %{y}<extra></extra>" }], { bargap: 0.08, margin: { t: 6 }, yaxis: { title: { text: "tools" } } });
    }
    const rm = d.risk_feature_means || {};
    const cols = Object.keys(rm["0"] || {});
    plot($("#rs", view), [["0", "Success", S[0]], ["1", "Agent error", S[1]]].map(([k, n, c]) => ({ type: "bar", name: n, x: cols.map((x) => x.replace("_score", "").replaceAll("_", " ")), y: cols.map((x) => rm[k][x]),
      marker: { color: c, cornerradius: 3 }, hovertemplate: `${n}<br>%{x}: %{y:.3f}<extra></extra>` })), { barmode: "group", bargap: 0.3, yaxis: { range: [0, 1], title: { text: "mean score" } }, xaxis: { tickangle: 0 }, legend: { y: 1.12, x: 0 }, margin: { t: 34 } });
  });

  const load = async () => {
    const qs = new URLSearchParams({ sort: st.sort, desc: st.desc, page: st.page, page_size: 20 });
    ["q", "risk_level", "error_type", "server", "source", "split", "actual", "predicted"].forEach((k) => st[k] !== "" && qs.set(k, st[k]));
    $("#tbl", view).innerHTML = skeleton(300);
    const r = await api(`/dataset/tools?${qs}`, { cacheable: false });
    const etSel = $("#f-error_type", view), svSel = $("#f-server", view);
    if (etSel.options.length === 1) {
      etSel.insertAdjacentHTML("beforeend", r.facets.error_type.map((x) => `<option value="${esc(x)}">${esc(prettyType(x))}</option>`).join(""));
      svSel.insertAdjacentHTML("beforeend", r.facets.server.map((x) => `<option>${esc(x)}</option>`).join(""));
    }
    const pages = Math.max(1, Math.ceil(r.total / r.page_size));
    $("#tbl-count", view).textContent = `${int(r.total)} matching tools · click a row for details`;
    $("#pg-info", view).textContent = `Page ${r.page} of ${pages}`;
    $("#prev", view).disabled = r.page <= 1; $("#next", view).disabled = r.page >= pages;
    const th = (k, l, num) => `<th class="sortable ${num ? "num" : ""}" data-k="${k}">${l}${st.sort === k ? (st.desc ? " ↓" : " ↑") : ""}</th>`;
    $("#tbl", view).innerHTML = `<table class="tbl"><thead><tr>${th("tool_name", "Tool")}<th>Source</th>${th("parameter_count", "Params", 1)}<th>Actual</th><th>Error type</th>${th("failure_rate", "Fail rate", 1)}${th("proba", "P(error)", 1)}<th>Risk</th></tr></thead><tbody>
      ${r.rows.map((t) => `<tr class="clickable" data-id="${esc(t.tool_id)}"><td><strong class="mono">${esc(t.tool_name)}</strong><div class="cell-desc" style="max-width:360px;-webkit-line-clamp:1">${esc(t.tool_description)}</div></td><td><span class="chip">${esc((SRC[t.source] || t.source).split(" ")[0])}</span></td><td class="num">${t.parameter_count}</td>
        <td>${t.error_occurred ? '<span style="color:var(--critical)">error</span>' : '<span style="color:var(--good)">success</span>'}</td><td class="text-2" style="font-size:12px">${esc(prettyType(t.error_type))}</td>
        <td class="num">${(t.failure_rate * 100).toFixed(0)}%</td><td class="num">${probCell(t.proba)}</td><td>${riskBadge(t.risk_level)}</td></tr>`).join("")}</tbody></table>`;
    $("#tbl", view).querySelectorAll("tr[data-id]").forEach((tr) => tr.addEventListener("click", () => toolDrawer(tr.dataset.id)));
    $("#tbl", view).querySelectorAll("th.sortable").forEach((h) => h.addEventListener("click", () => {
      st.desc = st.sort === h.dataset.k ? !st.desc : h.dataset.k !== "tool_name"; st.sort = h.dataset.k; st.page = 1; load();
    }));
  };
  let deb;
  ["q", "risk_level", "error_type", "server", "source", "split", "actual", "predicted"].forEach((k) => {
    const el = $(`#f-${k}`, view);
    el.addEventListener(k === "q" ? "input" : "change", () => { clearTimeout(deb); deb = setTimeout(() => { st[k] = el.value; st.page = 1; load(); }, k === "q" ? 250 : 0); });
  });
  $("#prev", view).onclick = () => { st.page--; load(); };
  $("#next", view).onclick = () => { st.page++; load(); };
  load();
}
