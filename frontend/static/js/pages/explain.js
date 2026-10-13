import { $, api, esc, familySplit, fmt, icon, recList, riskBadge, shapBars, errorState, skeleton, cssVar, FAMILY_COLORS, probCell, emptyState } from "../core.js";
import { plot, themed } from "../charts.js";

export default async function explain(view, params) {
  view.innerHTML = `<div class="page-head"><div><div class="eyebrow">Explainable AI</div><h1 class="mt8">Explainability</h1></div></div>${skeleton(420)}`;
  let sg, proj;
  try { [sg, proj] = await Promise.all([api("/feature-importance?top=40"), api("/projections")]); } catch (e) { view.innerHTML = errorState(e); return; }

  view.innerHTML = `
  <div class="page-head"><div><div class="eyebrow">Explainable AI</div><h1 class="mt8">Explainability</h1>
    <p>SHAP (${esc(sg.explainer)} explainer on ${esc(sg.model)}) decomposes every prediction into additive feature contributions in log-odds space. Text themes are LSA components labelled with their top terms.</p></div></div>

  <div class="grid g3">
    <div class="panel span2"><div class="panel-head"><div><h3>Global feature importance</h3><div class="sub">mean |SHAP| over the held-out test split · colour = feature family</div></div></div><div class="chart xl" id="gi"></div></div>
    <div class="panel stack">
      <div><h3>Importance by feature family</h3><div class="sub muted" style="font-size:12.5px">Share of total mean |SHAP|</div></div>
      <div class="chart sm" id="fam"></div>
      <div class="callout">${icon("bulb")}<div>Model base value <strong class="mono">${fmt(sg.base_value)}</strong> log-odds. Positive SHAP values push a tool towards <strong>agent error</strong>; negative values towards success.</div></div>
    </div>
  </div>

  <div class="panel mt16"><div class="panel-head"><div><h3>SHAP summary (beeswarm)</h3><div class="sub">Each dot is a test-set tool · x = impact on log-odds · colour = feature value (blue low → red high)</div></div></div><div class="chart xl" id="bee"></div></div>

  <div class="grid g2 mt16" style="align-items:start">
    <div class="panel">
      <div class="panel-head"><div><h3>${icon("search")} Local explanation</h3><div class="sub">Select any tool from the dataset</div></div></div>
      <div class="row"><input class="input" id="q" placeholder="Search tool name or description…" style="flex:1;min-width:200px" />
        <select class="select" id="qsplit" style="width:auto"><option value="test">test split</option><option value="">all splits</option></select></div>
      <div class="table-wrap mt12" style="max-height:360px" id="tool-list"></div>
    </div>
    <div class="panel" id="local">${emptyState("No tool selected", "Pick a tool on the left to see its individual SHAP explanation.", "spark")}</div>
  </div>

  <div class="panel mt16">
    <div class="panel-head"><div><h3>${icon("globe")} Semantic embedding projection</h3><div class="sub">all-MiniLM-L6-v2 tool embeddings projected to 2-D · hover for tool names</div></div>
      <div class="row"><div class="seg" id="proj-kind"><button data-k="umap" class="on">UMAP</button><button data-k="pca">PCA</button></div>
      <div class="seg" id="proj-color"><button data-c="y" class="on">Observed outcome</button><button data-c="p">Predicted risk</button></div></div></div>
    <div class="chart xl" id="proj"></div>
    <p class="muted" style="font-size:12px">PCA explains ${(proj.pca_explained_variance.reduce((a, b) => a + b, 0) * 100).toFixed(1)}% of embedding variance in 2-D. Clusters are semantic tool families (weather, finance, file systems…); failures are spread across clusters, which is why semantic features alone underperform the fused model.</p>
  </div>

  <div class="panel mt16"><div class="panel-head"><div><h3>Text themes (LSA components)</h3><div class="sub">Top TF-IDF terms of the components that appear in the global top-40</div></div></div>
    <div class="grid g4" id="topics"></div></div>`;

  themed(() => {
    const fam = (f) => cssVar(FAMILY_COLORS[f] || "--s8");
    const rows = sg.global.slice(0, 20).reverse();
    plot($("#gi", view), [{ type: "bar", orientation: "h", y: rows.map((r) => r.display_name), x: rows.map((r) => r.mean_abs_shap),
      marker: { color: rows.map((r) => fam(r.family)), cornerradius: 4 }, customdata: rows.map((r) => [r.family_label, r.mean_shap]),
      hovertemplate: "<b>%{y}</b><br>%{customdata[0]}<br>mean |SHAP| %{x:.4f}<br>mean SHAP %{customdata[1]:.4f}<extra></extra>" }],
    { margin: { l: 250, t: 8 }, xaxis: { title: { text: "mean |SHAP value|" } }, bargap: 0.28 });
    plot($("#fam", view), [{ type: "pie", hole: 0.62, labels: sg.families.map((f) => f.label), values: sg.families.map((f) => f.mean_abs_shap),
      marker: { colors: sg.families.map((f) => fam(f.family)), line: { color: cssVar("--surface-solid"), width: 2 } }, sort: false,
      textinfo: "percent", textposition: "inside", insidetextorientation: "horizontal", textfont: { color: "#fff", size: 11 }, hovertemplate: "%{label}: %{percent}<extra></extra>" }],
    { margin: { l: 10, r: 10, t: 10, b: 10 }, showlegend: true, legend: { orientation: "h", x: 0, y: -0.05, font: { size: 11 } } });
    const bees = sg.beeswarm.slice(0, 12).reverse();
    const data = bees.map((b, i) => ({
      type: "scattergl", mode: "markers", x: b.shap, y: b.shap.map(() => i + (Math.random() - 0.5) * 0.55), showlegend: false,
      marker: { size: 5.5, color: b.value_norm, cmin: 0, cmax: 1, colorscale: [[0, cssVar("--decrease")], [0.5, cssVar("--surface-3") || "#ccc"], [1, cssVar("--increase")]], opacity: 0.85,
        colorbar: i === 0 ? { title: { text: "feature value", side: "right", font: { size: 11 } }, tickvals: [0, 1], ticktext: ["low", "high"], thickness: 10, len: 0.6, outlinewidth: 0 } : undefined, showscale: i === 0 },
      customdata: b.value_norm, hovertemplate: `${esc(b.display_name)}<br>SHAP %{x:.3f}<br>value pct %{customdata:.2f}<extra></extra>`,
    }));
    plot($("#bee", view), data, { margin: { l: 250, t: 8 }, yaxis: { tickvals: bees.map((_, i) => i), ticktext: bees.map((b) => b.display_name), showgrid: false, zeroline: false },
      xaxis: { title: { text: "SHAP value (impact on log-odds of agent error)" }, zeroline: true, zerolinecolor: cssVar("--border-strong") } });
  });

  // ---------------------------------------------------------- projection
  let kind = "umap", colorBy = "y";
  const drawProj = () => {
    const pts = proj.points;
    const xy = (p) => (kind === "umap" ? p.umap : p.pca);
    if (colorBy === "y") {
      const groups = [[0, "Success", cssVar("--s1")], [1, "Agent error", cssVar("--s2")]];
      plot($("#proj", view), groups.map(([v, n, c]) => {
        const g = pts.filter((p) => p.y === v);
        return { type: "scattergl", mode: "markers", name: n, x: g.map((p) => xy(p)[0]), y: g.map((p) => xy(p)[1]),
          marker: { size: 5, color: c, opacity: 0.75 }, customdata: g.map((p) => [p.tool_name, p.server, p.error_type, p.proba]),
          hovertemplate: "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]} · P(err) %{customdata[3]:.2f}<extra></extra>" };
      }), { xaxis: { showticklabels: false, title: { text: `${kind.toUpperCase()} 1` } }, yaxis: { showticklabels: false, title: { text: `${kind.toUpperCase()} 2` } }, legend: { y: -0.08 } });
    } else {
      plot($("#proj", view), [{ type: "scattergl", mode: "markers", x: pts.map((p) => xy(p)[0]), y: pts.map((p) => xy(p)[1]),
        marker: { size: 5, color: pts.map((p) => p.proba), cmin: 0, cmax: 1, colorscale: [[0, cssVar("--decrease")], [0.5, cssVar("--surface-3") || "#ccc"], [1, cssVar("--increase")]], opacity: 0.8,
          colorbar: { title: { text: "P(error)", side: "right" }, thickness: 10, outlinewidth: 0 } },
        customdata: pts.map((p) => [p.tool_name, p.server, p.error_type, p.proba]), hovertemplate: "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>P(err) %{customdata[3]:.2f}<extra></extra>" }],
      { xaxis: { showticklabels: false, title: { text: `${kind.toUpperCase()} 1` } }, yaxis: { showticklabels: false, title: { text: `${kind.toUpperCase()} 2` } } });
    }
  };
  themed(drawProj);
  view.querySelectorAll("#proj-kind button").forEach((b) => b.addEventListener("click", () => { kind = b.dataset.k; view.querySelectorAll("#proj-kind button").forEach((x) => x.classList.toggle("on", x === b)); drawProj(); }));
  view.querySelectorAll("#proj-color button").forEach((b) => b.addEventListener("click", () => { colorBy = b.dataset.c; view.querySelectorAll("#proj-color button").forEach((x) => x.classList.toggle("on", x === b)); drawProj(); }));

  // -------------------------------------------------------------- topics
  const topicIds = [...new Set(sg.global.filter((g) => g.feature.startsWith("text:lsa_")).map((g) => String(parseInt(g.feature.split("_")[1], 10))))].slice(0, 8);
  $("#topics", view).innerHTML = topicIds.map((k) => `<div class="panel" style="padding:14px;box-shadow:none"><div class="eyebrow">Theme #${k}</div>
    <div class="row mt8" style="gap:6px">${(sg.lsa_topics[k] || []).map((t) => `<span class="chip">${esc(t)}</span>`).join("")}</div></div>`).join("") || '<p class="muted">No text themes in the top features.</p>';

  // --------------------------------------------------------- local picker
  const list = $("#tool-list", view);
  const loadList = async () => {
    const q = $("#q", view).value.trim(), s = $("#qsplit", view).value;
    const r = await api(`/dataset/tools?page_size=40&sort=proba&desc=true${s ? `&split=${s}` : ""}${q ? `&q=${encodeURIComponent(q)}` : ""}`);
    list.innerHTML = `<table class="tbl"><thead><tr><th>Tool</th><th class="num">P(error)</th></tr></thead><tbody>${r.rows.map((t) => `
      <tr class="clickable" data-id="${esc(t.tool_id)}"><td><strong class="mono" style="word-break:break-all">${esc(t.tool_name)}</strong><div class="muted" style="font-size:11.5px">actual: ${t.error_occurred ? '<span style="color:var(--critical)">error</span>' : '<span style="color:var(--good)">success</span>'}</div></td>
      <td class="num">${probCell(t.proba)}</td></tr>`).join("")}</tbody></table>`;
    list.querySelectorAll("tr[data-id]").forEach((tr) => tr.addEventListener("click", () => showLocal(tr.dataset.id)));
  };
  let deb;
  $("#q", view).addEventListener("input", () => { clearTimeout(deb); deb = setTimeout(loadList, 250); });
  $("#qsplit", view).addEventListener("change", loadList);
  await loadList();

  async function showLocal(id) {
    const box = $("#local", view);
    box.innerHTML = skeleton(380);
    const t = await api(`/tools/${encodeURIComponent(id)}`);
    const e = t.explanation || {};
    box.innerHTML = `<div class="panel-head"><div><h3 class="mono">${esc(t.tool_name)}</h3><div class="sub">${esc(t.server_name)} · ${esc(t.source)} · ${esc(t.split)} split</div></div>${riskBadge(t.risk_level)}</div>
      <p class="text-2" style="font-size:13px">${esc(t.tool_description)}</p>
      <div class="kv mt12"><div><div class="k">P(error)</div><div class="v">${(t.proba * 100).toFixed(1)}%</div></div><div><div class="k">Actual</div><div class="v">${t.error_occurred ? "error" : "success"}</div></div>
        <div><div class="k">Failure rate</div><div class="v">${(t.failure_rate * 100).toFixed(0)}%</div></div><div><div class="k">Mode</div><div class="v" style="font-size:12px">${esc(t.error_type.replaceAll("_", " "))}</div></div></div>
      <div class="mt16">${shapBars(e.top)}</div>
      <div class="mt16">${familySplit(e.families)}</div>
      <h3 class="mt16" style="margin-bottom:10px">${icon("wand")} Recommendations</h3>${recList(e.recommendations)}`;
  }
  const first = list.querySelector("tr[data-id]");
  if (params.get("tool")) showLocal(params.get("tool"));
  else if (first) showLocal(first.dataset.id);
}
