import { $, api, downloadText, esc, icon, int, riskBadge, toCSV, toast, emptyState, fmt } from "../core.js";

const SAMPLE = [
  { tool_name: "get_current_weather", description: "Get the current weather for a city. Use for present conditions, not forecasts.", input_schema: { type: "object", properties: { location: { type: "string", description: "City and state, e.g. 'San Francisco, CA'" }, unit: { type: "string", enum: ["celsius", "fahrenheit"] } }, required: ["location"] } },
  { tool_name: "process", description: "Handles data operations as needed.", input_schema: { type: "object", properties: { data: {}, mode: {}, options: { type: "object" } } } },
  { tool_name: "create_issue", description: "Create a new issue in a GitHub repository", input_schema: { type: "object", properties: { owner: { type: "string" }, repo: { type: "string" }, title: { type: "string" }, body: { type: "string" }, labels: { type: "array", items: { type: "string" } } }, required: ["owner", "repo", "title"] } },
  { tool_name: "convert_time", description: "Convert time between timezones", input_schema: { type: "object", properties: { source_timezone: { type: "string", description: "Source IANA timezone name (e.g., 'America/New_York')." }, time: { type: "string", description: "Time to convert in 24-hour format (HH:MM)" }, target_timezone: { type: "string", description: "Target IANA timezone name (e.g., 'Asia/Tokyo')." } }, required: ["source_timezone", "time", "target_timezone"] } },
  { tool_name: "book_flight", description: "Books a flight.", input_schema: { type: "object", properties: { from: { type: "string" }, to: { type: "string" }, date: { type: "string" }, passengers: { type: "array", items: { type: "object", properties: { name: { type: "string" }, age: { type: "integer" } } } }, class: { type: "string" } } } },
  { tool_name: "slack_post_message", description: "Post a new message to a Slack channel", input_schema: { type: "object", properties: { channel_id: { type: "string", description: "The ID of the channel to post to" }, text: { type: "string", description: "The message text to post" } }, required: ["channel_id", "text"] } },
];
const COLS = ["tool_name", "risk_score", "risk_level", "predicted_error", "confidence", "top_risk_factor"];

export default async function batch(view) {
  view.innerHTML = `
  <div class="page-head"><div><div class="eyebrow">Batch inference</div><h1 class="mt8">Batch Analyzer</h1>
    <p>Upload a CSV or JSON catalogue of MCP tools and score every tool in one pass. Invalid rows are reported individually without failing the batch.</p></div></div>
  <div class="grid g3">
    <div class="panel span2">
      <label class="dropzone" id="dz">
        <input type="file" id="file" accept=".csv,.json,application/json,text/csv" aria-label="Upload CSV or JSON file" />
        <div class="big-icon">${icon("upload")}</div>
        <h3>Drop a .csv or .json file here</h3>
        <p class="muted">or <span class="link-like">click to choose a file</span> · max 500 tools · 10 MB</p>
        <div class="row" style="justify-content:center"><span class="chip">CSV: tool_name, description, input_schema</span><span class="chip">JSON: [ {tool_name, description, input_schema} ]</span></div>
      </label>
      <div class="file-status" id="file-status" hidden></div>
    </div>
    <div class="panel stack">
      <h3>${icon("file")} Sample files</h3>
      <p class="text-2">Six example tools — from well documented to vague and deeply nested — in both supported formats.</p>
      <button class="btn" id="dl-csv">${icon("download")}Sample CSV</button>
      <button class="btn" id="dl-json">${icon("download")}Sample JSON</button>
      <button class="btn btn-primary" id="run-sample">${icon("scan")}Analyze the sample now</button>
    </div>
  </div>
  <div id="batch-out" class="stack mt16"></div>`;

  const sampleCSV = () => toCSV(SAMPLE.map((t) => ({ ...t, input_schema: JSON.stringify(t.input_schema) })), ["tool_name", "description", "input_schema"]);
  $("#dl-csv", view).onclick = () => downloadText("mcpguard_sample_tools.csv", sampleCSV());
  $("#dl-json", view).onclick = () => downloadText("mcpguard_sample_tools.json", JSON.stringify(SAMPLE, null, 2), "application/json");

  const out = $("#batch-out", view);
  const render = (res, label) => {
    const ok = res.results.filter((r) => !r.error);
    const tot = ok.length || 1;
    const sevColor = { LOW: "--good", MEDIUM: "--warning", HIGH: "--serious", CRITICAL: "--critical" };
    let sortKey = "risk_score", asc = false;
    const table = () => {
      const rows = [...res.results].sort((a, b) => {
        const x = a[sortKey] ?? -1, y = b[sortKey] ?? -1;
        return (x > y ? 1 : x < y ? -1 : 0) * (asc ? 1 : -1);
      });
      return `<table class="tbl"><thead><tr>
        <th class="num">#</th><th class="sortable" data-k="tool_name">Tool</th><th class="sortable num" data-k="risk_score">Risk score</th><th>Risk level</th>
        <th class="sortable num" data-k="predicted_error">Predicted</th><th class="sortable num" data-k="confidence">Confidence</th><th>Top risk factor</th></tr></thead>
        <tbody>${rows.map((r) => r.error ? `<tr><td class="num">${r.row}</td><td><strong class="mono">${esc(r.tool_name || "—")}</strong></td><td colspan="5"><span class="risk risk-CRITICAL">${icon("x")}INVALID</span> <span class="muted">${esc(r.error)}</span></td></tr>`
          : `<tr><td class="num">${r.row}</td><td><strong class="mono">${esc(r.tool_name)}</strong></td><td class="num">${fmt(r.risk_score, 1)}%</td><td>${riskBadge(r.risk_level)}</td>
          <td class="num">${r.predicted_error ? '<span style="color:var(--critical)">error</span>' : '<span style="color:var(--good)">success</span>'}</td><td class="num">${(r.confidence * 100).toFixed(1)}%</td>
          <td class="text-2">${esc(r.top_risk_factor || "—")}</td></tr>`).join("")}</tbody></table>`;
    };
    out.innerHTML = `
    <div class="grid g4">${["LOW", "MEDIUM", "HIGH", "CRITICAL"].map((k) => `<div class="panel kpi"><div class="label">${riskBadge(k)}</div><div class="value">${int(res.summary[k])}</div><div class="foot">${((res.summary[k] / tot) * 100).toFixed(0)}% of valid tools</div></div>`).join("")}</div>
    <div class="panel">
      <div class="panel-head"><div><h3>Results · ${esc(label)}</h3><div class="sub">${res.n_ok} scored · ${res.n_failed} invalid · click headers to sort</div></div>
        <button class="btn btn-primary btn-sm" id="dl-res">${icon("download")}Download results (CSV)</button></div>
      <div class="dist-bar">${["LOW", "MEDIUM", "HIGH", "CRITICAL"].filter((k) => res.summary[k]).map((k) => `<span style="width:${(res.summary[k] / tot) * 100}%;background:var(${sevColor[k]})" title="${k}: ${res.summary[k]}">${res.summary[k]}</span>`).join("")}</div>
      <div class="table-wrap mt16" id="res-table">${table()}</div>
    </div>`;
    const bind = () => out.querySelectorAll("th.sortable").forEach((th) => th.addEventListener("click", () => {
      const k = th.dataset.k; asc = sortKey === k ? !asc : k === "tool_name"; sortKey = k;
      $("#res-table", out).innerHTML = table(); bind();
    }));
    bind();
    $("#dl-res", out).onclick = () => downloadText("mcpguard_predictions.csv", toCSV(res.results.map((r) => ({ ...r, prediction: r.predicted_error })), [...COLS.slice(0, 3), "prediction", "confidence", "top_risk_factor", "error"]));
  };

  const statusEl = $("#file-status", view);
  let lastFile = null;
  const kb = (n) => (n < 1024 * 1024 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1024 / 1024).toFixed(2)} MB`);
  const setStatus = (f, state, msg) => {
    if (!f) { statusEl.hidden = true; statusEl.innerHTML = ""; return; }
    const ic = { busy: '<span class="spinner dark"></span>', ok: icon("check"), bad: icon("alert") }[state];
    statusEl.hidden = false;
    statusEl.className = `file-status ${state}`;
    statusEl.innerHTML = `<div class="fs-icon">${icon("file")}</div>
      <div class="fs-main"><div class="fs-name" title="${esc(f.name)}">${esc(f.name)}</div><div class="fs-meta">${kb(f.size)} · <span class="fs-state">${ic}${esc(msg)}</span></div></div>
      <div class="row" style="gap:6px">${state !== "busy" && f instanceof File && /\.(csv|json)$/i.test(f.name) ? `<button type="button" class="btn btn-sm" id="fs-retry">${icon("scan")}Analyze again</button>` : ""}
        <button type="button" class="btn btn-sm btn-ghost" id="fs-clear" aria-label="Clear file">${icon("x")}Clear</button></div>`;
    const r = $("#fs-retry", statusEl); if (r) r.onclick = () => send(lastFile);
    $("#fs-clear", statusEl).onclick = () => { setStatus(null); out.innerHTML = ""; lastFile = null; };
  };

  const run = async (form, label, f) => {
    out.innerHTML = `<div class="skeleton" style="height:120px"></div><div class="skeleton" style="height:320px"></div>`;
    if (f) setStatus(f, "busy", "Uploading and scoring…");
    try {
      const res = form instanceof FormData ? await api("/batch-predict/file", { method: "POST", form }) : await api("/batch-predict", { method: "POST", body: form });
      render(res, label);
      const msg = `Scored ${res.n_ok} tool(s)${res.n_failed ? ` · ${res.n_failed} invalid row(s)` : ""}`;
      if (f) setStatus(f, res.n_ok ? "ok" : "bad", msg);
      toast(msg, "ok");
      out.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) {
      const msg = (e.data && e.data.errors) ? e.data.errors.map((x) => `${x.field}: ${x.message}`).join("; ") : e.message;
      out.innerHTML = `<div class="panel">${emptyState("Batch failed", msg, "alert")}</div>`;
      if (f) setStatus(f, "bad", msg);
      toast(msg);
    }
  };
  $("#run-sample", view).onclick = () => { setStatus(null); run({ tools: SAMPLE }, "sample catalogue"); };
  const dz = $("#dz", view), file = $("#file", view);
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, () => dz.classList.remove("drag")));
  dz.addEventListener("drop", (e) => { e.preventDefault(); if (e.dataTransfer.files[0]) send(e.dataTransfer.files[0]); });
  file.addEventListener("change", () => file.files[0] && send(file.files[0]));
  function send(f) {
    if (!f) return;
    lastFile = f;
    const ext = (f.name.split(".").pop() || "").toLowerCase();
    if (["csv", "json"].includes(ext) === false || f.size > 10 * 1024 * 1024 || !f.size) out.innerHTML = "";
    if (!["csv", "json"].includes(ext)) { setStatus(f, "bad", "Unsupported file type — choose a .csv or .json file."); toast("Unsupported file type — choose a .csv or .json file."); file.value = ""; return; }
    if (f.size > 10 * 1024 * 1024) { setStatus(f, "bad", "File is larger than 10 MB."); toast("File is larger than 10 MB."); file.value = ""; return; }
    if (!f.size) { setStatus(f, "bad", "File is empty."); toast("File is empty."); file.value = ""; return; }
    const fd = new FormData();
    fd.append("file", f);
    run(fd, f.name, f);
    file.value = "";
  }
}
