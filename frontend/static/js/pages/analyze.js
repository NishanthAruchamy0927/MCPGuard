import { $, api, bindTabs, tabs, esc, familySplit, fmt, icon, recList, riskBadge, riskColor, shapBars, tip, toast, emptyState, RISK_META, cssVar } from "../core.js";

const PRESETS = {
  spec: { label: "Brief example", dot: "--s1", tool_name: "search_customer", description: "Searches for a customer using customer ID.",
    input_schema: { type: "object", properties: { customer_id: { type: "string", description: "Unique customer identifier" } }, required: ["customer_id"] } },
  good: { label: "Well documented", dot: "--good", tool_name: "get_current_weather",
    description: "Get the current weather for a city. Use this when the user asks about present conditions (not forecasts). Returns temperature, humidity and a short summary.",
    input_schema: { type: "object", properties: {
      location: { type: "string", description: "City and state or country in the format 'City, State', e.g. 'San Francisco, CA' or 'Paris, France'." },
      unit: { type: "string", enum: ["celsius", "fahrenheit"], description: "Temperature unit. One of 'celsius' or 'fahrenheit'. Defaults to 'celsius'.", default: "celsius" } },
      required: ["location"] } },
  vague: { label: "Vague & untyped", dot: "--serious", tool_name: "process", description: "Handles data operations as needed.",
    input_schema: { type: "object", properties: { data: {}, mode: {}, options: { type: "object" }, target: {}, flag: {} } } },
  nested: { label: "Deeply nested", dot: "--critical", tool_name: "book_itinerary",
    description: "Books an itinerary.",
    input_schema: { type: "object", properties: {
      trip: { type: "object", properties: {
        legs: { type: "array", items: { type: "object", properties: { from: { type: "string" }, to: { type: "string" }, date: { type: "string" },
          passengers: { type: "array", items: { type: "object", properties: { name: { type: "string" }, age: { type: "integer" }, seat: { type: "object", properties: { row: { type: "integer" }, letter: { type: "string" } } } } } } } } },
        budget: { type: "number" }, currency: { type: "string" } } },
      preferences: { type: "object", properties: { cabin: { type: "string" }, meal: { type: "string" }, loyalty: { type: "object" } } },
      notify: { type: "boolean" }, notes: { type: "string" }, channel: { type: "string" }, priority: { type: "string" } }, required: ["trip"] } },
};

function gauge(p, level) {
  const R = 100, C = Math.PI * R;
  const color = riskColor(p);
  const bands = [["--good", 0, 0.25], ["--warning", 0.25, 0.5], ["--serious", 0.5, 0.75], ["--critical", 0.75, 1]];
  const arc = (a, b) => {
    const pt = (t) => { const ang = Math.PI * (1 - t); return [120 + R * Math.cos(ang), 120 - R * Math.sin(ang)]; };
    const [x1, y1] = pt(a), [x2, y2] = pt(b);
    return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${R} ${R} 0 0 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`;
  };
  const ang = Math.PI * (1 - p);
  const nx = 120 + R * Math.cos(ang), ny = 120 - R * Math.sin(ang);
  return `<div class="gauge" role="img" aria-label="Risk score ${(p * 100).toFixed(1)} percent, ${level}">
    <svg viewBox="0 0 240 140">
      ${bands.map(([c, a, b]) => `<path d="${arc(a + 0.006, b - 0.006)}" stroke="var(${c})" stroke-opacity=".22" stroke-width="16" fill="none" stroke-linecap="butt"/>`).join("")}
      <path d="${arc(0, 1)}" stroke="${color}" stroke-width="16" fill="none" stroke-linecap="round" stroke-dasharray="${C}" stroke-dashoffset="${C}" style="transition:stroke-dashoffset 1s cubic-bezier(.2,.8,.2,1)" class="g-arc" data-target="${C * (1 - p)}"/>
      <circle cx="${nx.toFixed(2)}" cy="${ny.toFixed(2)}" r="9" fill="${color}" stroke="var(--surface-solid)" stroke-width="3" class="g-knob" style="opacity:0;transition:opacity .4s .8s"/>
      ${[0, 25, 50, 75, 100].map((t) => { const a = Math.PI * (1 - t / 100); return `<text x="${120 + (R + 16) * Math.cos(a)}" y="${124 - (R + 16) * Math.sin(a)}" font-size="9" fill="var(--muted)" text-anchor="middle" font-family="JetBrains Mono">${t}</text>`; }).join("")}
    </svg>
    <div class="g-value"><div class="g-num" style="color:${color}">${(p * 100).toFixed(1)}<small>%</small></div><div class="g-cap">Risk score</div></div>
  </div>`;
}

const FEATURE_KEYS = ["ambiguity_score", "missing_constraint_score", "parameter_complexity_score", "schema_complexity_score", "description_quality_score", "instruction_clarity_score",
  "desc_token_count", "parameter_count", "required_parameter_count", "nested_schema_depth", "param_described_ratio", "hinted_parameter_ratio",
  "enum_count", "untyped_parameter_count", "example_presence"];

function resultHTML(r) {
  const why = r.risk_factors.length ? r.risk_factors.slice(0, 4) : ["No dominant risk drivers — most contributions reduce risk."];
  const recs = r.recommendations || [];
  const summary = `
    <div class="result-hero">
      ${gauge(r.probability, r.risk_level)}
      <div class="verdict">
        ${riskBadge(r.risk_level, true)}
        <h2>${r.prediction ? "Likely to cause agent errors" : "Likely to work reliably"}</h2>
        <p class="text-2"><b style="color:${riskColor(r.probability)}">${(r.probability * 100).toFixed(1)}%</b> estimated chance that agents fail when calling <code>${esc(r.tool_name)}</code> · threshold ${(r.decision_threshold * 100).toFixed(1)}%.</p>
      </div>
    </div>
    <div class="stat-strip mt16">
      <div><div class="k">Prediction</div><div class="v">${r.prediction ? "ERROR" : "SUCCESS"}</div></div>
      <div><div class="k">Confidence ${tip("max(p, 1 − p) — distance from an undecided 50/50 call.")}</div><div class="v">${(r.confidence * 100).toFixed(1)}%</div></div>
      <div><div class="k">Latency</div><div class="v">${fmt(r.latency_ms, 0)} ms</div></div>
    </div>
    <h3 class="mt24">Main reasons</h3>
    <ul class="reasons mt12">${why.map((w) => `<li>${esc(w)}</li>`).join("")}</ul>
    ${r.schema_issues.length ? `<div class="callout warn mt16">${icon("alert")}<div><strong>Schema issues:</strong> ${r.schema_issues.map(esc).join(", ")}</div></div>` : ""}
    ${recs.length ? `<div class="callout mt16">${icon("wand")}<div><strong>Top fix:</strong> ${esc(recs[0].title)} — ${esc(recs[0].detail)}</div></div>` : ""}`;
  const fixes = recs.length ? `<p class="muted" style="font-size:12.5px;margin-bottom:14px">Ordered by priority; "SHAP driver" marks fixes that target this tool's own top risk factors.</p>${recList(recs)}`
    : emptyState("Nothing to fix", "No rule fired for this tool.", "check");
  const shap = `<p class="muted" style="font-size:12.5px;margin-bottom:14px">Top ${r.top_features.length} feature contributions to the log-odds of an agent error.</p>${shapBars(r.top_features)}
    <h3 class="mt24" style="font-size:14px">By feature family</h3><div class="mt12">${familySplit(r.family_contributions)}</div>`;
  const feats = `<p class="muted" style="font-size:12.5px;margin-bottom:14px">Interpretable scores and schema statistics extracted from your input.</p>
    <div class="feature-grid">${FEATURE_KEYS.map((k) => `<div><span>${esc(k.replaceAll("_", " "))}</span><b>${fmt(r.feature_values[k], k.includes("count") || k.includes("depth") ? 0 : 2)}</b></div>`).join("")}</div>`;
  return `<div class="panel">${tabs("res", [["sum", "Summary", summary], ["fix", `Fixes (${recs.length})`, fixes], ["shap", "SHAP", shap], ["feat", "Features", feats]])}</div>`;
}

export default async function analyze(view, params) {
  view.innerHTML = `
  <div class="page-head"><div><div class="eyebrow">Real-time inference</div><h1 class="mt8">Tool Risk Analyzer</h1>
    <p>Paste an MCP tool definition. MCPGuard validates it, extracts text, semantic and schema features, scores it with the trained model and explains the result.</p></div></div>
  <div class="analyzer">
    <form class="panel stack" id="tool-form" novalidate>
      <div><h3>${icon("box")} Tool definition</h3><p class="muted mt8" style="font-size:12.5px">Start from an example or paste your own.</p></div>
      <div class="presets">${Object.entries(PRESETS).map(([k, p]) => `<button type="button" class="preset" data-preset="${k}"><span class="sw" style="background:var(${p.dot})"></span>${p.label}</button>`).join("")}</div>
      <div class="grid g2">
        <div class="field"><label for="f-name">Tool name <span class="hint">required</span></label><input class="input mono" id="f-name" placeholder="e.g. search_customer" maxlength="200" /><div class="field-error" id="e-tool_name"></div></div>
        <div class="field"><label for="f-server">Server name <span class="hint">optional</span></label><input class="input" id="f-server" placeholder="e.g. crm-server" /></div>
      </div>
      <div class="field"><label for="f-desc">Tool description <span class="hint" id="desc-count">0 chars</span></label>
        <textarea class="textarea" id="f-desc" rows="4" placeholder="What the tool does, when to use it, what it returns…"></textarea><div class="field-error" id="e-description"></div></div>
      <div class="field"><label for="f-schema"><span>Input schema (JSON) ${tip("The MCP inputSchema: a JSON-schema object with type 'object', 'properties' and optional 'required'.")}</span>
        <span class="row" style="gap:8px"><span class="json-status" id="json-status"></span><button type="button" class="btn btn-sm btn-ghost" id="fmt-btn">${icon("braces")}Format</button></span></label>
        <textarea class="textarea code-area" id="f-schema" spellcheck="false"></textarea><div class="field-error" id="e-input_schema"></div></div>
      <div class="grid g2">
        <div class="field"><label for="f-cat">Tool category <span class="hint">optional</span></label><select class="select" id="f-cat"><option value="">—</option><option>search</option><option>data access</option><option>file system</option><option>communication</option><option>developer tools</option><option>finance</option><option>travel</option><option>other</option></select></div>
        <div class="field"><label for="f-ctx">Execution context <span class="hint">optional</span></label><input class="input" id="f-ctx" placeholder="e.g. customer-support agent" /></div>
      </div>
      <div class="row" style="justify-content:space-between">
        <span class="muted" style="font-size:12px">${icon("shield")} Inputs are validated server-side; descriptions are never stored (only a hash).</span>
        <button class="btn btn-primary btn-lg" id="go" type="submit">${icon("scan")}Analyze Risk</button>
      </div>
    </form>
    <div class="stack" id="result"><div class="panel">${emptyState("No analysis yet", "Choose a preset or paste your own tool, then click Analyze Risk to get a risk score, SHAP explanation and recommendations.", "gauge")}</div></div>
  </div>`;

  const name = $("#f-name", view), desc = $("#f-desc", view), schema = $("#f-schema", view), status = $("#json-status", view);
  const load = (k) => {
    const p = PRESETS[k];
    name.value = p.tool_name; desc.value = p.description; schema.value = JSON.stringify(p.input_schema, null, 2);
    validate();
  };
  const validate = () => {
    $("#desc-count", view).textContent = `${desc.value.length} chars`;
    if (!schema.value.trim()) { status.className = "json-status"; status.textContent = ""; schema.classList.remove("invalid"); return true; }
    try {
      const v = JSON.parse(schema.value);
      if (typeof v !== "object" || Array.isArray(v) || v === null) throw new Error("must be a JSON object");
      status.className = "json-status ok"; status.innerHTML = `${icon("check")}valid JSON`; schema.classList.remove("invalid"); return true;
    } catch (e) {
      status.className = "json-status bad"; status.innerHTML = `${icon("alert")}${esc(e.message.split("\n")[0].slice(0, 60))}`; schema.classList.add("invalid"); return false;
    }
  };
  view.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => load(b.dataset.preset)));
  [desc, schema].forEach((el) => el.addEventListener("input", validate));
  $("#fmt-btn", view).addEventListener("click", () => { try { schema.value = JSON.stringify(JSON.parse(schema.value), null, 2); validate(); } catch (e) { validate(); } });
  schema.addEventListener("keydown", (e) => { if (e.key === "Tab") { e.preventDefault(); const s = schema.selectionStart; schema.setRangeText("  ", s, schema.selectionEnd, "end"); } });
  load(PRESETS[params && params.get("preset")] ? params.get("preset") : "spec");
  if (params && params.get("run") === "1") setTimeout(() => $("#tool-form", view).requestSubmit(), 0);

  $("#tool-form", view).addEventListener("submit", async (e) => {
    e.preventDefault();
    view.querySelectorAll(".field-error").forEach((x) => (x.textContent = ""));
    [name, desc, schema].forEach((x) => x.classList.remove("invalid"));
    let parsed = {};
    if (!name.value.trim()) { $("#e-tool_name", view).textContent = "Tool name is required."; name.classList.add("invalid"); }
    if (!desc.value.trim()) { $("#e-description", view).textContent = "Description is required."; desc.classList.add("invalid"); }
    if (schema.value.trim()) { try { parsed = JSON.parse(schema.value); } catch (err) { $("#e-input_schema", view).textContent = "Input schema is not valid JSON."; schema.classList.add("invalid"); } }
    if (view.querySelector(".invalid")) return;
    const btn = $("#go", view);
    btn.disabled = true; btn.innerHTML = `<span class="spinner"></span>Analyzing…`;
    const out = $("#result", view);
    out.innerHTML = `<div class="skeleton" style="height:520px"></div>`;
    try {
      const r = await api("/predict", { method: "POST", body: { tool_name: name.value, description: desc.value, input_schema: parsed,
        server_name: $("#f-server", view).value || null, tool_category: $("#f-cat", view).value || null, execution_context: $("#f-ctx", view).value || null, explain: true } });
      out.innerHTML = resultHTML(r);
      bindTabs(out, "res");
      requestAnimationFrame(() => requestAnimationFrame(() => {
        out.querySelectorAll(".g-arc").forEach((a) => (a.style.strokeDashoffset = a.dataset.target));
        out.querySelectorAll(".g-knob").forEach((k) => (k.style.opacity = 1));
      }));
      if (window.innerWidth < 1100) out.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      out.innerHTML = `<div class="panel">${emptyState("Validation failed", err.message, "alert")}</div>`;
      (err.data && err.data.errors || []).forEach((fe) => {
        const el = $(`#e-${fe.field}`, view);
        if (el) el.textContent = fe.message;
      });
      toast(err.data && err.data.errors ? err.data.errors.map((x) => `${x.field}: ${x.message}`).join("; ") : err.message);
    } finally {
      btn.disabled = false; btn.innerHTML = `${icon("scan")}Analyze Risk`;
    }
  });
}
