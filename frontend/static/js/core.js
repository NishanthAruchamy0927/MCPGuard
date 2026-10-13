import { icon, hydrateIcons } from "./icons.js";

export { icon, hydrateIcons };

// ---------------------------------------------------------------- api
const cache = new Map();
export async function api(path, { method = "GET", body, form, cacheable = true } = {}) {
  if (method === "GET" && cacheable && cache.has(path)) return cache.get(path);
  const opts = { method, headers: { Accept: "application/json" } };
  if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
  if (form) opts.body = form;
  const p = fetch(path, opts).then(async (r) => {
    const ct = r.headers.get("content-type") || "";
    const data = ct.includes("json") ? await r.json() : await r.text();
    if (!r.ok) {
      const err = new Error((data && data.detail) || `Request failed (${r.status})`);
      err.status = r.status; err.data = data; throw err;
    }
    return data;
  });
  if (method === "GET" && cacheable) { cache.set(path, p); p.catch(() => cache.delete(path)); }
  return p;
}
export const clearCache = () => cache.clear();

// ------------------------------------------------------------ helpers
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
export const pct = (v, d = 1) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(d)}%`);
export const int = (v) => (v === null || v === undefined ? "—" : Number(v).toLocaleString());
export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
export const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export const tip = (text) => `<span class="tip" tabindex="0" data-tip="${esc(text)}">?</span>`;

export const RISK_META = {
  LOW: { icon: "check", label: "Low risk", color: "--good" },
  MEDIUM: { icon: "info", label: "Medium risk", color: "--warning" },
  HIGH: { icon: "alert", label: "High risk", color: "--serious" },
  CRITICAL: { icon: "octagon", label: "Critical risk", color: "--critical" },
};
export function riskBadge(level, long = false) {
  if (!level) return '<span class="muted">—</span>';
  const m = RISK_META[level] || RISK_META.LOW;
  return `<span class="risk risk-${level}">${icon(m.icon)}${long ? m.label.toUpperCase() : level}</span>`;
}
export function riskColor(p) {
  const s = p * 100;
  if (s >= 75) return cssVar("--critical");
  if (s >= 50) return cssVar("--serious");
  if (s >= 25) return cssVar("--warning");
  return cssVar("--good");
}
export function probCell(p) {
  if (p === null || p === undefined) return '<span class="muted">—</span>';
  return `<span class="prob"><span class="track"><span style="width:${(p * 100).toFixed(1)}%;background:${riskColor(p)}"></span></span>${(p * 100).toFixed(1)}%</span>`;
}
export const outcomeTag = (o) => `<span class="outcome o-${o}">${o}</span>`;
export const prettyType = (t) => String(t || "").replaceAll("_", " ").toLowerCase().replace(/^\w/, (c) => c.toUpperCase());

export function skeleton(h = 120, n = 1) {
  return Array.from({ length: n }, () => `<div class="skeleton" style="height:${h}px"></div>`).join("");
}
export function emptyState(title, text, ic = "info") {
  return `<div class="empty"><div class="big-icon">${icon(ic)}</div><h3>${esc(title)}</h3><p>${esc(text)}</p></div>`;
}
export function errorState(err) {
  const msg = err && err.status === 503 ? err.message : (err && err.message) || "Something went wrong.";
  return `<div class="panel">${emptyState("Data unavailable", msg, "alert")}</div>`;
}

// ------------------------------------------------------------- toasts
export function toast(msg, kind = "error") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.innerHTML = `${icon(kind === "ok" ? "check" : "alert")}<div>${esc(msg)}</div>`;
  $("#toasts").appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; el.style.transition = "opacity .3s"; setTimeout(() => el.remove(), 320); }, 4200);
}

// ------------------------------------------------------------- drawer
let lastFocus = null;
export function openDrawer(title, html) {
  lastFocus = document.activeElement;
  $("#drawer-title").textContent = title;
  $("#drawer-body").innerHTML = html;
  hydrateIcons($("#drawer-body"));
  $("#drawer-scrim").hidden = false;
  const d = $("#drawer");
  d.classList.add("open");
  d.setAttribute("aria-hidden", "false");
  $("#drawer-close").focus();
  return $("#drawer-body");
}
export function closeDrawer() {
  $("#drawer").classList.remove("open");
  $("#drawer").setAttribute("aria-hidden", "true");
  $("#drawer-scrim").hidden = true;
  if (lastFocus) lastFocus.focus();
}

// ---------------------------------------------------- shared renderers
export function shapBars(contribs, { max } = {}) {
  if (!contribs || !contribs.length) return emptyState("No explanation", "SHAP values are unavailable for this model.");
  const m = max || Math.max(...contribs.map((c) => Math.abs(c.shap))) || 1;
  return `<div class="shap-bars">${contribs.map((c) => {
    const w = (Math.abs(c.shap) / m) * 50;
    const pos = c.shap > 0;
    return `<div class="shap-row" title="${esc(c.feature)} = ${fmt(c.value, 3)}">
      <div class="name">${esc(c.display_name)}<small>${esc(c.family_label)} · value ${fmt(c.value, 2)}</small></div>
      <div class="track"><div class="fill ${pos ? "pos" : "neg"}" style="width:${w}%"></div></div>
      <div class="val ${pos ? "pos" : "neg"}">${pos ? "+" : "−"}${Math.abs(c.shap).toFixed(3)}</div></div>`;
  }).join("")}</div>
  <div class="legend-row mt12"><span><i class="sw" style="background:var(--increase)"></i>increases error risk</span><span><i class="sw" style="background:var(--decrease)"></i>decreases error risk</span><span class="muted">SHAP values in log-odds</span></div>`;
}

export function recList(recs) {
  return `<div class="stack" style="gap:10px">${(recs || []).map((r, i) => `
    <div class="rec ${r.priority}"><div class="num">${i + 1}</div><div>
      <h4>${esc(r.title)} <span class="chip">${r.priority}</span>${r.model_driver ? '<span class="driver">SHAP driver</span>' : ""}</h4>
      <p>${esc(r.detail)}</p></div></div>`).join("")}</div>`;
}

export const FAMILY_COLORS = { text: "--s1", emb: "--s2", schema: "--s3", lexical: "--s4", risk: "--s7", tfidf: "--s5" };
export const FAMILY_LABELS = { text: "Text (LSA)", emb: "Semantic embedding", schema: "Schema structure", lexical: "Lexical / naming", risk: "Risk scores", tfidf: "TF-IDF" };
export function familySplit(fam) {
  const entries = Object.entries(fam || {});
  const tot = entries.reduce((a, [, v]) => a + Math.abs(v), 0) || 1;
  return `<div class="family-split">${entries.map(([k, v]) => `<span title="${esc(FAMILY_LABELS[k] || k)}: ${v >= 0 ? "+" : ""}${v.toFixed(3)}" style="width:${(Math.abs(v) / tot) * 100}%;background:var(${FAMILY_COLORS[k] || "--s8"})"></span>`).join("")}</div>
  <div class="legend-row mt8">${entries.map(([k, v]) => `<span><i class="sw" style="background:var(${FAMILY_COLORS[k] || "--s8"})"></i>${esc(FAMILY_LABELS[k] || k)} <b class="mono" style="color:${v > 0 ? "var(--increase)" : "var(--decrease)"}">${v >= 0 ? "+" : ""}${v.toFixed(2)}</b></span>`).join("")}</div>`;
}

// ----------------------------------------------------------------- tabs
// items: [[key, label, html], ...]; the first tab is shown.
export function tabs(id, items, { bar = true } = {}) {
  const head = `<div class="seg tab-bar" role="tablist" data-tabs="${id}">${items.map(([k, l], i) =>
    `<button type="button" role="tab" data-tab="${k}" class="${i ? "" : "on"}" aria-selected="${!i}">${l}</button>`).join("")}</div>`;
  const panes = items.map(([k, , html], i) => `<div class="tab-pane" data-pane="${id}:${k}" ${i ? "hidden" : ""}>${html}</div>`).join("");
  return bar ? head + panes : { head, panes };
}
export function bindTabs(root, id) {
  root.querySelectorAll(`[data-tabs="${id}"] [data-tab]`).forEach((b) => b.addEventListener("click", () => {
    root.querySelectorAll(`[data-tabs="${id}"] [data-tab]`).forEach((x) => { x.classList.toggle("on", x === b); x.setAttribute("aria-selected", x === b); });
    root.querySelectorAll(`[data-pane^="${id}:"]`).forEach((p) => {
      p.hidden = p.dataset.pane !== `${id}:${b.dataset.tab}`;
      // Plotly sizes charts drawn while hidden to a default box; fit them now.
      if (!p.hidden && window.Plotly) p.querySelectorAll(".js-plotly-plot").forEach((el) => window.Plotly.Plots.resize(el));
    });
  }));
}

export function downloadText(filename, text, type = "text/csv") {
  const blob = new Blob([text], { type });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
export function toCSV(rows, cols) {
  const q = (v) => { const s = v === null || v === undefined ? "" : String(v); return /[",\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s; };
  return [cols.join(","), ...rows.map((r) => cols.map((c) => q(r[c])).join(","))].join("\n");
}
