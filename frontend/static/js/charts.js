import { cssVar } from "./core.js";

// Theme-aware Plotly helpers. Colours are read from CSS tokens so light/dark
// each use their own validated steps.
export const series = () => ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6", "--s7", "--s8"].map(cssVar);

export function baseLayout(extra = {}) {
  const text2 = cssVar("--text-2"), muted = cssVar("--muted"), grid = cssVar("--border");
  return {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, sans-serif", size: 12, color: text2 },
    margin: { l: 56, r: 18, t: 16, b: 46 },
    hoverlabel: { bgcolor: cssVar("--surface-solid"), bordercolor: cssVar("--border-strong"), font: { color: cssVar("--text"), family: "Inter", size: 12 } },
    xaxis: { gridcolor: grid, zerolinecolor: grid, linecolor: grid, tickfont: { color: muted }, title: { font: { color: text2, size: 12 } }, automargin: true },
    yaxis: { gridcolor: grid, zerolinecolor: grid, linecolor: grid, tickfont: { color: muted }, title: { font: { color: text2, size: 12 } }, automargin: true },
    legend: { orientation: "h", y: -0.18, x: 0, font: { color: text2, size: 11.5 }, bgcolor: "rgba(0,0,0,0)" },
    ...extra,
  };
}

export const config = { displayModeBar: false, responsive: true };

export function plot(el, data, layout = {}, cfg = {}) {
  if (!window.Plotly || !el) return;
  const base = baseLayout();
  const merged = { ...base, ...layout,
    xaxis: { ...base.xaxis, ...(layout.xaxis || {}) }, yaxis: { ...base.yaxis, ...(layout.yaxis || {}) },
    legend: { ...base.legend, ...(layout.legend || {}) }, margin: { ...base.margin, ...(layout.margin || {}) } };
  window.Plotly.react(el, data, merged, { ...config, ...cfg });
}

export function whenPlotly(fn) {
  if (window.Plotly) return fn();
  const t = setInterval(() => { if (window.Plotly) { clearInterval(t); fn(); } }, 60);
}

// re-render registered charts on theme change
const registry = new Set();
export function themed(fn) { registry.add(fn); whenPlotly(fn); }
export function rerenderCharts() { registry.forEach((fn) => { try { fn(); } catch (e) { /* view gone */ } }); }
export function resetCharts() { registry.clear(); }
