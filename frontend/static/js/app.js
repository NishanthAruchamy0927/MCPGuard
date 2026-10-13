import { $, $$, api, closeDrawer, hydrateIcons, icon, int } from "./core.js";
import { rerenderCharts, resetCharts } from "./charts.js";
import home from "./pages/home.js";
import analyze from "./pages/analyze.js";
import batch from "./pages/batch.js";
import performance from "./pages/performance.js";
import explain from "./pages/explain.js";
import errors from "./pages/errors.js";
import experiments from "./pages/experiments.js";
import dataset from "./pages/dataset.js";
import about from "./pages/about.js";

const ROUTES = {
  "": { page: home, section: "Overview", title: "Dashboard", key: "home" },
  analyze: { page: analyze, section: "Analyze", title: "Tool Risk Analyzer", key: "analyze" },
  batch: { page: batch, section: "Analyze", title: "Batch Analyzer", key: "batch" },
  performance: { page: performance, section: "Research", title: "Model Performance", key: "performance" },
  explain: { page: explain, section: "Research", title: "Explainability", key: "explain" },
  errors: { page: errors, section: "Research", title: "Error Analysis", key: "errors" },
  experiments: { page: experiments, section: "Research", title: "Experiments", key: "experiments" },
  dataset: { page: dataset, section: "Data", title: "Dataset Explorer", key: "dataset" },
  about: { page: about, section: "About", title: "About & Research", key: "about" },
};

function parseHash() {
  const h = location.hash.replace(/^#\/?/, "");
  const [path, query = ""] = h.split("?");
  return { path: path.split("/")[0], params: new URLSearchParams(query) };
}

async function render() {
  const { path, params } = parseHash();
  const route = ROUTES[path] || ROUTES[""];
  resetCharts();
  closeDrawer();
  $$("#nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === route.key));
  $("#crumb-section").textContent = route.section;
  $("#crumb-page").textContent = route.title;
  document.title = `${route.title} · MCPGuard`;
  $("#sidebar").classList.remove("open");
  const view = $("#view");
  view.style.animation = "none"; void view.offsetWidth; view.style.animation = "";
  view.innerHTML = "";
  try {
    await route.page(view, params);
  } catch (e) {
    console.error(e);
    view.innerHTML = `<div class="panel"><div class="empty"><div class="big-icon">${icon("alert")}</div><h3>Could not render this page</h3><p>${e.message || e}</p></div></div>`;
  }
  hydrateIcons(view);
  window.scrollTo({ top: 0 });
}

function setTheme(t) {
  document.documentElement.setAttribute("data-theme", t);
  try { localStorage.setItem("mcpguard-theme", t); } catch (e) { /* storage blocked */ }
  $("#theme-btn").innerHTML = icon(t === "dark" ? "sun" : "moon");
  rerenderCharts();
}

async function pollStatus() {
  const dot = $("#status-card .dot");
  try {
    const h = await api("/health", { cacheable: false });
    dot.className = `dot ${h.model_loaded ? "ok" : "bad"}`;
    $("#status-model").textContent = h.model_loaded ? `${h.model} · v${h.model_version}` : "model not loaded";
  } catch (e) {
    dot.className = "dot bad";
    $("#status-model").textContent = "API unreachable";
  }
}

async function datasetPill() {
  try {
    const s = await api("/stats");
    $("#dataset-pill").innerHTML = `${icon("database")}<span>${int(s.dataset_size)} tools · ${s.servers} servers</span>`;
  } catch (e) { /* optional */ }
}

window.addEventListener("hashchange", render);
document.addEventListener("DOMContentLoaded", () => {
  hydrateIcons();
  setTheme(document.documentElement.getAttribute("data-theme") || "dark");
  $("#theme-btn").addEventListener("click", () => setTheme(document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark"));
  $("#menu-btn").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
  $("#drawer-close").addEventListener("click", closeDrawer);
  $("#drawer-scrim").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
  render();
  pollStatus();
  datasetPill();
  setInterval(pollStatus, 30000);
});
