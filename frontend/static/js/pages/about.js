import { api, esc, icon, int, pct, fmt } from "../core.js";

const FLOW = [
  ["Data sources", "BFCL v3 functions, queries & gold calls; MCP reference servers"],
  ["MCP tool collection", "Convert to MCP {name, description, inputSchema}"],
  ["Data validation", "JSON-schema parsing, type-alias repair, de-duplication"],
  ["Controlled benchmark", "Simulated agent × 10 episodes, BFCL-style validation", true],
  ["Text + schema preprocessing", "Unicode/whitespace normalisation, schema statistics"],
  ["Feature engineering", "Lexical, schema & 6 engineered risk scores"],
  ["TF-IDF + embeddings", "Word/char n-grams, all-MiniLM-L6-v2"],
  ["Dimensionality reduction", "LSA (SVD), PCA; UMAP for visualisation"],
  ["Feature fusion", "Fused vector: text LSA, semantic PCA, schema, risk", true],
  ["Model training", "LR, RF, XGBoost, LightGBM, HistGB, MLP"],
  ["Optuna optimisation", "TPE + median pruning, grouped CV"],
  ["Best model", "Selected by validation PR-AUC → F1 → ROC-AUC", true],
  ["SHAP explainability", "TreeExplainer, global + local"],
  ["Error analysis", "TP/TN/FP/FN, slices, bootstrap tests"],
  ["FastAPI", "Validated REST API, /docs, SQLite"],
  ["Web dashboard", "This research console"],
  ["Docker", "Single non-root container + health check"],
  ["Hugging Face / AWS", "Docker Space on port 7860"],
];

export default async function about(view) {
  let s = null, info = null;
  try { [s, info] = await Promise.all([api("/stats"), api("/model-info")]); } catch (e) { /* page still renders statically */ }
  view.innerHTML = `
  <div class="page-head"><div><div class="eyebrow">About & research</div><h1 class="mt8">Predicting Which MCP Tool Descriptions Cause Agent Errors</h1>
    <p>An Explainable Machine Learning Framework for Detecting High-Risk Model Context Protocol Tool Descriptions — 19MAM56 Advanced Machine Learning Lab, CAT II project.</p></div></div>

  <div class="grid g3">
    <div class="panel prose"><div class="eyebrow">Problem</div>
      <p class="mt8">AI agents connect to external systems through Model Context Protocol (MCP) tools. The agent only knows a tool through its <b>name, natural-language description and JSON input schema</b>. When these are vague, incomplete or structurally complex, agents select the wrong tool, omit required arguments or produce invalid values.</p></div>
    <div class="panel prose"><div class="eyebrow">Motivation</div>
      <p class="mt8">Tool catalogues grow faster than they can be manually reviewed. A model that flags risky descriptions <b>before deployment</b> — and explains why — lets developers fix documentation instead of debugging agent failures in production.</p></div>
    <div class="panel prose"><div class="eyebrow">Research question</div>
      <p class="mt8"><b>Can machine learning identify high-risk MCP tool descriptions before they are deployed to AI agents?</b> And which textual, semantic and structural properties drive that risk?</p></div>
  </div>

  <div class="panel mt16 prose"><div class="eyebrow">Research gap</div>
    <p class="mt8">Tool-use research has largely evaluated <i>agents</i> — their ability to select and call tools (e.g. Toolformer, Gorilla/BFCL, ToolLLM) — and shown that better documentation improves tool use (Hsieh et al., 2023; EasyTool, 2024). Comparatively little work treats the <i>tool description itself</i> as the unit of prediction. This project frames tool-documentation quality as a supervised risk-prediction problem with explicit explanations. This is presented as a gap in the literature surveyed for this project, not as an established fact; see the literature review in <code>docs/LITERATURE_REVIEW.md</code>.</p></div>

  <div class="panel mt16">
    <div class="panel-head"><div><div class="eyebrow">Methodology</div><h3 class="mt8">End-to-end pipeline</h3></div></div>
    <div class="flow">${FLOW.map(([t, d, hl]) => `<div class="node ${hl ? "hl" : ""}"><b>${esc(t)}</b><span>${esc(d)}</span></div>`).join("")}</div>
  </div>

  <div class="grid g2 mt16">
    <div class="panel prose"><div class="eyebrow">Dataset & label provenance</div>
      <ul class="mt8">
        <li><b>Real tool definitions:</b> BFCL v3 (gorilla-llm, Apache-2.0) functions with their real user questions and gold calls, plus curated MCP reference-server tools.</li>
        <li><b>Controlled benchmark labels:</b> a documentation-bounded simulated agent attempts each tool in 10 episodes; calls are validated against the hidden contract and gold values (BFCL AST-style). <code>error_occurred = failure_rate ≥ 0.30</code>.</li>
        <li><b>Synthetic augmentation:</b> documented degradation / enrichment operators; labels re-measured, never assigned.</li>
        <li><b>Leakage control:</b> grouped stratified split by tool lineage (70/15/15).</li>
      </ul>
      ${s ? `<p class="mt8 muted">${int(s.dataset_size)} records · ${int(s.servers)} servers · agent-error rate ${pct(s.positive_rate)}</p>` : ""}</div>
    <div class="panel prose"><div class="eyebrow">Models, tuning & explainability</div>
      <ul class="mt8">
        <li>Baselines: majority, documentation-quality heuristic, TF-IDF + Logistic Regression, Random Forest.</li>
        <li>Advanced: XGBoost, LightGBM, Histogram GB, MLP on fused features; XGBoost & LightGBM tuned with Optuna.</li>
        <li>Imbalance: cost-sensitive weights + validation-tuned threshold (no SMOTE on embeddings).</li>
        <li>Explainability: SHAP TreeExplainer (global, local, per-family) + rule-based recommendation layer.</li>
      </ul>
      ${info ? `<p class="mt8 muted">Production model: ${esc(info.model_name)} · test PR-AUC ${fmt(info.metrics.test.pr_auc)} · F1 ${fmt(info.metrics.test.f1)}</p>` : ""}</div>
  </div>

  <div class="grid g2 mt16">
    <div class="panel prose"><div class="eyebrow">Deployment</div>
      <ul class="mt8"><li>FastAPI backend with Pydantic validation, OpenAPI docs at <a href="/docs">/docs</a> and <a href="/redoc">/redoc</a>.</li>
      <li>This dashboard is served by the same app — one container, one port (7860).</li>
      <li>Dockerfile (non-root, health check) and docker-compose; Hugging Face Docker Space ready.</li>
      <li>SQLite persistence (PostgreSQL via <code>DATABASE_URL</code>); only hashes of descriptions are stored.</li></ul></div>
    <div class="panel prose"><div class="eyebrow">Limitations & future scope</div>
      <ul class="mt8"><li>Labels come from a <b>simulated</b> agent; replace it with real LLM agents (pass@k over several models) to validate external validity.</li>
      <li>Some failure modes (sibling-tool routing, user phrasing, server strictness) are not observable from a single description.</li>
      <li>Future: LLM-based description rewriting, cross-model risk calibration, active learning from production telemetry.</li></ul></div>
  </div>

  <div class="panel mt16"><div class="eyebrow">Key references (IEEE)</div>
    <ol class="ref-list mt12">
      <li>Anthropic, "Model Context Protocol specification," 2024. [Online]. Available: https://modelcontextprotocol.io</li>
      <li>S. G. Patil, T. Zhang, X. Wang, and J. E. Gonzalez, "Gorilla: Large language model connected with massive APIs," arXiv:2305.15334, 2023.</li>
      <li>C.-Y. Hsieh <i>et al.</i>, "Tool documentation enables zero-shot tool-usage with large language models," arXiv:2308.00675, 2023.</li>
      <li>S. Yuan <i>et al.</i>, "EasyTool: Enhancing LLM-based agents with concise tool instruction," arXiv:2401.06201, 2024.</li>
      <li>Y. Qin <i>et al.</i>, "ToolLLM: Facilitating large language models to master 16000+ real-world APIs," arXiv:2307.16789, 2023.</li>
      <li>T. Schick <i>et al.</i>, "Toolformer: Language models can teach themselves to use tools," in <i>Proc. NeurIPS</i>, 2023.</li>
      <li>T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in <i>Proc. ACM SIGKDD</i>, 2016, pp. 785–794.</li>
      <li>S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in <i>Proc. NeurIPS</i>, 2017.</li>
      <li>T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A next-generation hyperparameter optimization framework," in <i>Proc. ACM SIGKDD</i>, 2019.</li>
      <li>N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in <i>Proc. EMNLP-IJCNLP</i>, 2019.</li>
    </ol>
    <p class="muted mt12" style="font-size:12px">Full list with additional entries: <code>docs/REFERENCES.md</code>. Verify every entry against the publisher before submission.</p>
  </div>`;
}
