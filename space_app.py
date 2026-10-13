"""Hugging Face Spaces entry point (Gradio SDK).

Serves the full MCPGuard FastAPI app (REST API + research dashboard at /) and
mounts a Gradio analyzer at /gradio, all on one port:

    python space_app.py        # http://localhost:7860  ·  /gradio  ·  /docs
"""
from __future__ import annotations

import json
import os

import gradio as gr
import uvicorn

from backend.app.main import app
from src.inference.predictor import get_predictor

try:  # ZeroGPU hardware requires at least one @spaces.GPU function; MCPGuard itself runs on CPU.
    import spaces

    @spaces.GPU
    def _gpu_probe() -> bool:  # never called
        return True
except Exception:  # noqa: BLE001 - `spaces` only exists on Hugging Face
    pass

EXAMPLES = [
    ["get_current_weather", "Get the current weather for a city. Use this when the user asks about present conditions (not forecasts). "
     "Returns temperature, humidity and a short summary.",
     json.dumps({"type": "object", "properties": {"location": {"type": "string", "description": "City and country, e.g. 'Paris, France'."},
                                                   "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}, indent=2)],
    ["process", "Handles data operations as needed.",
     json.dumps({"type": "object", "properties": {"data": {}, "mode": {}, "options": {"type": "object"}}}, indent=2)],
    ["search_customer", "Searches for a customer using customer ID.",
     json.dumps({"type": "object", "properties": {"customer_id": {"type": "string", "description": "Unique customer identifier"}},
                 "required": ["customer_id"]}, indent=2)],
]


def analyze(tool_name: str, description: str, schema_text: str):
    tool_name, description = (tool_name or "").strip(), (description or "").strip()
    if not tool_name or not description:
        raise gr.Error("Tool name and description are required.")
    try:
        schema = json.loads(schema_text) if (schema_text or "").strip() else {}
    except json.JSONDecodeError as exc:
        raise gr.Error(f"Input schema is not valid JSON (line {exc.lineno}, column {exc.colno}).") from None
    if not isinstance(schema, dict):
        raise gr.Error("Input schema must be a JSON object.")
    r = get_predictor().predict([{"tool_name": tool_name[:200], "description": description[:5000], "input_schema": schema}], explain=True)[0]
    verdict = "Likely to cause agent errors" if r["prediction"] else "Likely to work reliably"
    summary = (f"## {r['risk_level']} risk · {r['risk_score']:.1f}%\n**{verdict}** (decision threshold {r['decision_threshold'] * 100:.1f}%).\n\n"
               "### Main reasons\n" + "\n".join(f"- {w}" for w in (r["risk_factors"][:4] or ["No dominant risk drivers."])))
    if r.get("schema_issues"):
        summary += "\n\n**Schema issues:** " + ", ".join(r["schema_issues"])
    fixes = "\n".join(f"{i}. **{x['title']}** ({x['priority']}) — {x['detail']}" for i, x in enumerate(r["recommendations"], 1)) or "No fixes suggested."
    shap_rows = [[f["display_name"], f["family_label"], round(f["shap"], 4)] for f in r["top_features"]]
    return {"P(agent error)": r["probability"], "P(success)": 1 - r["probability"]}, summary, fixes, shap_rows


with gr.Blocks(title="MCPGuard · Gradio analyzer") as demo:
    gr.Markdown("# MCPGuard — Tool Risk Analyzer\nPredict whether an MCP tool definition will cause AI-agent errors, see why (SHAP), "
                "and get concrete fixes. Full dashboard: [/](/) · API docs: [/docs](/docs)")
    with gr.Row():
        with gr.Column():
            name = gr.Textbox(label="Tool name", placeholder="e.g. search_customer")
            desc = gr.Textbox(label="Tool description", lines=4)
            schema = gr.Code(label="Input schema (JSON)", language="json", lines=12)
            btn = gr.Button("Analyze risk", variant="primary")
        with gr.Column():
            prob = gr.Label(label="Prediction")
            summary = gr.Markdown()
            with gr.Tab("Fixes"):
                fixes = gr.Markdown()
            with gr.Tab("SHAP"):
                shap_tbl = gr.Dataframe(headers=["feature", "family", "SHAP (log-odds)"], interactive=False)
    gr.Examples(EXAMPLES, inputs=[name, desc, schema])
    btn.click(analyze, inputs=[name, desc, schema], outputs=[prob, summary, fixes, shap_tbl])

app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "7860")))
