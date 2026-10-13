import io
import json

from tests.conftest import GOOD_TOOL, VAGUE_TOOL, needs_model


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] in ("ok", "degraded")


def test_docs_and_openapi(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/predict", "/batch-predict", "/model-metrics", "/feature-importance", "/dataset-summary", "/error-analysis",
              "/experiments", "/model-info", "/reload-model", "/health"):
        assert p in paths


def test_root_serves_dashboard_and_json(client):
    assert "MCPGuard" in client.get("/", headers={"accept": "text/html"}).text
    assert client.get("/", headers={"accept": "application/json"}).json()["name"] == "MCPGuard"


def test_invalid_json_schema_returns_422(client):
    r = client.post("/predict", json={**GOOD_TOOL, "input_schema": "{not json"})
    assert r.status_code == 422
    body = r.json()
    assert body["errors"][0]["field"] == "input_schema" and "valid JSON" in body["errors"][0]["message"]


def test_missing_fields_return_422(client):
    r = client.post("/predict", json={"description": "x"})
    assert r.status_code == 422
    assert any(e["field"] == "tool_name" for e in r.json()["errors"])
    assert client.post("/predict", json={**GOOD_TOOL, "description": "   "}).status_code == 422


def test_wrong_schema_structure_returns_422(client):
    assert client.post("/predict", json={**GOOD_TOOL, "input_schema": {"type": "string"}}).status_code == 422
    assert client.post("/predict", json={**GOOD_TOOL, "input_schema": {"type": "object", "properties": []}}).status_code == 422
    assert client.post("/predict", json={**GOOD_TOOL, "input_schema": [1, 2]}).status_code == 422


def test_unknown_routes_do_not_leak_internals(client):
    r = client.get("/does-not-exist")
    assert r.status_code == 404 and "Traceback" not in r.text


@needs_model
def test_predict_valid(client):
    r = client.post("/predict", json=GOOD_TOOL)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability"] <= 1 and body["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert body["top_features"] and body["recommendations"]


@needs_model
def test_predict_accepts_schema_as_json_string(client):
    r = client.post("/predict", json={**GOOD_TOOL, "input_schema": json.dumps(GOOD_TOOL["input_schema"])})
    assert r.status_code == 200


@needs_model
def test_batch_predict_reports_invalid_rows(client):
    r = client.post("/batch-predict", json={"tools": [GOOD_TOOL, VAGUE_TOOL]})
    assert r.status_code == 200 and r.json()["n_ok"] == 2
    csv = "tool_name,description,input_schema\nok_tool,Does a thing,{}\nbad_tool,Broken schema,{oops\n"
    r = client.post("/batch-predict/file", files={"file": ("tools.csv", io.BytesIO(csv.encode()), "text/csv")})
    body = r.json()
    assert r.status_code == 200 and body["n_ok"] == 1 and body["n_failed"] == 1
    r = client.post("/batch-predict/file?format=csv", files={"file": ("tools.json", io.BytesIO(json.dumps([GOOD_TOOL]).encode()), "application/json")})
    assert r.status_code == 200 and r.text.startswith("tool_name,risk_score,risk_level,predicted_error,confidence,top_risk_factor")


@needs_model
def test_research_endpoints(client):
    for path in ("/model-metrics", "/feature-importance", "/dataset-summary", "/error-analysis", "/experiments", "/model-info",
                 "/stats", "/projections", "/dataset/tools?page_size=5&risk_level=HIGH"):
        assert client.get(path).status_code == 200, path
    tid = client.get("/dataset/tools?page_size=1").json()["rows"][0]["tool_id"]
    assert client.get(f"/tools/{tid}").json()["explanation"]["top"]
    assert client.get("/tools/nope").status_code == 404


@needs_model
def test_reload_model(client):
    r = client.post("/reload-model")
    assert r.status_code == 200 and r.json()["status"] == "reloaded"
