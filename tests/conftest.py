import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GOOD_TOOL = {
    "tool_name": "search_customer",
    "description": "Searches for a customer using customer ID.",
    "input_schema": {"type": "object", "properties": {"customer_id": {"type": "string", "description": "Unique customer identifier"}},
                     "required": ["customer_id"]},
}
VAGUE_TOOL = {"tool_name": "process", "description": "Handles data operations as needed.",
              "input_schema": {"type": "object", "properties": {"data": {}, "mode": {}, "options": {"type": "object"}}}}

MODEL_READY = (ROOT / "models" / "best_model.joblib").exists()
needs_model = pytest.mark.skipif(not MODEL_READY, reason="trained model not found - run python -m src.pipeline.run_all")


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from backend.app.main import app
    with TestClient(app) as c:
        yield c
