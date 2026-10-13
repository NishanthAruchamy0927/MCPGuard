"""Hugging Face Space entry point: Gradio analyzer + mounted FastAPI app."""
import pytest

gr = pytest.importorskip("gradio")


@pytest.fixture(scope="module")
def space():
    import space_app
    return space_app


def test_gradio_mounted_alongside_api(space):
    from fastapi.testclient import TestClient
    client = TestClient(space.app)
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/gradio/").status_code == 200


def test_analyze_returns_label_summary_fixes_shap(space):
    label, summary, fixes, shap_rows = space.analyze("process", "Handles data operations as needed.", '{"type":"object","properties":{"data":{}}}')
    assert abs(label["P(agent error)"] + label["P(success)"] - 1) < 1e-9
    assert "risk" in summary and "Main reasons" in summary
    assert fixes and shap_rows and len(shap_rows[0]) == 3


@pytest.mark.parametrize("args", [("", "desc", "{}"), ("name", "", "{}"), ("name", "desc", "{bad"), ("name", "desc", "[1, 2]")])
def test_analyze_rejects_invalid_input(space, args):
    with pytest.raises(gr.Error):
        space.analyze(*args)
