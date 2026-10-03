import importlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings


def test_public_http_demo_streams_all_nodes_and_citations(monkeypatch, tmp_path):
    web = importlib.import_module("app.main")
    fixtures = Path(__file__).parent / "fixtures" / "corpus"
    monkeypatch.setattr(web, "settings", Settings(
        data_root=fixtures, index_root=tmp_path,
        model="unused", base_url=None, api_key=None, demo_mode=True,
    ))
    with TestClient(web.app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/health").json() == {
            "status": "ok", "mode": "demo", "data_root_exists": True,
        }
        response = client.post("/api/chat/stream", json={"question": "安心保险的等待期是多久？"})
        assert response.status_code == 200
        events = [json.loads(line) for line in response.text.splitlines() if line]
        nodes = {item["data"]["node"] for item in events if item["type"] == "trace"}
        assert {"prepare", "decompose", "worker", "finalize"} <= nodes
        answer = next(item["data"] for item in events if item["type"] == "answer")
        assert answer["citations"][0]["document_id"] == "policy"
        assert "九十日" in answer["citations"][0]["quote"]
        assert events[-1]["type"] == "done"
        assert not any(item["type"] == "error" for item in events)


def test_meta_reports_the_model_actually_used_for_finalization(monkeypatch, tmp_path):
    web = importlib.import_module("app.main")
    monkeypatch.setattr(web, "settings", Settings(
        data_root=tmp_path, index_root=tmp_path, model="flash-actual",
        answer_model="legacy-unused", planner_model="max-planner",
        base_url=None, api_key=None, demo_mode=False,
    ))
    # No lifespan / model initialization: this tests metadata only, without network access.
    client = TestClient(web.app)
    response = client.get("/api/meta")
    assert response.status_code == 200
    assert response.json()["answer_model"] == "flash-actual"
    assert response.json()["planner_model"] == "max-planner"
    client.close()
