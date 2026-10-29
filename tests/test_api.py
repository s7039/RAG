"""FastAPI 接口测试（本地向量 + 离线模拟模型）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

RECORD = """健康档案
一、基本信息
张三九，男，出生于1985年5月20日，职业为IT工程师。
二、医疗历史
张三九对青霉素有过敏反应，曾出现皮疹。
三、体检与化验结果
张三九的空腹血糖为6.8毫摩尔/升，略高于正常范围。
"""


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "健康档案.txt").write_text(RECORD, encoding="utf-8")

    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("COLLECTION_NAME", "api_test_records")
    monkeypatch.setenv("EMBED_PROVIDER", "local")
    monkeypatch.setenv("RETRIEVE_MODE", "hybrid")
    monkeypatch.setenv("ALLOW_MOCK_LLM", "true")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    from app.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


def ingest(client: TestClient) -> dict:
    response = client.post("/api/ingest", json={"reset": True})
    assert response.status_code == 200, response.text
    return response.json()


def test_health_and_index_page(client: TestClient):
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["chunks"] == 0

    page = client.get("/")
    assert page.status_code == 200
    assert "健康档案" in page.text


def test_ingest_then_query(client: TestClient):
    stats = ingest(client)["stats"]
    assert stats["chunks"] > 0
    assert stats["files"] == 1
    assert stats["collection_count"] == stats["chunks"]

    response = client.post(
        "/api/query",
        json={"question": "张三九对什么过敏？", "top_k": 3},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["mode"] == "mock"
    assert body["sources"]
    assert "青霉素" in body["answer"]
    assert body["timings_ms"]["total"] > 0


def test_retrieve_endpoint_and_documents(client: TestClient):
    ingest(client)

    response = client.post("/api/retrieve", json={"query": "空腹血糖", "top_k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "hybrid"
    assert body["chunks"]
    assert "6.8" in body["chunks"][0]["content"]

    docs = client.get("/api/documents").json()
    assert docs["count"] == 1
    assert "张三九" in docs["data"][0]["patients"]


def test_stream_endpoint_emits_events(client: TestClient):
    ingest(client)
    with client.stream(
        "POST",
        "/api/query/stream",
        json={"question": "张三九的过敏史", "top_k": 2},
    ) as response:
        assert response.status_code == 200
        events: list[tuple[str, dict]] = []
        name = ""
        for line in response.iter_lines():
            if line.startswith("event:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                events.append((name, json.loads(line.split(":", 1)[1])))

    names = [name for name, _ in events]
    assert "retrieval" in names
    assert "token" in names
    assert names[-1] == "done"
    text = "".join(data["text"] for name, data in events if name == "token")
    assert text.strip()


def test_clear_index_and_validation(client: TestClient):
    ingest(client)
    assert client.delete("/api/index").json()["chunks"] == 0

    invalid = client.post("/api/query", json={"question": ""})
    assert invalid.status_code == 422


def test_query_before_ingest_still_answers(client: TestClient):
    response = client.post("/api/query", json={"question": "张三九的血压"})
    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == []
    assert body["warnings"]
