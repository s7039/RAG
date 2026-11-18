"""Agent 端到端场景：离线模式（无 Key）也必须全过。"""

from __future__ import annotations

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
四、医生建议
建议张三九减少高糖、高脂肪食物的摄入，增加蔬菜和水果的摄入量。
"""


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "健康档案.txt").write_text(RECORD, encoding="utf-8")

    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("COLLECTION_NAME", "agent_test_records")
    monkeypatch.setenv("EMBED_PROVIDER", "local")
    monkeypatch.setenv("RETRIEVE_MODE", "hybrid")
    monkeypatch.setenv("ALLOW_MOCK_LLM", "true")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    # 本机可能配了 SiliconFlow/自定义供应商，测试必须把外部服务关掉
    monkeypatch.setenv("CUSTOM_API_KEY", "")
    monkeypatch.setenv("CUSTOM_BASE_URL", "")
    monkeypatch.setenv("CUSTOM_CHAT_MODEL", "")
    monkeypatch.setenv("CUSTOM_EMBEDDING_MODEL", "")
    monkeypatch.setenv("DEFAULT_PROVIDER", "deepseek")
    monkeypatch.setenv("DEFAULT_MODEL", "")

    from app.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as test_client:
        response = test_client.post("/api/ingest", json={"reset": True})
        assert response.status_code == 200, response.text
        yield test_client
    get_settings.cache_clear()


def pg_ready() -> bool:
    """本地没起 db 容器时跳过落库断言，CI 里会起 PostgreSQL 服务。"""
    try:
        from sqlalchemy import text

        from app.db import SessionLocal, healthcheck

        healthcheck()
        with SessionLocal() as db:
            return db.execute(text("select to_regclass('public.tasks')")).scalar_one() is not None
    except Exception:
        return False


def post(client: TestClient, text: str, thread: str | None = None) -> dict:
    response = client.post("/api/agent/chat", json={"message": text, "thread_id": thread})
    assert response.status_code == 200, response.text
    return response.json()


def test_search_single_turn(client: TestClient):
    body = post(client, "张三九的空腹血糖是多少")
    assert any(c["name"] == "search_knowledge" for c in body["tool_calls"])
    assert "6.8" in body["answer"]


def test_create_task(client: TestClient):
    if not pg_ready():
        pytest.skip("PostgreSQL 不可用，跳过待办落库断言")

    body = post(client, "明天下午三点提醒我复查血糖")
    assert any(c["name"] == "create_task" for c in body["tool_calls"])

    from sqlalchemy import text

    from app.db import SessionLocal

    with SessionLocal() as db:
        row = db.execute(text("select title from tasks order by id desc limit 1")).first()
    assert row is not None
    assert row.title == "复查血糖"


def test_invalid_args_are_reported():
    from app.agent.tools import run_tool

    res = run_tool("search_knowledge", {"query": "x", "top_k": 999}, service=None)
    assert res["ok"] is False and "参数不合法" in res["error"]


def test_repair_plan_uses_current_patient_and_original_query():
    from app.agent.graph import _repair_plan

    calls = _repair_plan(
        [{"name": "search_knowledge", "args": {"query": "血型", "patient": "她"}}],
        "张三九的血型是什么？", "", ["张三九", "李四六"], "",
    )
    assert calls[0]["args"]["query"] == "张三九的血型是什么？"
    assert calls[0]["args"]["patient"] == "张三九"


def test_repair_plan_falls_back_for_incomplete_reminder():
    from app.agent.graph import _repair_plan

    calls = _repair_plan(
        [
            {"name": "search_knowledge", "args": {"query": "x"}},
            {"name": "parse_time", "args": {"text": "明天下午三点"}},
        ],
        "明天下午三点提醒我复查", "", [], "",
    )
    assert [c["name"] for c in calls] == ["parse_time", "create_task"]


def test_create_task_accepts_chinese_due_at():
    if not pg_ready():
        pytest.skip("PostgreSQL 不可用，跳过待办落库断言")

    from app.agent.tools import run_tool

    res = run_tool("create_task", {"title": "复查", "due_at": "明天下午三点"}, service=None)
    assert res["ok"] is True
    assert "T15:00:00" in res["summary"]


def test_multi_turn_thread(client: TestClient):
    first = post(client, "张三九的空腹血糖是多少")
    second = post(client, "那他的饮食建议呢", thread=first["thread_id"])
    assert second["thread_id"] == first["thread_id"]
    assert second["answer"]
    # 指代「他」要能沿用上一轮的患者，检索不会跑偏
    assert second["tool_calls"][0]["args"]["patient"] == "张三九"


def test_out_of_scope(client: TestClient):
    body = post(client, "今天杭州天气怎么样")
    assert body["tool_calls"] == []
    assert "没有查到" in body["answer"]


def test_out_of_corpus_health_question_is_refused(client: TestClient):
    # 患者存在、但知识库里没有 MRI：检索能返回片段，也不能硬答
    body = post(client, "张三九的 MRI 报告在哪一页")
    assert "没查到" in body["answer"]
