"""RAG 服务端到端测试：本地向量 + 离线模拟模型，不依赖网络。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.rag.pipeline import RagService

RECORD = """健康档案
一、基本信息
张三九，男，出生于1985年5月20日，职业为IT工程师，血型O型。
二、医疗历史
张三九在2005年被确诊为高血压，2010年被确诊为二型糖尿病。
张三九对青霉素有过敏反应。
三、体检与化验结果
| 项目 | 结果 | 参考范围 | 评估 |
| --- | --- | --- | --- |
| 空腹血糖 | 6.8毫摩尔/升 | 4.0-6.1毫摩尔/升 | 略高 |
| 总胆固醇 | 5.2毫摩尔/升 | <5.2毫摩尔/升 | 正常 |

健康档案
一、基本信息
李四六，女，出生于1990年3月15日，职业为小学教师，血型A型。
二、医疗历史
李四六对阿司匹林过敏，曾出现皮疹。
李四六无重大手术史。
"""


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "健康档案.txt").write_text(RECORD, encoding="utf-8")

    return Settings(
        _env_file=None,
        input_dir=str(input_dir),
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_records",
        embed_provider="local",
        retrieve_mode="hybrid",
        use_mmr=False,
        chunk_size=300,
        chunk_overlap=50,
        allow_mock_llm=True,
        deepseek_api_key="",
        openai_api_key="",
    )


@pytest.fixture()
def service(settings: Settings) -> RagService:
    svc = RagService(settings)
    stats = svc.ingest(reset=True)
    assert stats.chunks > 0
    return svc


def test_ingest_builds_vector_store_and_keyword_index(service: RagService):
    assert service.store.count() > 0
    assert service.store.count() == service.bm25.size
    docs = service.store.document_stats()
    assert len(docs) == 1
    assert "张三九" in docs[0]["patients"]
    assert "李四六" in docs[0]["patients"]


def test_retrieve_hybrid_returns_patient_specific_chunk(service: RagService):
    result = service.retriever.retrieve("李四六对什么过敏？", top_k=3)
    assert result.chunks
    assert result.strategy == "hybrid"
    assert result.chunks[0].patient == "李四六"
    assert "阿司匹林" in result.chunks[0].content


def test_retrieve_with_patient_filter(service: RagService):
    result = service.retriever.retrieve("空腹血糖是多少", top_k=3, where={"patient": "李四六"})
    assert all(chunk.patient == "李四六" for chunk in result.chunks)


def test_ask_uses_mock_model_without_api_key(service: RagService):
    answer = service.ask("张三九的空腹血糖是多少？", top_k=3)
    assert answer.mode == "mock"
    assert "6.8" in answer.answer
    assert answer.sources
    assert answer.timings_ms["total"] > 0


def test_status_reports_index(service: RagService):
    status = service.status()
    assert status["chunks"] == service.store.count()
    assert status["retrieve_mode"] == "hybrid"
    assert status["keyword_index_docs"] == service.bm25.size


def test_append_mode_keeps_existing_vectors(settings: Settings):
    svc = RagService(settings)
    first = svc.ingest(reset=True)
    second = svc.ingest(reset=False)
    # upsert 使用稳定 id，重复灌入不会产生重复向量
    assert svc.store.count() == first.chunks == second.chunks
