"""BM25 关键词检索与分词单元测试。"""

from __future__ import annotations

from app.rag.bm25 import BM25Index
from app.rag.tokenize import tokenize


def test_tokenize_chinese_bigrams_and_english_words():
    tokens = tokenize("空腹血糖 6.8 mmol")
    assert "空腹" in tokens
    assert "血糖" in tokens
    assert "mmol" in tokens
    # 高频虚词单字不作为独立词
    assert "的" not in tokenize("张三九的血糖")


def test_bm25_ranks_relevant_document_first():
    index = BM25Index()
    index.build(
        [
            {"id": "a", "document": "张三九的空腹血糖为6.8毫摩尔/升，略高于参考范围。", "metadata": {"patient": "张三九"}},
            {"id": "b", "document": "李四六对阿司匹林过敏，曾出现皮疹。", "metadata": {"patient": "李四六"}},
            {"id": "c", "document": "王五的血型为B型，职业是银行经理。", "metadata": {"patient": "王五"}},
        ]
    )
    hits = index.search("空腹血糖", k=3)
    assert hits and hits[0][0] == "a"
    assert hits[0][1] > 0


def test_bm25_metadata_filter():
    index = BM25Index()
    index.build(
        [
            {"id": "a", "document": "张三九血糖偏高。", "metadata": {"patient": "张三九"}},
            {"id": "b", "document": "李四六血糖正常。", "metadata": {"patient": "李四六"}},
        ]
    )
    hits = index.search("血糖", k=5, where={"patient": "李四六"})
    assert [doc_id for doc_id, _ in hits] == ["b"]


def test_bm25_empty_index_is_safe():
    index = BM25Index()
    assert index.search("血压") == []
    assert index.size == 0
