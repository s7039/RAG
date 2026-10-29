"""BM25 关键词检索。

向量检索擅长「语义相近」，但对「空腹血糖」「血红蛋白」这类专有名词、
数值指标容易召回偏；BM25 恰好擅长精确词匹配。两者融合（Hybrid）后，
在中文健康档案上召回质量明显更稳。

索引常驻内存、随灌库重建；数据量大时（超过 MAX_DOCS）自动降级为纯向量检索。
"""

from __future__ import annotations

import logging
import math
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

from app.rag.tokenize import tokenize

logger = logging.getLogger(__name__)


@dataclass
class _Doc:
    doc_id: str
    length: int
    metadata: dict[str, Any] = field(default_factory=dict)


class BM25Index:
    """Okapi BM25，支持按元数据过滤。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75, max_docs: int = 50_000) -> None:
        self.k1 = k1
        self.b = b
        self.max_docs = max_docs
        self._lock = threading.RLock()
        self._docs: dict[str, _Doc] = {}
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._avg_len = 0.0
        self._n = 0

    @property
    def size(self) -> int:
        return self._n

    def clear(self) -> None:
        with self._lock:
            self._docs = {}
            self._postings = defaultdict(dict)
            self._avg_len = 0.0
            self._n = 0

    def build(self, rows: Iterable[dict]) -> int:
        """rows: [{"id": str, "document": str, "metadata": dict}, ...]"""
        with self._lock:
            self.clear()
            total_len = 0
            for row in rows:
                doc_id = str(row.get("id") or "")
                text = row.get("document") or ""
                if not doc_id or not text.strip():
                    continue
                tokens = tokenize(text)
                if not tokens:
                    continue
                counts = Counter(tokens)
                self._docs[doc_id] = _Doc(doc_id, len(tokens), dict(row.get("metadata") or {}))
                for token, tf in counts.items():
                    self._postings[token][doc_id] = tf
                total_len += len(tokens)
            self._n = len(self._docs)
            self._avg_len = total_len / self._n if self._n else 0.0

            if self._n > self.max_docs:
                logger.warning("文档数 %s 超过 BM25 上限 %s，停用关键词检索", self._n, self.max_docs)
                self.clear()
                return 0
            logger.info("BM25 索引就绪：%s 篇文档，%s 个词项", self._n, len(self._postings))
            return self._n

    def _match(self, meta: dict[str, Any], where: dict | None) -> bool:
        if not where:
            return True
        for key, value in where.items():
            if meta.get(key) != value:
                return False
        return True

    def search(self, query: str, k: int = 20, where: dict | None = None) -> list[tuple[str, float]]:
        """返回 [(doc_id, 原始 BM25 分数)]，按分数降序。

        这里不把分数压到 0~1：归一化交给调用方（融合时用软饱和），
        否则「最相关的一条」永远等于 1，会掩盖它其实只匹配了通用词。
        """
        if not self._n:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []

        with self._lock:
            n = self._n
            avg_len = self._avg_len or 1.0
            scores: dict[str, float] = defaultdict(float)
            for token in set(tokens):
                postings = self._postings.get(token)
                if not postings:
                    continue
                # 单字（尤其是「的/有/多/少」）噪声大，降权；二元组/单词权重为 1
                weight = 0.25 if len(token) == 1 else 1.0
                df = len(postings)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                for doc_id, tf in postings.items():
                    doc = self._docs[doc_id]
                    if not self._match(doc.metadata, where):
                        continue
                    denom = tf + self.k1 * (1 - self.b + self.b * doc.length / avg_len)
                    scores[doc_id] += weight * idf * tf * (self.k1 + 1) / denom

        if not scores:
            return []
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
