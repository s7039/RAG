"""检索层（Hybrid：向量 + BM25 关键词，可选 MMR 去重）。

在线流程里的「问题向量化 -> 检索 -> 重排」都发生在这里。

1. 向量召回：把问题向量化，到 Chroma 里找 cosine 最相近的 chunk；
2. 关键词召回：BM25 找词面最匹配的 chunk（专有名词、指标名更准）；
3. 加权融合：score = alpha * 向量相似度 + (1 - alpha) * BM25 归一化分数；
4. 可选 MMR：在「相关」和「不重复」之间取平衡，减少返回内容雷同；
5. 取 Top-K，可按相似度阈值过滤。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.config import Settings
from app.rag.bm25 import BM25Index
from app.rag.vectorstore import StoredHit, VectorStore


@dataclass
class RetrievedChunk:
    content: str
    metadata: dict[str, Any]
    score: float
    rank: int
    source: str = ""
    page: int | None = None
    page_end: int | None = None
    section: str = ""
    patient: str = ""
    vector_score: float = 0.0
    keyword_score: float = 0.0
    chunk_id: str = ""

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "score": round(float(self.score), 4),
            "vector_score": round(float(self.vector_score), 4),
            "keyword_score": round(float(self.keyword_score), 4),
            "rank": self.rank,
            "source": self.source,
            "page": self.page,
            "page_end": self.page_end,
            "section": self.section,
            "patient": self.patient,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
        }

    @property
    def citation(self) -> str:
        where = self.source
        if self.page:
            where += f" 第{self.page}页"
        if self.section:
            where += f" · {self.section}"
        return where


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    elapsed_ms: float = 0.0
    strategy: str = "hybrid"
    candidates: int = 0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "strategy": self.strategy,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "count": len(self.chunks),
            "candidates": self.candidates,
            "chunks": [c.to_dict() for c in self.chunks],
        }


class Retriever:
    def __init__(self, settings: Settings, store: VectorStore, bm25: BM25Index | None = None) -> None:
        self.settings = settings
        self.store = store
        self.bm25 = bm25 or BM25Index()

    # ------------------------------------------------------------ 组装结果
    def _make_chunk(self, hit: StoredHit, score: float, vector_score: float, keyword_score: float, rank: int) -> RetrievedChunk:
        meta = hit.metadata or {}
        return RetrievedChunk(
            content=hit.text,
            metadata=meta,
            score=score,
            rank=rank,
            source=str(meta.get("source", "")),
            page=int(meta["page"]) if meta.get("page") is not None else None,
            page_end=int(meta["page_end"]) if meta.get("page_end") is not None else None,
            section=str(meta.get("section", "")),
            patient=str(meta.get("patient", "")),
            vector_score=vector_score,
            keyword_score=keyword_score,
            chunk_id=hit.id,
        )

    def _hybrid_enabled(self) -> bool:
        return (self.settings.retrieve_mode or "hybrid").lower() == "hybrid" and self.bm25.size > 0

    # ------------------------------------------------------------ 主流程
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: dict | None = None,
        use_mmr: bool | None = None,
        score_threshold: float | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        k = max(1, top_k or self.settings.top_k)
        mmr = self.settings.use_mmr if use_mmr is None else use_mmr
        threshold = self.settings.score_threshold if score_threshold is None else score_threshold
        hybrid = self._hybrid_enabled()

        if self.store.count() == 0:
            return RetrievalResult(query=query, chunks=[], elapsed_ms=0.0, strategy="empty")

        fetch_k = max(k, self.settings.fetch_k)

        # 1) 向量召回
        vector_hits = self.store.query(query, n_results=fetch_k, where=where)
        candidates: dict[str, dict[str, Any]] = {}
        for hit in vector_hits:
            if not hit.text.strip():
                continue
            candidates[hit.id] = {"hit": hit, "vector": hit.similarity, "keyword": 0.0}

        # 2) 关键词召回
        if hybrid:
            for doc_id, raw_score in self.bm25.search(query, k=fetch_k, where=where):
                # 软饱和到 0~1：弱匹配不会被抬成 1 分，强匹配也能接近满分
                keyword_score = 1.0 - math.exp(-raw_score / max(0.1, self.settings.hybrid_kw_scale))
                if doc_id in candidates:
                    candidates[doc_id]["keyword"] = keyword_score
                else:
                    candidates[doc_id] = {"hit": None, "vector": 0.0, "keyword": keyword_score}

        # 补齐只在 BM25 里出现的文档正文
        missing = [doc_id for doc_id, item in candidates.items() if item["hit"] is None]
        if missing:
            for hit in self.store.get_by_ids(missing):
                if hit.id in candidates:
                    candidates[hit.id]["hit"] = hit

        # 3) 融合打分
        alpha = float(min(1.0, max(0.0, self.settings.hybrid_alpha))) if hybrid else 1.0
        ranked: list[tuple[StoredHit, float, float, float]] = []
        for item in candidates.values():
            hit = item["hit"]
            if hit is None or not hit.text.strip():
                continue
            fused = alpha * item["vector"] + (1 - alpha) * item["keyword"]
            if fused <= 1e-9:
                continue
            ranked.append((hit, fused, item["vector"], item["keyword"]))
        ranked.sort(key=lambda row: row[1], reverse=True)

        # 4) MMR 去重（可选）
        if mmr and len(ranked) > k:
            ranked = self._mmr(query, ranked, k)

        chunks = [
            self._make_chunk(hit, score, vec, kw, rank=i + 1)
            for i, (hit, score, vec, kw) in enumerate(ranked[:k])
        ]
        if threshold > 0:
            chunks = [c for c in chunks if c.score >= threshold]
        for i, chunk in enumerate(chunks):
            chunk.rank = i + 1

        strategy = "hybrid+mmr" if (hybrid and mmr) else "hybrid" if hybrid else ("vector+mmr" if mmr else "vector")
        return RetrievalResult(
            query=query,
            chunks=chunks,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            strategy=strategy,
            candidates=len(ranked),
        )

    # ------------------------------------------------------------ MMR
    def _mmr(self, query: str, ranked: list[tuple[StoredHit, float, float, float]], k: int) -> list[tuple[StoredHit, float, float, float]]:
        ids = [row[0].id for row in ranked]
        vectors = self.store.get_embeddings(ids)
        matrix, keep = [], []
        for i, row in enumerate(ranked):
            vector = vectors.get(row[0].id)
            if vector:
                matrix.append(vector)
                keep.append(i)
        if len(keep) <= k:
            return ranked

        doc_vecs = np.asarray(matrix, dtype=np.float32)
        norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        doc_vecs = doc_vecs / norms

        relevance = np.asarray([ranked[i][1] for i in keep], dtype=np.float32)
        if relevance.max() > 0:
            relevance = relevance / relevance.max()

        lam = float(min(1.0, max(0.0, self.settings.mmr_lambda)))
        selected: list[int] = []
        remaining = list(range(len(keep)))

        while remaining and len(selected) < k:
            if not selected:
                best = max(remaining, key=lambda i: relevance[i])
            else:
                selected_vecs = doc_vecs[selected]
                best, best_score = -1, -1e9
                for i in remaining:
                    redundancy = float(np.max(selected_vecs @ doc_vecs[i]))
                    score = lam * float(relevance[i]) - (1 - lam) * redundancy
                    if score > best_score:
                        best, best_score = i, score
            selected.append(best)
            remaining.remove(best)

        return [ranked[keep[i]] for i in selected]
