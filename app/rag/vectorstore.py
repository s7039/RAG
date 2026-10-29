"""向量数据库层（Chroma）。

职责很单一：把「向量 + 文本 + 元数据」存进去，按相似度取回来。
这里直接使用 chromadb 的 PersistentClient，好处是：
- 数据落盘在 ``data/chroma``，重启不丢；
- 可以显式控制 embedding（我们自己算好再写入），换向量模型时更透明；
- 支持按元数据过滤（例如只看某个患者的记录）。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import Settings

logger = logging.getLogger(__name__)

_ALLOWED_META = (str, int, float, bool)


@dataclass
class StoredHit:
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float

    @property
    def similarity(self) -> float:
        """cosine 距离 -> 相似度（0~1，越大越相关）。"""
        return max(0.0, min(1.0, 1.0 - float(self.distance)))


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, _ALLOWED_META):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


class VectorStore:
    """Chroma 持久化向量库的薄封装。"""

    def __init__(self, settings: Settings, embeddings: Embeddings) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.persist_dir = settings.persist_path
        self.collection_name = settings.collection_name
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._client = None
        self._collection = None

    # ------------------------------------------------------------ 基础
    @property
    def client(self):
        if self._client is None:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )
        return self._collection

    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception:  # pragma: no cover
            return 0

    def reset(self) -> None:
        """删除并重建集合（重新灌库时使用）。"""
        with self._lock:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._collection = None
            _ = self.collection

    # ------------------------------------------------------------ 写入
    def add_embeddings(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict],
        vectors: list[list[float]],
    ) -> int:
        """把已算好的向量写入 Chroma（批量 upsert，可重复执行）。"""
        if not texts:
            return 0
        batch = max(1, self.settings.embed_batch_size)
        total = 0
        with self._lock:
            for start in range(0, len(texts), batch):
                self.collection.upsert(
                    ids=ids[start : start + batch],
                    documents=texts[start : start + batch],
                    metadatas=metadatas[start : start + batch],
                    embeddings=vectors[start : start + batch],
                )
                total += len(texts[start : start + batch])
        return total

    def add_documents(self, docs: Iterable[Document]) -> int:
        docs = list(docs)
        if not docs:
            return 0

        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict] = []
        for doc in docs:
            meta = _sanitize_metadata(doc.metadata)
            ids.append(str(meta.get("chunk_id") or f"chunk-{len(ids)}"))
            texts.append(doc.page_content)
            metadatas.append(meta)

        batch = max(1, self.settings.embed_batch_size)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch):
            vectors.extend(self.embeddings.embed_documents(texts[start : start + batch]))
            logger.info("向量化 %s/%s", min(start + batch, len(texts)), len(texts))
        return self.add_embeddings(ids, texts, metadatas, vectors)

    # ------------------------------------------------------------ 查询
    def query(
        self,
        query_text: str,
        top_k: int = 4,
        where: dict | None = None,
        n_results: int | None = None,
    ) -> list[StoredHit]:
        k = max(1, n_results or top_k)
        query_vector = self.embeddings.embed_query(query_text)
        raw = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[StoredHit] = []
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        ids = (raw.get("ids") or [[]])[0]
        for i, text in enumerate(documents):
            hits.append(
                StoredHit(
                    id=ids[i] if i < len(ids) else str(i),
                    text=text or "",
                    metadata=dict(metadatas[i] or {}) if i < len(metadatas) else {},
                    distance=float(distances[i]) if i < len(distances) else 1.0,
                )
            )
        return hits

    # ------------------------------------------------------------ 元信息
    def get_by_ids(self, ids: list[str]) -> list[StoredHit]:
        """按 id 取回文本与元数据（用于补齐只在 BM25 命中的文档）。"""
        if not ids:
            return []
        raw = self.collection.get(ids=ids, include=["documents", "metadatas"])
        hits: list[StoredHit] = []
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        for i, doc_id in enumerate(raw.get("ids") or []):
            hits.append(
                StoredHit(
                    id=doc_id,
                    text=documents[i] or "" if i < len(documents) else "",
                    metadata=dict(metadatas[i] or {}) if i < len(metadatas) and metadatas[i] else {},
                    distance=1.0,
                )
            )
        return hits

    def get_embeddings(self, ids: list[str]) -> dict[str, list[float]]:
        """按 id 取回已存向量（MMR 重排用）。"""
        if not ids:
            return {}
        raw = self.collection.get(ids=ids, include=["embeddings"])
        rows = raw.get("embeddings")
        result: dict[str, list[float]] = {}
        if rows is None:
            return result
        for doc_id, vector in zip(raw.get("ids") or [], rows):
            if vector is None:
                continue
            result[str(doc_id)] = list(vector)
        return result

    def all_rows(self) -> list[dict]:
        """导出全部 chunk，用于构建 BM25 索引。"""
        try:
            raw = self.collection.get(include=["documents", "metadatas"])
        except Exception:  # pragma: no cover
            return []
        rows = []
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        for i, doc_id in enumerate(raw.get("ids") or []):
            rows.append(
                {
                    "id": doc_id,
                    "document": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                }
            )
        return rows

    @property
    def meta_file(self) -> Path:
        return self.persist_dir / "index_meta.json"

    def read_meta(self) -> dict:
        try:
            return json.loads(self.meta_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def write_meta(self, data: dict) -> None:
        payload = {**data, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.meta_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def document_stats(self) -> list[dict]:
        """按文件聚合 chunk 数、患者、页码范围。"""
        try:
            raw = self.collection.get(include=["metadatas"])
        except Exception:  # pragma: no cover
            return []
        stats: dict[str, dict] = {}
        for meta in raw.get("metadatas") or []:
            if not meta:
                continue
            source = str(meta.get("source", "unknown"))
            item = stats.setdefault(
                source,
                {
                    "source": source,
                    "doc_title": meta.get("doc_title", ""),
                    "doc_type": meta.get("doc_type", ""),
                    "chunks": 0,
                    "pages": set(),
                    "patients": set(),
                    "sections": set(),
                },
            )
            item["chunks"] += 1
            if meta.get("page") is not None:
                item["pages"].add(int(meta["page"]))
            if meta.get("patient"):
                item["patients"].add(str(meta["patient"]))
            if meta.get("section"):
                item["sections"].add(str(meta["section"]))
        result = []
        for item in sorted(stats.values(), key=lambda x: x["source"]):
            pages = sorted(item["pages"])
            result.append(
                {
                    "source": item["source"],
                    "doc_title": item["doc_title"],
                    "doc_type": item["doc_type"],
                    "chunks": item["chunks"],
                    "pages": f"{pages[0]}-{pages[-1]}" if pages else "",
                    "patients": sorted(item["patients"]),
                    "sections": sorted(item["sections"]),
                }
            )
        return result
