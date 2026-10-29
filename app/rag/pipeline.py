"""RAG 编排层：把离线入库与在线问答串成完整流程。

离线：文档加载 -> 文档切分 -> 向量化 -> 灌入向量数据库
在线：用户问题 -> 问题向量化 -> 检索 -> 填 prompt -> 调用 LLM -> 生成回复
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Iterable

from app.config import BASE_DIR, Settings, get_settings
from app.rag.bm25 import BM25Index
from app.rag.embeddings import build_embeddings
from app.rag.llm import build_chat_model
from app.rag.loader import discover_files, load_documents
from app.rag.prompt import build_context, build_messages
from app.rag.retriever import Retriever, RetrievedChunk
from app.rag.splitter import split_documents
from app.rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestStats:
    files: int = 0
    pages: int = 0
    chunks: int = 0
    chars: int = 0
    load_ms: float = 0.0
    split_ms: float = 0.0
    embed_ms: float = 0.0
    store_ms: float = 0.0
    total_ms: float = 0.0
    collection_count: int = 0
    embedding: dict = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "files": self.files,
            "pages": self.pages,
            "chunks": self.chunks,
            "chars": self.chars,
            "timings_ms": {
                "load": round(self.load_ms, 1),
                "split": round(self.split_ms, 1),
                "embed": round(self.embed_ms, 1),
                "store": round(self.store_ms, 1),
                "total": round(self.total_ms, 1),
            },
            "collection_count": self.collection_count,
            "embedding": self.embedding,
            "sources": self.sources,
            "warnings": self.warnings,
        }


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[RetrievedChunk]
    provider: dict
    timings_ms: dict
    context_chars: int = 0
    mode: str = "llm"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "mode": self.mode,
            "provider": self.provider,
            "sources": [s.to_dict() for s in self.sources],
            "timings_ms": self.timings_ms,
            "context_chars": self.context_chars,
            "warnings": self.warnings,
        }


class RagService:
    """整个应用共享的单例服务。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.embeddings, self.embedding_info = build_embeddings(self.settings)
        self.store = VectorStore(self.settings, self.embeddings)
        self.bm25 = BM25Index()
        self.retriever = Retriever(self.settings, self.store, self.bm25)
        self._keyword_index_ready = False

    def _ensure_keyword_index(self) -> None:
        """按需构建 BM25 索引（进程重启后第一次检索时构建）。"""
        if self._keyword_index_ready:
            return
        if self.store.count() == 0:
            self._keyword_index_ready = True
            return
        rows = self.store.all_rows()
        self.bm25.build(rows)
        self._keyword_index_ready = True

    def rebuild_keyword_index(self) -> int:
        self._keyword_index_ready = False
        self._ensure_keyword_index()
        return self.bm25.size

    # -------------------------------------------------------- 离线：入库
    def ingest(
        self,
        reset: bool = True,
        paths: Iterable[str] | None = None,
        input_dir: str | None = None,
    ) -> IngestStats:
        started = time.perf_counter()
        stats = IngestStats(embedding=dict(self.embedding_info))
        stats.warnings = []

        if self.embedding_info.get("note"):
            stats.warnings.append(str(self.embedding_info["note"]))

        # 0. 定位文件
        base_dir = Path(input_dir).expanduser() if input_dir else self.settings.input_path
        if not base_dir.is_absolute():
            base_dir = BASE_DIR / base_dir
        if paths:
            targets = [Path(p) for p in paths]
            targets = [p if p.is_absolute() else self.settings.input_path / p for p in targets]
            targets = [p for p in targets if p.exists()]
            if len(targets) < len(list(paths)):
                stats.warnings.append("部分指定文件不存在，已跳过")
        else:
            targets = discover_files(base_dir)

        if not targets:
            stats.total_ms = (time.perf_counter() - started) * 1000
            stats.collection_count = self.store.count()
            stats.warnings.append(f"目录 {base_dir} 下没有可加载的文件")
            return stats

        # 1. 文档加载
        t0 = time.perf_counter()
        docs = load_documents(base_dir, targets)
        stats.load_ms = (time.perf_counter() - t0) * 1000
        stats.files = len({d.metadata.get("source") for d in docs})
        stats.pages = len(docs)

        # 2. 文档切分
        t0 = time.perf_counter()
        chunks = split_documents(
            docs,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            min_chunk_size=self.settings.min_chunk_size,
        )
        stats.split_ms = (time.perf_counter() - t0) * 1000
        stats.chunks = len(chunks)
        stats.chars = sum(len(d.page_content) for d in chunks)
        stats.sources = sorted({str(d.metadata.get("source")) for d in chunks})

        if not chunks:
            stats.total_ms = (time.perf_counter() - started) * 1000
            stats.collection_count = self.store.count()
            stats.warnings.append("切分后没有产生任何 chunk，请检查文档内容")
            return stats

        # 3. 向量化
        t0 = time.perf_counter()
        texts = [d.page_content for d in chunks]
        vectors: list[list[float]] = []
        batch = max(1, self.settings.embed_batch_size)
        for start in range(0, len(texts), batch):
            vectors.extend(self.embeddings.embed_documents(texts[start : start + batch]))
            logger.info("向量化进度 %s/%s", min(start + batch, len(texts)), len(texts))
        stats.embed_ms = (time.perf_counter() - t0) * 1000
        if vectors:
            stats.embedding["dim"] = len(vectors[0])

        # 4. 灌入向量数据库
        if reset:
            self.store.reset()
        t0 = time.perf_counter()
        ids = [str(d.metadata.get("chunk_id")) for d in chunks]
        metadatas = [
            {k: v for k, v in d.metadata.items() if isinstance(v, (str, int, float, bool))}
            for d in chunks
        ]
        self.store.add_embeddings(ids, texts, metadatas, vectors)
        stats.store_ms = (time.perf_counter() - t0) * 1000
        stats.collection_count = self.store.count()
        self.rebuild_keyword_index()

        # 5. 记录索引元数据（用于检测向量模型变更）
        self.store.write_meta(
            {
                "collection": self.settings.collection_name,
                "embedding": stats.embedding,
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
                "files": stats.files,
                "pages": stats.pages,
                "chunks": stats.chunks,
                "sources": stats.sources,
            }
        )

        stats.total_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "入库完成：%s 个文件 / %s 页 / %s 个 chunk，用时 %.1fs",
            stats.files,
            stats.pages,
            stats.chunks,
            stats.total_ms / 1000,
        )
        return stats

    # -------------------------------------------------------- 在线：问答
    def _consistency_warning(self) -> str | None:
        meta = self.store.read_meta()
        saved = meta.get("embedding") or {}
        current = self.embedding_info
        if saved and (saved.get("provider") != current.get("provider") or saved.get("model") != current.get("model")):
            return (
                f"当前向量化配置（{current.get('provider')}/{current.get('model')}）"
                f"与建库时（{saved.get('provider')}/{saved.get('model')}）不一致，"
                "检索结果可能失真，建议重新灌库。"
            )
        return None

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        use_mmr: bool | None = None,
        patient: str | None = None,
    ) -> Answer:
        question = (question or "").strip()
        if not question:
            raise ValueError("问题不能为空")

        warnings: list[str] = []
        consistency = self._consistency_warning()
        if consistency:
            warnings.append(consistency)

        # 1. 检索（内部完成问题向量化）
        self._ensure_keyword_index()
        where = {"patient": patient} if patient else None
        retrieval = self.retriever.retrieve(question, top_k=top_k, where=where, use_mmr=use_mmr)

        # 2. 拼 prompt
        t0 = time.perf_counter()
        messages = build_messages(question, retrieval.chunks, self.settings.max_context_chars)
        prompt_ms = (time.perf_counter() - t0) * 1000
        context_chars = sum(len(m.content) for m in messages)

        # 3. 调用 LLM
        t0 = time.perf_counter()
        llm, provider_meta = build_chat_model(
            self.settings,
            provider_name=provider,
            model=model,
            temperature=temperature,
        )
        try:
            message = llm.invoke(messages)
            answer_text = message.content if isinstance(message.content, str) else str(message.content)
        except Exception as exc:  # pragma: no cover - 网络/鉴权异常
            logger.exception("调用大模型失败")
            raise RuntimeError(f"调用大模型失败：{exc}") from exc
        llm_ms = (time.perf_counter() - t0) * 1000

        if provider_meta.get("note"):
            warnings.append(str(provider_meta["note"]))
        if not retrieval.chunks:
            warnings.append("未检索到相关资料，回答可能不准确")

        return Answer(
            question=question,
            answer=answer_text,
            sources=retrieval.chunks,
            provider=provider_meta,
            timings_ms={
                "retrieve": round(retrieval.elapsed_ms, 1),
                "prompt": round(prompt_ms, 1),
                "llm": round(llm_ms, 1),
                "total": round(retrieval.elapsed_ms + prompt_ms + llm_ms, 1),
            },
            context_chars=context_chars,
            mode=str(provider_meta.get("mode", "llm")),
            warnings=warnings,
        )

    async def astream_answer(
        self,
        question: str,
        top_k: int | None = None,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        use_mmr: bool | None = None,
        patient: str | None = None,
    ) -> AsyncIterator[dict]:
        """流式问答：先给检索结果，再逐 token 推答案。"""
        question = (question or "").strip()
        if not question:
            raise ValueError("问题不能为空")

        started = time.perf_counter()
        warnings: list[str] = []
        consistency = self._consistency_warning()
        if consistency:
            warnings.append(consistency)

        where = {"patient": patient} if patient else None
        self._ensure_keyword_index()
        retrieval = self.retriever.retrieve(question, top_k=top_k, where=where, use_mmr=use_mmr)

        llm, provider_meta = build_chat_model(
            self.settings,
            provider_name=provider,
            model=model,
            temperature=temperature,
            streaming=True,
        )
        messages = build_messages(question, retrieval.chunks, self.settings.max_context_chars)

        yield {
            "event": "retrieval",
            "data": {
                "sources": [c.to_dict() for c in retrieval.chunks],
                "strategy": retrieval.strategy,
                "elapsed_ms": round(retrieval.elapsed_ms, 1),
                "provider": provider_meta,
                "warnings": warnings,
            },
        }

        try:
            async for piece in llm.astream(messages):
                text = piece.content if isinstance(piece.content, str) else str(piece.content)
                if text:
                    yield {"event": "token", "data": {"text": text}}
        except Exception as exc:  # pragma: no cover
            logger.exception("流式生成失败")
            yield {"event": "error", "data": {"message": f"生成失败：{exc}"}}
            return

        yield {
            "event": "done",
            "data": {
                "mode": provider_meta.get("mode", "llm"),
                "provider": provider_meta,
                "timings_ms": {"total": round((time.perf_counter() - started) * 1000, 1)},
                "warnings": warnings,
            },
        }

    # -------------------------------------------------------- 状态
    def status(self) -> dict:
        meta = self.store.read_meta()
        return {
            "collection": self.settings.collection_name,
            "chunks": self.store.count(),
            "documents": self.store.document_stats(),
            "embedding": {**self.embedding_info, **{k: v for k, v in (meta.get("embedding") or {}).items() if v}},
            "indexed_at": meta.get("updated_at"),
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "top_k": self.settings.top_k,
            "use_mmr": self.settings.use_mmr,
            "retrieve_mode": self.settings.retrieve_mode,
            "hybrid_alpha": self.settings.hybrid_alpha,
            "keyword_index_docs": self.bm25.size,
            "persist_dir": str(self.settings.persist_path),
        }
