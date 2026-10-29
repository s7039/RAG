"""FastAPI 路由：入库、检索、问答（含 SSE 流式）、状态查询。"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from app.rag.llm import list_provider_status

logger = logging.getLogger(__name__)
router = APIRouter()


def get_service(request: Request):
    service = getattr(request.app.state, "rag", None)
    if service is None:  # pragma: no cover - 应用未初始化
        raise HTTPException(status_code=503, detail="RAG 服务尚未初始化")
    return service


# ---------------------------------------------------------------- 健康检查
@router.get("/health", summary="健康检查")
def health(request: Request) -> dict:
    service = get_service(request)
    settings = service.settings
    providers = list_provider_status(settings)
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "collection": settings.collection_name,
        "chunks": service.store.count(),
        "embedding": service.embedding_info,
        "default_provider": settings.default_provider,
        "mock_active": providers["mock_active"],
    }


@router.get("/status", summary="知识库状态")
def status(request: Request) -> dict:
    service = get_service(request)
    return {"success": True, "data": service.status()}


@router.get("/providers", summary="可用大模型列表")
def providers(request: Request) -> dict:
    service = get_service(request)
    return {"success": True, "data": list_provider_status(service.settings)}


# ---------------------------------------------------------------- 离线：入库
@router.post("/ingest", response_model=IngestResponse, summary="重建知识库（离线流程）")
def ingest(payload: IngestRequest, request: Request) -> IngestResponse:
    service = get_service(request)
    try:
        stats = service.ingest(reset=payload.reset, paths=payload.paths, input_dir=payload.input_dir)
    except Exception as exc:  # pragma: no cover
        logger.exception("灌库失败")
        raise HTTPException(status_code=500, detail=f"灌库失败：{exc}") from exc

    if stats.chunks == 0:
        return IngestResponse(
            success=False,
            message="没有可入库的内容：" + "；".join(stats.warnings or ["请检查 input 目录"]),
            stats=stats.to_dict(),
        )
    return IngestResponse(
        success=True,
        message=f"入库完成：{stats.files} 个文件、{stats.pages} 页、{stats.chunks} 个 chunk",
        stats=stats.to_dict(),
    )


@router.delete("/index", summary="清空向量库")
def clear_index(request: Request) -> dict:
    service = get_service(request)
    service.store.reset()
    try:
        service.store.meta_file.unlink(missing_ok=True)
    except OSError:  # pragma: no cover
        pass
    return {"success": True, "message": "向量库已清空", "chunks": service.store.count()}


@router.get("/documents", summary="已入库文档列表")
def documents(request: Request) -> dict:
    service = get_service(request)
    docs = service.store.document_stats()
    return {"success": True, "count": len(docs), "data": docs}


# ---------------------------------------------------------------- 在线：检索
@router.post("/retrieve", response_model=RetrieveResponse, summary="仅检索（调试用）")
def retrieve(payload: RetrieveRequest, request: Request) -> RetrieveResponse:
    service = get_service(request)
    where = {"patient": payload.patient} if payload.patient else None
    result = service.retriever.retrieve(
        payload.query, top_k=payload.top_k, where=where, use_mmr=payload.use_mmr
    )
    return RetrieveResponse(
        query=result.query,
        strategy=result.strategy,
        elapsed_ms=round(result.elapsed_ms, 2),
        count=len(result.chunks),
        chunks=[c.to_dict() for c in result.chunks],
    )


# ---------------------------------------------------------------- 在线：问答
@router.post("/query", response_model=QueryResponse, summary="RAG 问答")
def query(payload: QueryRequest, request: Request) -> QueryResponse:
    service = get_service(request)
    try:
        answer = service.ask(
            question=payload.question,
            top_k=payload.top_k,
            provider=payload.provider,
            model=payload.model,
            temperature=payload.temperature,
            use_mmr=payload.use_mmr,
            patient=payload.patient,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    body = answer.to_dict()
    return QueryResponse(**body)


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/query/stream", summary="RAG 问答（SSE 流式）")
async def query_stream(payload: QueryRequest, request: Request) -> StreamingResponse:
    service = get_service(request)

    async def event_source():
        try:
            async for item in service.astream_answer(
                question=payload.question,
                top_k=payload.top_k,
                provider=payload.provider,
                model=payload.model,
                temperature=payload.temperature,
                use_mmr=payload.use_mmr,
                patient=payload.patient,
            ):
                yield _sse(item["event"], item["data"])
        except ValueError as exc:
            yield _sse("error", {"message": str(exc)})
        except KeyError as exc:
            yield _sse("error", {"message": str(exc)})
        except Exception as exc:  # pragma: no cover
            logger.exception("流式问答失败")
            yield _sse("error", {"message": f"生成失败：{exc}"})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
