"""FastAPI 应用入口。

启动方式：
    python -m uvicorn app.main:app --reload --port 8000
或：
    python run.py
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.config import BASE_DIR, get_settings
from app.rag.pipeline import RagService

logger = logging.getLogger("rag")
WEB_DIR = Path(__file__).resolve().parent / "web"


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    started = time.perf_counter()
    logger.info("正在初始化 RAG 服务…")
    service = RagService(settings)
    app.state.rag = service
    logger.info(
        "初始化完成：向量库 %s 个 chunk，向量化 %s/%s，用时 %.2fs",
        service.store.count(),
        service.embedding_info.get("provider"),
        service.embedding_info.get("model"),
        time.perf_counter() - started,
    )
    logger.info("Web 界面： http://127.0.0.1:8000/   接口文档： http://127.0.0.1:8000/docs")
    yield
    logger.info("RAG 服务已关闭")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "基于 LangChain + FastAPI + OpenAI / DeepSeek 的私有领域知识库（RAG）示例。\n\n"
            "离线：文档加载 → 切分 → 向量化 → 写入向量数据库\n\n"
            "在线：问题向量化 → 检索 → 填充 Prompt → 调用大模型 → 生成回答"
        ),
        lifespan=lifespan,
    )

    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix=settings.api_prefix)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):  # pragma: no cover
        logger.exception("未处理的异常: %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": f"服务器内部错误：{exc}"})

    @app.get("/api", include_in_schema=False)
    def api_root() -> dict:
        return {
            "name": settings.app_name,
            "docs": "/docs",
            "endpoints": [
                "GET  /api/health",
                "GET  /api/status",
                "GET  /api/providers",
                "POST /api/ingest",
                "POST /api/retrieve",
                "POST /api/query",
                "POST /api/query/stream",
                "GET  /api/documents",
                "DELETE /api/index",
            ],
        }

    if WEB_DIR.exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    settings = get_settings()
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True, reload_dirs=[str(BASE_DIR / "app")])
