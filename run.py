#!/usr/bin/env python
"""一键启动 FastAPI 服务。

    python run.py                 # 127.0.0.1:8000，自动重载
    python run.py --port 8080     # 指定端口
    python run.py --no-reload     # 关闭自动重载
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="启动健康档案 RAG 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    settings = get_settings()
    print(f"{settings.app_name} v{settings.app_version}")
    print(f"  Web 界面 : http://{args.host}:{args.port}/")
    print(f"  接口文档 : http://{args.host}:{args.port}/docs")
    print(f"  文档目录 : {settings.input_path}")
    print(f"  向量库   : {settings.persist_path}")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        reload_dirs=[str(ROOT / "app")],
    )


if __name__ == "__main__":
    main()
