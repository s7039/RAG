#!/usr/bin/env python
"""导出 LangGraph 流程图（Mermaid），粘进 README。

用法：
    python scripts/export_graph.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent.graph import build_graph, graph_mermaid  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.rag.pipeline import RagService  # noqa: E402


def main() -> int:
    agent = build_graph(RagService(get_settings()))
    print(graph_mermaid(agent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
