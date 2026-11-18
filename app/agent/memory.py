"""会话日志：每轮对话写入 PostgreSQL 的 sessions / messages。

短期对话上下文由 LangGraph 的 MemorySaver 保存（进程内）；这里额外落一份
到 PG，重启后仍能查到历史。写失败不影响主流程（例如本地没起 db 时）。
"""

from __future__ import annotations

import json
import logging

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.db import SessionLocal

log = logging.getLogger(__name__)


def log_turn(thread_id: str, question: str, answer: str, tool_calls: list[dict]) -> None:
    try:
        with SessionLocal() as db:
            db.execute(
                text("INSERT INTO sessions(thread_id) VALUES (:t) ON CONFLICT (thread_id) DO NOTHING"),
                {"t": thread_id},
            )
            db.execute(
                text("INSERT INTO messages(thread_id, role, content) VALUES (:t, 'user', :c)"),
                {"t": thread_id, "c": question},
            )
            db.execute(
                text(
                    "INSERT INTO messages(thread_id, role, content, tool_calls) "
                    "VALUES (:t, 'assistant', :c, CAST(:tc AS JSONB))"
                ),
                {"t": thread_id, "c": answer, "tc": json.dumps(tool_calls, ensure_ascii=False)},
            )
            db.commit()
    except SQLAlchemyError:
        log.warning("会话日志写入失败（PostgreSQL 不可用），已忽略", exc_info=True)
