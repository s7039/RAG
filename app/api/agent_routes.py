"""Agent 接口。"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.agent.memory import log_turn

router = APIRouter(tags=["agent"])


class AgentChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="用户输入")
    thread_id: str | None = Field(None, description="会话 id，用于多轮")


class AgentChatOut(BaseModel):
    thread_id: str
    answer: str
    tool_calls: list[dict] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    elapsed_ms: float = 0.0


def _agent(request: Request):
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(503, "Agent 未初始化")
    return agent


@router.post("/agent/chat", response_model=AgentChatOut)
def chat(payload: AgentChatIn, request: Request):
    thread_id = payload.thread_id or uuid.uuid4().hex
    agent = _agent(request)
    started = time.perf_counter()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": payload.message}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    results = result.get("results") or []
    by_name = {r.get("name"): r for r in results}
    calls = [
        {
            "name": c["name"],
            "args": c.get("args"),
            "ok": by_name.get(c["name"], {}).get("ok"),
            "summary": by_name.get(c["name"], {}).get("summary"),
        }
        for c in (result.get("plan") or [])
    ]

    sources: list[dict] = []
    seen: set = set()
    for r in results:
        for src in r.get("sources") or []:
            key = src.get("chunk_id") or (src.get("source"), src.get("page"), src.get("rank"))
            if key in seen:
                continue
            seen.add(key)
            sources.append(src)

    answer = result.get("final", "")
    log_turn(thread_id, payload.message, answer, calls)
    return AgentChatOut(
        thread_id=thread_id,
        answer=answer,
        tool_calls=calls,
        sources=sources,
        elapsed_ms=round(elapsed_ms, 1),
    )
