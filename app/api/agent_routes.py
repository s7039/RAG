"""Agent 接口。"""

from __future__ import annotations

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


def _agent(request: Request):
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(503, "Agent 未初始化")
    return agent


@router.post("/agent/chat", response_model=AgentChatOut)
def chat(payload: AgentChatIn, request: Request):
    thread_id = payload.thread_id or uuid.uuid4().hex
    agent = _agent(request)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": payload.message}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    calls = [{"name": c["name"], "args": c.get("args")} for c in (result.get("plan") or [])]
    answer = result.get("final", "")
    log_turn(thread_id, payload.message, answer, calls)
    return AgentChatOut(thread_id=thread_id, answer=answer, tool_calls=calls)
