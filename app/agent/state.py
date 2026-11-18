"""Agent 图的状态定义。"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]   # 必须用 add_messages，否则每步覆盖
    intent: Literal["answer", "tool"]
    plan: list[dict]          # [{name, args}]
    results: list[dict]       # [{name, ok, summary, sources, error}]
    last_error: str
    retries: int
    final: str
    patient: str              # 当前会话指向的患者，供多轮指代使用
