"""LangGraph 编排：route → plan → act → observe →（重规划 | final）。"""

from __future__ import annotations

import json
import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentState
from app.agent.tools import (
    is_out_of_scope,
    known_patients,
    plan_by_rules,
    resolve_patient,
    run_tool,
)

log = logging.getLogger(__name__)

REFUSAL = (
    "我在健康档案里没有查到相关内容。这类问题不在知识库范围内，"
    "可以换成患者的体检指标、病史、用药、过敏史或复查安排来问。"
)


def _last_text(state: AgentState) -> str:
    m = state["messages"][-1]
    return m.get("content", "") if isinstance(m, dict) else getattr(m, "content", str(m))


def _llm_plan(llm, text: str, last_error: str = "") -> list[dict] | None:
    """让模型产出工具调用；解析失败返回 None，交给规则兜底。"""
    if llm is None:
        return None
    try:
        prompt = (
            "你是工具规划器。可用工具：search_knowledge(query, top_k, patient)、"
            "create_task(title, due_at)、parse_time(text)。"
            "只输出 JSON 数组，元素形如 {\"name\": \"...\", \"args\": {...}}，不要解释。\n"
            f"用户输入：{text}\n上次错误：{last_error or '无'}"
        )
        out = llm.invoke(prompt)
        raw = out.content if hasattr(out, "content") else str(out)
        calls = json.loads(raw[raw.find("["): raw.rfind("]") + 1])
        if isinstance(calls, list) and calls:
            return [c for c in calls if isinstance(c, dict) and c.get("name")]
    except Exception:                      # noqa: BLE001 - 规划失败一律兜底
        log.warning("LLM 规划失败，回退规则规划器", exc_info=True)
    return None


def _cite(source: dict) -> str:
    where = source.get("source", "")
    if source.get("page"):
        where += f" 第{source['page']}页"
    return f"（{where}）" if where else ""


def _compose(state: AgentState, service) -> str:
    results = state.get("results") or []
    if not results:
        return REFUSAL

    errors = [r for r in results if not r.get("ok")]
    if errors and len(errors) == len(results):
        return f"没查到：{errors[0].get('error')}。可以换个说法，或告诉我具体患者名。"

    lines: list[str] = []
    idx = 0
    for r in results:
        if not r.get("ok"):
            lines.append(f"· {r.get('name')} 未完成：{r.get('error')}")
            continue
        if r["name"] == "search_knowledge":
            for src in r.get("sources", [])[:3]:
                idx += 1
                lines.append(f"[{idx}] {src.get('content', '')}{_cite(src)}")
        elif r["name"] == "parse_time":
            # 时间解析是中间结果，已经并进 create_task 的摘要里
            continue
        else:
            lines.append(f"· {r.get('summary', '')}")
    return "\n".join(lines) or REFUSAL


def build_graph(service, llm=None, checkpointer=None):
    # 患者名单按 chunk 数缓存：灌库后自动失效，避免每次提问都扫一遍全量元数据
    cache = {"count": -1, "patients": []}

    def patients() -> list[str]:
        count = service.store.count()
        if count != cache["count"]:
            cache.update(count=count, patients=known_patients(service))
        return cache["patients"]

    def route(state: AgentState) -> dict:
        # 每轮开始重置上一轮的规划/错误，避免上一轮的重试次数影响这一轮
        return {
            "intent": "answer" if is_out_of_scope(_last_text(state)) else "tool",
            "plan": [],
            "results": [],
            "last_error": "",
            "retries": 0,
        }

    def plan(state: AgentState) -> dict:
        text = _last_text(state)
        last_error = state.get("last_error", "")
        last_patient = state.get("patient", "")
        names = patients()
        calls = _llm_plan(llm, text, last_error) or plan_by_rules(
            text, last_error, names, last_patient
        )
        # 模型漏填患者时，用上一轮的患者补上，多轮追问才不会跑偏
        for call in calls:
            args = call.setdefault("args", {})
            if call.get("name") == "search_knowledge" and not args.get("patient") and last_patient:
                args["patient"] = last_patient
        retries = state.get("retries", 0) + (1 if last_error else 0)
        return {
            "plan": calls,
            "retries": retries,
            "last_error": "",
            "patient": resolve_patient(text, names, last_patient) or last_patient,
        }

    def act(state: AgentState) -> dict:
        results: list[dict] = []
        carry: dict = {}
        for call in state.get("plan") or []:
            args = dict(call.get("args") or {})
            # 工具结果在链路里流转：parse_time 的结果自动填进 create_task
            if call.get("name") == "create_task" and not args.get("due_at") and carry.get("due_at"):
                args["due_at"] = carry["due_at"]
            res = run_tool(call.get("name", ""), args, service)
            if call.get("name") == "parse_time" and res.get("ok"):
                carry["due_at"] = res["summary"]
            results.append(res)
        # 只读工具失败才值得重规划；已经创建了待办就不要再跑一遍，避免重复写
        wrote = any(r["name"] == "create_task" and r.get("ok") for r in results)
        err = "" if wrote else next((r["error"] for r in results if not r.get("ok")), "")
        return {"results": results, "last_error": err}

    def observe(state: AgentState) -> dict:
        return {}

    def final(state: AgentState) -> dict:
        answer = _compose(state, service)
        return {"final": answer, "messages": [{"role": "assistant", "content": answer}]}

    def after_route(state: AgentState) -> str:
        return "plan" if state.get("intent") == "tool" else "final"

    def after_observe(state: AgentState) -> str:
        if state.get("last_error") and state.get("retries", 0) < 1:
            return "plan"
        return "final"

    g = StateGraph(AgentState)
    g.add_node("route", route)
    g.add_node("plan", plan)
    g.add_node("act", act)
    g.add_node("observe", observe)
    g.add_node("final", final)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", after_route, {"plan": "plan", "final": "final"})
    g.add_edge("plan", "act")
    g.add_edge("act", "observe")
    g.add_conditional_edges("observe", after_observe, {"plan": "plan", "final": "final"})
    g.add_edge("final", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def graph_mermaid(app) -> str:
    """导出流程图（放进 README）。"""
    return app.get_graph().draw_mermaid()
