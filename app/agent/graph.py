"""LangGraph 编排：route → plan → act → observe →（重规划 | final）。"""

from __future__ import annotations

import json
import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent.state import AgentState
from app.agent.tools import (
    is_out_of_scope,
    is_reminder,
    known_patients,
    plan_by_rules,
    resolve_patient,
    run_tool,
)
from app.rag.prompt import build_messages
from app.rag.retriever import RetrievedChunk

log = logging.getLogger(__name__)

REFUSAL = (
    "我在健康档案里没有查到相关内容。这类问题不在知识库范围内，"
    "可以换成患者的体检指标、病史、用药、过敏史或复查安排来问。"
)


def _last_text(state: AgentState) -> str:
    m = state["messages"][-1]
    return m.get("content", "") if isinstance(m, dict) else getattr(m, "content", str(m))


def _history(state: AgentState) -> str:
    """最近几轮对话（不含当前这句），给 LLM 规划器补上指代所需的上下文。"""
    rows: list[str] = []
    for m in (state["messages"][:-1] or [])[-4:]:
        if isinstance(m, dict):
            role, content = m.get("role", ""), m.get("content", "")
        else:
            role = getattr(m, "type", "")
            content = getattr(m, "content", "")
        role = {"user": "用户", "human": "用户", "assistant": "助手", "ai": "助手"}.get(role, "")
        if role and content:
            rows.append(f"{role}：{str(content)[:200]}")
    return "\n".join(rows)


def _llm_plan(
    llm, text: str, last_error: str = "", patient: str = "", history: str = ""
) -> list[dict] | None:
    """让模型产出工具调用；解析失败返回 None，交给规则兜底。"""
    if llm is None:
        return None
    try:
        prompt = (
            "你是工具规划器。可用工具：search_knowledge(query, top_k, patient)、"
            "create_task(title, due_at)、parse_time(text)。\n"
            f"当前会话患者：{patient or '未知'}\n"
            f"最近对话：\n{history or '无'}\n"
            "规则：\n"
            "1. search_knowledge 的 query 必须原样使用用户当前这句话，不要改写或只写关键词；\n"
            "2. 用户要求提醒/待办时，必须同时规划 parse_time 与 create_task；\n"
            "3. 只问知识时只规划 search_knowledge；没有把握就输出 []。\n"
            "只输出 JSON 数组，元素形如 {\"name\": \"...\", \"args\": {...}}，不要解释。\n"
            f"用户输入：{text}\n上次错误：{last_error or '无'}"
        )
        out = llm.invoke(prompt)
        raw = out.content if hasattr(out, "content") else str(out)
        calls = json.loads(raw[raw.find("["): raw.rfind("]") + 1])
        if isinstance(calls, list) and calls:
            return [c for c in calls if isinstance(c, dict) and c.get("name")]
    except Exception as exc:                      # noqa: BLE001 - 规划失败一律兜底
        log.info("LLM 规划失败（%s），回退规则规划器", exc.__class__.__name__)
    return None


def _repair_plan(
    calls: list[dict], text: str, last_error: str, patients: list[str], last_patient: str
) -> list[dict]:
    """校验 LLM 计划：工具组合不对就交给规则；患者/检索词做纠正。"""
    names = [c.get("name") for c in calls]
    reminder = is_reminder(text)
    if reminder:
        # 提醒类必须 parse_time + create_task 齐全；漏了就交给规则，否则会漏建待办
        if not {"parse_time", "create_task"} <= set(names):
            return plan_by_rules(text, last_error, patients, last_patient)
    elif names != ["search_knowledge"]:
        # 非提醒类只允许检索；模型多规划了写工具就退回规则
        return plan_by_rules(text, last_error, patients, last_patient)

    repaired: list[dict] = []
    resolved = resolve_patient(text, patients, last_patient)
    for call in calls:
        name = call["name"]
        args = dict(call.get("args") or {})
        if name == "search_knowledge":
            args["query"] = text.strip()   # 不让模型改写检索词，保证评测可复现
            llm_patient = args.get("patient")
            # 优先用当前这句话/上一轮解析出的患者；再退回模型给的合法患者
            patient = resolved or (llm_patient if llm_patient in patients else last_patient) or None
            args["patient"] = patient
        repaired.append({"name": name, "args": args})
    return repaired


def _cite(source: dict) -> str:
    where = source.get("source", "")
    if source.get("page"):
        where += f" 第{source['page']}页"
    return f"（{where}）" if where else ""


def _chunks_from_sources(sources: list[dict]) -> list[RetrievedChunk]:
    """把工具返回的 source dict 还原成 prompt 需要的 RetrievedChunk。"""
    return [
        RetrievedChunk(
            content=src.get("content", ""),
            metadata=src.get("metadata") or {},
            score=float(src.get("score") or 0.0),
            rank=int(src.get("rank") or i + 1),
            source=src.get("source", ""),
            page=src.get("page"),
            page_end=src.get("page_end"),
            section=src.get("section", ""),
            patient=src.get("patient", ""),
            chunk_id=src.get("chunk_id", ""),
        )
        for i, src in enumerate(sources)
    ]


def _synthesize(state: AgentState, llm, service) -> str:
    """有真实 LLM 时基于工具结果写回答；没有/调用失败则回退片段拼接。"""
    results = state.get("results") or []
    sources = [
        src
        for r in results
        if r.get("name") == "search_knowledge" and r.get("ok")
        for src in r.get("sources") or []
    ]
    actions: list[str] = []
    for r in results:
        if r.get("name") == "search_knowledge":
            continue
        if not r.get("ok"):
            actions.append(f"{r.get('name')} 未完成：{r.get('error')}")
        elif r.get("name") == "create_task":
            actions.append(r.get("summary") or "已创建待办")
    if not sources or llm is None:
        return _compose(state, service)
    if not any(r.get("name") == "create_task" for r in results):
        # 纯检索题没有“动作”要汇报，直接片段拼接，省一次 LLM 调用
        return _compose(state, service)

    question = _last_text(state)
    if actions:
        question += (
            "\n\n【系统已执行的操作（已完成，不是建议）】\n"
            + "\n".join(f"- {a}" for a in actions)
            + "\n请在回答中如实告知用户这些操作已完成；不要说无法设置提醒，也不要建议用户自行设置；"
            "操作已成功时直接确认结果，不要再反问时间是否正确。"
        )

    try:
        messages = build_messages(
            question,
            _chunks_from_sources(sources),
            service.settings.max_context_chars,
        )
        out = llm.invoke(messages)
        text = out.content if hasattr(out, "content") else str(out)
        answer = str(text).strip()
    except Exception as exc:  # noqa: BLE001 - 总结失败不能影响工具结果
        log.warning("Agent 总结失败（%s），回退片段拼接", exc.__class__.__name__)
        return _compose(state, service)

    if not answer:
        return _compose(state, service)

    # 模型偶尔漏说已执行的操作，这里补一句兜底（已说则不重复）
    def _missing(action: str) -> bool:
        if action in answer:
            return False
        if action.startswith("已创建待办"):
            # 模型用自己的话说了「已创建待办 #N / 提醒已设置」就不重复
            return not (("待办" in answer or "提醒" in answer) and "无法" not in answer)
        return True

    missing = [a for a in actions if _missing(a)]
    return "\n\n".join([answer, *(f"· {a}" for a in missing)]) if missing else answer


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
        calls = _llm_plan(llm, text, last_error, last_patient, _history(state))
        if calls is None:
            calls = plan_by_rules(text, last_error, names, last_patient)
        else:
            calls = _repair_plan(calls, text, last_error, names, last_patient)
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
        # 先跑 parse_time，保证 create_task 拿到的一定是 ISO 时间（LLM 计划顺序可能反过来）
        plan = sorted(
            state.get("plan") or [],
            key=lambda c: 0 if c.get("name") == "parse_time" else 1,
        )
        for call in plan:
            args = dict(call.get("args") or {})
            # 工具结果在链路里流转：parse_time 的结果自动填进 create_task
            if call.get("name") == "create_task" and carry.get("due_at"):
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
        answer = _synthesize(state, llm, service)
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
