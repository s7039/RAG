"""Agent 工具：检索 / 待办 / 时间解析，以及离线规则规划器。"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.db import SessionLocal

log = logging.getLogger(__name__)


class SearchArgs(BaseModel):
    query: str = Field(..., min_length=1, description="要检索的问题")
    top_k: int = Field(4, ge=1, le=10, description="返回片段数")
    patient: str | None = Field(None, description="患者姓名，限定检索范围")


class CreateTaskArgs(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, description="待办内容")
    due_at: str | None = Field(None, description="ISO8601 时间，如 2026-11-18T15:00:00")


class ParseTimeArgs(BaseModel):
    text: str = Field(..., min_length=1, description="中文时间表达，如 明天下午三点")


TOOL_SPECS = {
    "search_knowledge": SearchArgs,
    "create_task": CreateTaskArgs,
    "parse_time": ParseTimeArgs,
}


_CN_HOURS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
             "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12}


def _to_hour(raw: str) -> int:
    raw = raw.strip()
    hour = int(raw) if raw.isdigit() else _CN_HOURS.get(raw, 9)
    # 「99 点」这种离谱输入退回默认值，避免 datetime.replace 抛异常
    return hour if 0 <= hour <= 23 else 9


def _parse_time(text: str) -> str:
    now = datetime.now()
    day = now
    if "后天" in text:
        day = now + timedelta(days=2)
    elif "明天" in text:
        day = now + timedelta(days=1)
    hour = 9
    m = re.search(r"(上午|中午|下午|晚上)?\s*([一二三四五六七八九十\d]{1,3})\s*[点时]", text)
    if m:
        hour = _to_hour(m.group(2))
        if m.group(1) in ("下午", "晚上") and hour < 12:
            hour += 12
        elif m.group(1) == "中午" and hour < 11:
            hour += 12
    return day.replace(hour=hour, minute=0, second=0, microsecond=0).isoformat()


def run_tool(name: str, args: dict, service) -> dict:
    spec = TOOL_SPECS.get(name)
    if spec is None:
        return {"name": name, "ok": False, "error": f"未知工具 {name}"}
    try:
        parsed = spec(**args)
    except ValidationError as exc:
        msg = exc.errors()[0]
        return {"name": name, "ok": False,
                "error": f"参数不合法：{msg['loc']} {msg['msg']}"}

    if name == "search_knowledge":
        if service is None:
            return {"name": name, "ok": False, "error": "检索服务未就绪"}
        # 进程重启后 BM25 索引是空的；这里复用 RagService 的懒加载逻辑
        service._ensure_keyword_index()
        retrieval = service.retriever.retrieve(
            parsed.query, top_k=parsed.top_k,
            where={"patient": parsed.patient} if parsed.patient else None,
        )
        sources = [c.to_dict() for c in retrieval.chunks]
        if not sources:
            return {"name": name, "ok": False, "error": "检索无结果，可能需要换关键词或放宽条件"}
        return {"name": name, "ok": True, "sources": sources,
                "summary": sources[0].get("content", "")[:120]}

    if name == "create_task":
        try:
            with SessionLocal() as db:
                row = db.execute(
                    text("INSERT INTO tasks(title, due_at) VALUES (:t, :d) RETURNING id"),
                    {"t": parsed.title, "d": parsed.due_at},
                ).scalar_one()
                db.commit()
        except SQLAlchemyError as exc:
            # 不把 DSN / 驱动细节抛给调用方
            log.warning("写入 tasks 失败", exc_info=True)
            return {"name": name, "ok": False, "error": f"待办库不可用（{exc.__class__.__name__}）"}
        return {"name": name, "ok": True,
                "summary": f"已创建待办 #{row}：{parsed.title}（{parsed.due_at or '无截止时间'}）"}

    if name == "parse_time":
        iso = _parse_time(parsed.text)
        return {"name": name, "ok": True, "summary": iso}

    return {"name": name, "ok": False, "error": "未实现"}


# ---------------------------------------------------------------- 规划器
# 明显不属于健康档案的问题，直接拒答，不去检索
_OUT_OF_SCOPE_RE = re.compile(r"(天气|气温|下雨|股票|股价|汇率|彩票|新闻|笑话|星座)")
# 追问/指代：出现这些词且没提新患者时，沿用上一轮的患者
_FOLLOWUP_RE = re.compile(r"(他|她|其|该患者|这个患者|那位患者|那|还有|再)")
_REMINDER_RE = re.compile(r"(提醒|待办|记得|安排|明天|后天|下周)")
# 同时问知识 + 让记待办时，先检索再建待办
_ASK_RE = re.compile(r"(是多少|多少|查一下|帮我查|查查|怎么样|怎么办|是什么|有什么|哪一|哪页|建议|报告)")


def is_out_of_scope(text: str) -> bool:
    return bool(_OUT_OF_SCOPE_RE.search(text or ""))


def known_patients(service) -> list[str]:
    """从已入库文档里拿到患者名单，避免在代码里写死人名。"""
    try:
        docs = service.store.document_stats()
    except Exception:  # pragma: no cover - 向量库不可用时退化为空
        log.warning("读取患者名单失败", exc_info=True)
        return []
    return sorted({p for doc in docs for p in doc.get("patients") or []})


def resolve_patient(text: str, patients: list[str] | None, last_patient: str = "") -> str:
    """本轮提到的患者；没提但用了「他/那/还有」时沿用上一轮。"""
    patients = patients or []
    patient = next((p for p in patients if p in text), "")
    if not patient and last_patient and _FOLLOWUP_RE.search(text):
        return last_patient
    return patient


def plan_by_rules(text: str, last_error: str = "", patients: list[str] | None = None,
                  last_patient: str = "") -> list[dict]:
    """离线/兜底规划器：没有大模型也能跑通整条图。"""
    t = (text or "").strip()
    if not t or is_out_of_scope(t):
        return []

    patient = resolve_patient(t, patients, last_patient)

    if _REMINDER_RE.search(t):
        plans: list[dict] = []
        if _ASK_RE.search(t):
            plans.append({"name": "search_knowledge",
                          "args": {"query": t, "top_k": 4, "patient": patient or None}})
        m = re.search(r"(提醒我?|记得|安排)\s*(.+)", t)
        title = (m.group(2) if m else t).strip()
        plans.append({"name": "parse_time", "args": {"text": t}})
        plans.append({"name": "create_task", "args": {"title": title}})
        return plans

    # 第一次失败就放宽 top_k 重试一次，这就是最小可用的「失败重规划」
    top_k = 8 if last_error else 4
    return [{"name": "search_knowledge",
             "args": {"query": t, "top_k": top_k, "patient": patient or None}}]
