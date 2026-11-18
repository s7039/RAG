"""跑评测集，产出 report.json。

用法：
    python -m evals.runner                 # 离线（本地向量 + 规则规划器）
    python -m evals.runner --online        # 用配置里的真实 LLM 做规划
    python -m evals.runner --fail-under 0.8  # 工具准确率低于阈值时退出码 1（默认 0.8）

指标口径：
- 工具选择准确率：23 条端到端口径的用例里，实际计划与 expect_tools 完全一致的比例
- 参数正确率：25 条用例里，pydantic 校验通过且 patient 关键参数正确的比例
  （2 条参数校验用例以「按预期报错」计为正确）
- 引用命中：15 条带 expect_source 的用例里，Top-1 / Top-4 是否出现期望的 文件+页码
- 拒答正确率：3 条库外用例里，回答是否明确说没查到
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import yaml
from pydantic import ValidationError

from app.agent.graph import build_graph
from app.agent.tools import TOOL_SPECS, run_tool
from app.rag.pipeline import RagService

BASE = Path(__file__).resolve().parent
REFUSAL_MARKERS = ("没查到", "没有查到", "无法回答", "不在知识库范围")


def _same_source(source: dict, expect: dict) -> bool:
    return source.get("source") == expect.get("file") and source.get("page") == expect.get("page")


def _params_ok(calls: list[dict], patient: str) -> bool:
    for call in calls:
        spec = TOOL_SPECS.get(call.get("name"))
        if spec is None:
            return False
        try:
            parsed = spec.model_validate(call.get("args") or {})
        except ValidationError:
            return False
        if call["name"] == "search_knowledge" and patient and parsed.patient != patient:
            return False
    return True


def _validation_row(case: dict, service) -> dict:
    result = run_tool(case["tool"], case.get("args") or {}, service)
    expected = case.get("expect_error", "参数不合法")
    ok = not result.get("ok") and expected in (result.get("error") or "")
    return {
        "id": case["id"],
        "kind": "validation",
        "difficulty": case.get("difficulty", "normal"),
        "question": f"{case['tool']}({case.get('args')})",
        "expect_tools": [],
        "actual_tools": [],
        "tool_ok": None,
        "param_ok": ok,
        "source_hit_top1": None,
        "source_hit_top4": None,
        "refused": None,
        "refusal_ok": None,
        "answer": result.get("error", ""),
        "ms": 0.0,
    }


def _qa_row(case: dict, agent, service) -> dict:
    config = {"configurable": {"thread_id": f"eval-{case['id']}"}}
    for prime in case.get("turns") or []:
        agent.invoke({"messages": [{"role": "user", "content": prime}]}, config=config)

    started = time.perf_counter()
    out = agent.invoke(
        {"messages": [{"role": "user", "content": case["question"]}]}, config=config
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    calls = out.get("plan") or []
    actual = [c.get("name") for c in calls]
    sources = [s for r in (out.get("results") or []) for s in r.get("sources") or []]
    answer = out.get("final", "")

    expect_source = case.get("expect_source")
    hit_top1 = bool(expect_source and sources and _same_source(sources[0], expect_source))
    hit_top4 = bool(expect_source and any(_same_source(s, expect_source) for s in sources[:4]))
    refused = any(marker in answer for marker in REFUSAL_MARKERS)

    return {
        "id": case["id"],
        "kind": "qa",
        "difficulty": case.get("difficulty", "normal"),
        "question": case["question"],
        "expect_tools": case.get("expect_tools") or [],
        "actual_tools": actual,
        "tool_ok": sorted(actual) == sorted(case.get("expect_tools") or []),
        "param_ok": _params_ok(calls, case.get("patient", "")),
        "source_hit_top1": hit_top1 if expect_source else None,
        "source_hit_top4": hit_top4 if expect_source else None,
        "refused": refused,
        "refusal_ok": refused if case.get("expect_refusal") else None,
        "answer": answer,
        "ms": round(elapsed_ms, 1),
    }


def summarize(rows: list[dict]) -> dict:
    qa = [r for r in rows if r["kind"] == "qa"]
    latencies = [r["ms"] for r in qa]
    source_rows = [r for r in qa if r["source_hit_top4"] is not None]
    refusal_rows = [r for r in qa if r["refusal_ok"] is not None]
    n = len(rows)
    qa_n = len(qa)
    return {
        "cases": n,
        "qa_cases": qa_n,
        "tool_accuracy": round(sum(r["tool_ok"] for r in qa) / qa_n, 3),
        "param_accuracy": round(sum(r["param_ok"] for r in rows) / n, 3),
        "source_hit_top1": f"{sum(r['source_hit_top1'] for r in source_rows)}/{len(source_rows)}",
        "source_hit_top4": f"{sum(r['source_hit_top4'] for r in source_rows)}/{len(source_rows)}",
        "top1_rate": round(sum(r["source_hit_top1"] for r in source_rows) / len(source_rows), 3),
        "top4_rate": round(sum(r["source_hit_top4"] for r in source_rows) / len(source_rows), 3),
        "refusal_accuracy": f"{sum(r['refusal_ok'] for r in refusal_rows)}/{len(refusal_rows)}",
        "p95_ms": round(sorted(latencies)[int(0.95 * (qa_n - 1))], 1),
        "avg_ms": round(statistics.mean(latencies), 1),
        "avg_tool_calls": round(statistics.mean(len(r["actual_tools"]) for r in qa), 2),
        "failed_cases": [r["id"] for r in rows if r["tool_ok"] is False or r["param_ok"] is False
                         or r["refusal_ok"] is False],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="跑 Agent 评测集")
    parser.add_argument("--online", action="store_true", help="用真实 LLM 做规划")
    parser.add_argument("--fail-under", type=float, default=0.8, help="工具准确率下限")
    parser.add_argument("--out", default="report.json", help="结果文件名（相对 evals/）")
    args = parser.parse_args(argv)

    cases = yaml.safe_load((BASE / "cases.yaml").read_text(encoding="utf-8"))
    service = RagService()
    llm = None
    if args.online:
        from app.rag.llm import build_chat_model

        llm, meta = build_chat_model(service.settings)
        if meta.get("mode") != "llm":
            print("[warn] 没有可用的真实模型 Key，本次仍用规则规划器")
            llm = None
    agent = build_graph(service, llm=llm)

    rows: list[dict] = []
    for case in cases:
        if case.get("kind") == "validation":
            rows.append(_validation_row(case, service))
        else:
            rows.append(_qa_row(case, agent, service))

    summary = summarize(rows)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = BASE / out_path
    out_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for row in rows:
        if row["tool_ok"] is False or row["param_ok"] is False or row["refusal_ok"] is False:
            print(f"[FAIL] {row['id']}: tools={row['actual_tools']} answer={row['answer'][:80]!r}")

    return 0 if summary["tool_accuracy"] >= args.fail_under else 1


if __name__ == "__main__":
    raise SystemExit(main())
