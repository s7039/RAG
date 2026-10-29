#!/usr/bin/env python
"""在线问答脚本：问题 -> 向量化 -> 检索 -> 填 prompt -> 调用 LLM -> 回复。

用法：
    python scripts/query.py -q "张三九的空腹血糖是多少？"
    python scripts/query.py -q "李四六对什么过敏？" --patient 李四六 -k 5
    python scripts/query.py -q "王五的既往病史" --provider deepseek --model deepseek-chat
    python scripts/query.py -q "..." --no-stream --show-sources --mmr
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.rag.pipeline import RagService  # noqa: E402

args: argparse.Namespace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="健康档案知识库 - 在线问答")
    parser.add_argument("-q", "--question", required=True, help="用户问题")
    parser.add_argument("-k", "--top-k", type=int, help="召回数量")
    parser.add_argument("--patient", help="只检索某位患者")
    parser.add_argument("--provider", help="deepseek / openai / custom / mock")
    parser.add_argument("--model", help="覆盖默认模型")
    parser.add_argument("--temperature", type=float)
    mmr_group = parser.add_mutually_exclusive_group()
    mmr_group.add_argument("--mmr", dest="mmr", action="store_true", help="启用 MMR 去重重排")
    mmr_group.add_argument("--no-mmr", dest="mmr", action="store_false", help="关闭 MMR 重排")
    parser.set_defaults(mmr=None)
    parser.add_argument("--no-stream", action="store_true", help="一次性返回，不流式输出")
    parser.add_argument("--show-sources", action="store_true", help="打印检索到的原文片段")
    return parser.parse_args()


def print_sources(sources) -> None:
    print("-" * 64)
    print(f"检索到 {len(sources)} 条资料：")
    for chunk in sources:
        where = f"{chunk.source} 第{chunk.page}页 · {chunk.section}"
        print(f"  [{chunk.rank}] 相似度 {chunk.score:.3f} "
              f"(向量 {chunk.vector_score:.3f} / 关键词 {chunk.keyword_score:.3f}) | {where} | 患者 {chunk.patient or '-'}")
        if args.show_sources:
            preview = chunk.content.strip().replace("\n", " ")
            print(f"      {preview[:200]}…")


def run_stream(service: RagService) -> None:
    async def _run() -> None:
        async for item in service.astream_answer(
            question=args.question,
            top_k=args.top_k,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            use_mmr=args.mmr,
            patient=args.patient,
        ):
            if item["event"] == "retrieval":
                data = item["data"]
                provider = data.get("provider") or {}
                print(f"检索策略：{data['strategy']}，命中 {len(data['sources'])} 条，"
                      f"用时 {data['elapsed_ms']:.0f}ms")
                print(f"生成模型：{provider.get('provider_label')} / {provider.get('model')} "
                      f"({provider.get('mode')})")
                print("-" * 64)
            elif item["event"] == "token":
                sys.stdout.write(item["data"]["text"])
                sys.stdout.flush()
            elif item["event"] == "error":
                print(f"\n[错误] {item['data']['message']}")
        print()

    asyncio.run(_run())


def run_once(service: RagService) -> None:
    result = service.ask(
        question=args.question,
        top_k=args.top_k,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        use_mmr=args.mmr,
        patient=args.patient,
    )
    provider = result.provider
    print(f"生成模型：{provider.get('provider_label')} / {provider.get('model')} ({result.mode})")
    print("-" * 64)
    print(result.answer)
    print("-" * 64)
    timings = result.timings_ms
    print(f"耗时：检索 {timings.get('retrieve')}ms | prompt {timings.get('prompt')}ms | "
          f"生成 {timings.get('llm')}ms | 合计 {timings.get('total')}ms")
    for warning in result.warnings:
        print(f"[提示] {warning}")
    print_sources(result.sources)


def main() -> int:
    global args
    args = parse_args()
    settings = get_settings()
    service = RagService(settings)

    if service.store.count() == 0:
        print("[提示] 向量库为空，请先运行：python scripts/ingest.py")
        return 1

    if args.no_stream:
        run_once(service)
    else:
        run_stream(service)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
