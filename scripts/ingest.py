#!/usr/bin/env python
"""离线入库脚本：文档加载 -> 切分 -> 向量化 -> 灌入向量数据库。

用法：
    python scripts/ingest.py                     # 重建整个知识库
    python scripts/ingest.py --append            # 追加，不清空
    python scripts/ingest.py --file 健康档案.pdf  # 只灌某个文件
    python scripts/ingest.py --input D:/data     # 指定目录
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.rag.pipeline import RagService  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="健康档案知识库 - 离线入库")
    parser.add_argument("--input", help="文档目录，默认取 INPUT_DIR（input/）")
    parser.add_argument("--file", action="append", help="只处理指定文件，可重复")
    parser.add_argument("--append", action="store_true", help="追加模式，不清空已有向量库")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    print("=" * 64)
    print(f"  健康档案私有知识库 · 离线入库")
    print("=" * 64)
    print(f"文档目录 : {args.input or settings.input_path}")
    print(f"向量库   : {settings.persist_path} (collection={settings.collection_name})")
    print(f"切分参数 : chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}")
    print(f"向量化   : {settings.embed_provider}")

    service = RagService(settings)
    print(f"实际向量化: {service.embedding_info.get('provider')} / {service.embedding_info.get('model')}")
    print("-" * 64)

    stats = service.ingest(reset=not args.append, paths=args.file, input_dir=args.input)

    if stats.warnings:
        for warning in stats.warnings:
            print(f"[提示] {warning}")

    if stats.chunks == 0:
        print("[失败] 没有生成任何 chunk，请检查 input 目录下是否有支持的文档。")
        return 1

    print(f"文件数量 : {stats.files}")
    print(f"页数     : {stats.pages}")
    print(f"Chunk 数 : {stats.chunks}（平均 {stats.chars / max(1, stats.chunks):.0f} 字符/块）")
    print(f"耗时     : 加载 {stats.load_ms:.0f}ms | 切分 {stats.split_ms:.0f}ms | "
          f"向量化 {stats.embed_ms:.0f}ms | 入库 {stats.store_ms:.0f}ms")
    print(f"库内总量 : {stats.collection_count} 条向量")
    print("-" * 64)
    for doc in service.store.document_stats():
        patients = "、".join(doc["patients"]) or "-"
        print(f"  · {doc['source']}  {doc['chunks']} 个片段  第{doc['pages']}页  患者：{patients}")
    print("=" * 64)
    print("入库完成，可运行：python scripts/query.py -q \"张三九的空腹血糖是多少？\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
