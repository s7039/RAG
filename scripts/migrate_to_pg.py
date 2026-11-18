#!/usr/bin/env python
"""把 Chroma 里的 chunk 元数据与正文同步到 PostgreSQL。

幂等：先 TRUNCATE 再全量写入，可以随便重跑。

用法：
    docker compose up -d db
    python scripts/migrate_to_pg.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import text  # noqa: E402

from app.db import SessionLocal, healthcheck  # noqa: E402
from app.rag.pipeline import RagService  # noqa: E402
from app.rag.tokenize import tokenize  # noqa: E402


def main() -> int:
    healthcheck()
    service = RagService()
    rows = service.store.all_rows()
    if not rows:
        print("[失败] Chroma 里没有 chunk，先运行 python scripts/ingest.py")
        return 1

    with SessionLocal() as db:
        # documents 由外键级联清空 chunks
        db.execute(text("TRUNCATE chunks, documents RESTART IDENTITY CASCADE"))
        doc_map: dict[str, int] = {}
        for row in rows:
            meta = row.get("metadata") or {}
            source = str(meta.get("source", ""))
            if source not in doc_map:
                doc_map[source] = db.execute(
                    text(
                        "INSERT INTO documents(source, page, section, patient) "
                        "VALUES (:s, :p, :sec, :pa) RETURNING id"
                    ),
                    {
                        "s": source,
                        "p": meta.get("page"),
                        "sec": meta.get("section"),
                        "pa": meta.get("patient"),
                    },
                ).scalar_one()

            chunk_id = str(meta.get("chunk_id") or row.get("id"))
            content = row.get("document", "")
            # PostgreSQL 没有内置中文分词：先用项目自己的 tokenize 切词，
            # 再交给 to_tsvector('simple', ...)，GIN 索引才对中文有效。
            tsvec_input = " ".join(tokenize(content))
            db.execute(
                text(
                    "INSERT INTO chunks(id, doc_id, ord, content, patient, tsvec) "
                    "VALUES (:id, :d, :o, :c, :p, to_tsvector('simple', :t)) "
                    "ON CONFLICT (id) DO UPDATE SET "
                    "doc_id = EXCLUDED.doc_id, ord = EXCLUDED.ord, content = EXCLUDED.content, "
                    "patient = EXCLUDED.patient, tsvec = EXCLUDED.tsvec"
                ),
                {
                    "id": chunk_id,
                    "d": doc_map[source],
                    "o": int(meta.get("chunk_index", 0)),
                    "c": content,
                    "p": meta.get("patient") or None,
                    "t": tsvec_input,
                },
            )
        db.commit()

        chunks = db.execute(text("SELECT count(*) FROM chunks")).scalar_one()
        documents = db.execute(text("SELECT count(*) FROM documents")).scalar_one()
        with_tsv = db.execute(text("SELECT count(*) FROM chunks WHERE tsvec IS NOT NULL")).scalar_one()

    print(f"documents : {documents}")
    print(f"chunks    : {chunks}（tsvec 已生成 {with_tsv}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
