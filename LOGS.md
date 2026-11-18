# 运行日志

- 2026-11-18 18:33 — `docker compose up -d db` — 成功 — `rag-db-1` healthy，5432 已映射
- 2026-11-18 18:34 — `python scripts/migrate_to_pg.py` — 成功 — documents 4 / chunks 58（tsvec 58）
- 2026-11-18 18:35 — `EXPLAIN ANALYZE select * from chunks where patient='张三九'` — 成功 — Bitmap Index Scan on `chunks_patient_idx`，Execution Time 0.078 ms
- 2026-11-18 18:35 — `docker compose down && docker compose up -d db` — 成功 — 重启后 chunks 仍为 58
- 2026-11-18 18:35 — `python -m pytest` — 成功 — 22 passed in 3.07s（补 Phase 3 前的基线）
- 2026-11-18 18:41 — `python -m pytest tests/test_agent.py -v` — 失败 — `test_multi_turn_thread` patient 为 None（已修复）
- 2026-11-18 18:41 — `python -m pytest` — 成功 — 27 passed in 4.23s
- 2026-11-18 18:43 — `python -m evals.runner` — 成功 — tool 1.0 / Top-4 15/15 / 拒答 2/3 / P95 11.5 ms
- 2026-11-18 18:45 — `npx newman run docs/postman/...` — 失败 — 8 个请求全部 `request url is empty`（已修复）
- 2026-11-18 18:45 — `npx newman run docs/postman/...` — 成功 — requests 8 / failed 0，assertions 4 / failed 0，平均 32ms
- 2026-11-18 18:46 — `python -m ruff check --select E9,F63,F7,F82 app tests evals scripts` — 成功 — All checks passed
- 2026-11-18 18:48 — `docker compose build api` — 成功 — 镜像 `rag-api:dev` 构建通过（含 langgraph 1.2.22 / psycopg 3.3.9）
- 2026-11-18 18:49 — `python -m pytest` — 成功 — 27 passed in 4.43s
- 2026-11-18 18:49 — `python -m evals.runner` — 成功 — tool 1.0 / Top-4 15/15 / 拒答 2/3 / P95 12.5 ms；t04 due 11-20 15:00、t05 due 11-25 09:00
