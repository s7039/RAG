# 项目上下文（RAG）

## 目标

健康档案私有知识库：PDF → 切分 → 向量化 → 混合检索 → LLM 回答。
在此基础上补齐 PostgreSQL 存储、LangGraph Agent、效果评测与 Postman 集合。

## 已确定的方案

- **检索**：hybrid（向量 + BM25 融合），本地哈希向量兜底，无 Key 也能跑。
- **Agent**：LangGraph `StateGraph`（route → plan → act → observe → final）；
  规则 + LLM 双规划器；工具失败重规划一次；`MemorySaver` 保存多轮状态。
- **存储**：Chroma 存向量与正文；PostgreSQL 存 chunk 元数据、待办、会话/消息。
- **评测**：`evals/cases.yaml` 25 条，离线可跑；CI 门槛为工具准确率 ≥ 0.8。
- **接口**：FastAPI `/api/*`；Postman 集合在 `docs/postman/`。

## 当前进度（2026-11-18）

- Phase 2 PostgreSQL：完成，tag `v0.1.1-pg`（58 chunk / 4 documents）。
- Phase 3 LangGraph Agent：完成，tag `v0.2-agent`（5 个端到端场景）。
- Phase 4 评测：完成，tag `v0.2.1-eval`（25 条；工具 23/23、Top-4 15/15、拒答 2/3）。
- Phase 5 Postman：完成（8 请求、4 断言，newman 全绿）。

## 下一步

1. 修源 PDF 的患者归属（`健康档案(含表格01/02).pdf` 混入王五数据），重建索引。
2. 给回答加相关性门槛，把拒答从 2/3 提到 3/3。
3. 接真实 embedding / rerank，把 Top-1 从 13/15 提到 15/15。
4. 会话记忆换 `PostgresSaver`；长期偏好写 `user_prefs`。

## 常用命令

```bash
python -m pytest                                           # 27 个用例
python scripts/ingest.py                                   # 灌库（58 chunk）
docker compose up -d db                                    # PostgreSQL
python scripts/migrate_to_pg.py                            # Chroma -> PG
python -m evals.runner                                     # 评测
npx newman run docs/postman/rag-api.postman_collection.json ^
  -e docs/postman/rag-api.postman_environment.json          # Postman 集合
```
