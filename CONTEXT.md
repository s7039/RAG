# 项目上下文（RAG）

## 目标

健康档案私有知识库：PDF → 切分 → 向量化 → 混合检索 → LLM 回答。
在此基础上补齐 PostgreSQL 存储、LangGraph Agent、效果评测与 Postman 集合。

## 已确定的方案

- **检索**：hybrid（向量 + BM25 融合），本地哈希向量兜底；当前 `.env` 走硅基流动
  `Qwen/Qwen3-Embedding-8B`（4096 维），换模型必须重新灌库。
- **Agent**：LangGraph `StateGraph`（route → plan → act → observe → final）；
  规则 + LLM 双规划器（LLM 计划会做患者/检索词/工具完整性校验）；
  工具失败重规划一次；`MemorySaver` 保存多轮状态。
- **存储**：Chroma 存向量与正文；PostgreSQL 存 chunk 元数据、待办、会话/消息。
- **评测**：`evals/cases.yaml` 25 条，离线可跑；CI 门槛为工具准确率 ≥ 0.8。
- **接口**：FastAPI `/api/*`；Postman 集合在 `docs/postman/`（query 走 mock，4 断言）；
  网页有「普通 RAG / Agent」双模式，默认 RAG。

## 当前进度（2026-11-19）

- Phase 2 PostgreSQL：完成，tag `v0.1.1-pg`（58 chunk / 4 documents）。
- Phase 3 LangGraph Agent：完成，tag `v0.2-agent`（5 个端到端场景）。
- Phase 4 评测：完成，tag `v0.2.1-eval`（25 条；工具 23/23、Top-4 15/15、拒答 2/3）。
- Phase 5 Postman：完成（8 请求、4 断言，newman 全绿）。
- 回归修复：语料患者补丁、相关性拒答、规则 rerank；无 Key 基线 25/25、Top-1 15/15、拒答 3/3。
- 硅基流动接入（2026-11-18）：chat `deepseek-ai/DeepSeek-V4.1-Flash` +
  embedding `Qwen/Qwen3-Embedding-8B`；规则 P95 5.7s、LLM 规划 P95 15.9s，质量仍 25/25、
  Top-1 15/15、拒答 3/3；修复 LLM 计划的上下文/改写/漏工具/due_at 四个问题。
- Agent 总结 + 网页双模式（2026-11-19）：带 `create_task` 的问题由 LLM 基于工具结果总结，
  纯检索题直接返回片段；`/api/agent/chat` 返回 sources/tool_calls/elapsed；网页默认 RAG、可切 Agent。
  在线 P95 10.1s、平均 6.3s，质量仍 25/25、Top-1 15/15、拒答 3/3；pytest 36 个用例。

## 下一步

1. 接真实 reranker（`BAAI/bge-reranker-v2-m3`），替换规则 rerank 后重跑 25 条。
2. 去重：同一事实在 3 份档案里重复，按「患者 + 章节 + 内容哈希」合并。
3. 把无关的 `llama2.pdf` 移出 `input/`，重建索引后重跑评测。
4. 延迟优化：Qwen3-Embedding-8B 单次 0.4–5s，可换 `BAAI/bge-m3`（实测 P95 299ms，Top-1 不变）。
5. 会话记忆换 `PostgresSaver`；长期偏好写 `user_prefs`。
6. Agent 回答改成流式（目前非流式，动作类问题要等 6–12s）。

## 常用命令

```bash
python -m pytest                                           # 36 个用例
python scripts/ingest.py                                   # 灌库（58 chunk）
docker compose up -d db                                    # PostgreSQL
python scripts/migrate_to_pg.py                            # Chroma -> PG
python -m evals.runner                                     # 规则规划评测 -> report.json
python -m evals.runner --online --out report.online.json   # 真实 LLM 规划评测
npx newman run docs/postman/rag-api.postman_collection.json ^
  -e docs/postman/rag-api.postman_environment.json          # Postman 集合
```
