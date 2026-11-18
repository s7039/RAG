# Postman 集合（RAG API）

- `rag-api.postman_collection.json`：8 个请求（基础 4 / 检索问答 3 / Agent 1），
  `/api/query` 上 4 条断言（状态码 / `answer` 字段 / `sources` 数组 / 响应时间 < 3000ms）。
- `rag-api.postman_environment.json`：环境变量 `base_url=http://127.0.0.1:8000`。

请求与 `http://127.0.0.1:8000/openapi.json` 的路径、请求体一一对应；没有收录
`/api/ingest` 和 `DELETE /api/index`（有副作用，交给脚本做）。

## 在 Postman 里跑

1. 起服务：`python run.py`（或 `docker compose up -d`）。
2. Postman → Import 两个 json → 右上角选环境「RAG 本地环境」。
3. Collection Runner 选 `RAG API` → Run。

## 命令行跑（newman）

```bash
npx newman run docs/postman/rag-api.postman_collection.json ^
  -e docs/postman/rag-api.postman_environment.json
```

本机 8000 被占用时，临时覆盖 `base_url`：

```bash
npx newman run docs/postman/rag-api.postman_collection.json ^
  -e docs/postman/rag-api.postman_environment.json ^
  --env-var base_url=http://127.0.0.1:8010
```

最近一次结果（2026-11-18，离线模式，58 chunk）：`requests 8 / failed 0`，
`assertions 4 / failed 0`，平均响应 37ms（max 187ms，SSE 那条）。
