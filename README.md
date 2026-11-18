# 健康档案私有知识库 RAG

![ci](https://github.com/s7039/RAG/actions/workflows/ci.yml/badge.svg)

基于 **LangChain + FastAPI + OpenAI / DeepSeek + Chroma** 的私有领域知识库（RAG）示例项目。
以 `input/` 目录下的健康档案 PDF 为语料，完整跑通「离线建库 + 在线问答」全流程，
并支持同时接入多种大模型、流式输出、引用溯源与患者级过滤，以及 LangGraph Agent
（检索 / 待办 / 时间解析三类工具、失败重规划、会话落 PostgreSQL）。


---

## 一、项目做了什么

### 离线流程（建库）

```
input/*.pdf
   │ ① 文档加载      app/rag/loader.py       pdfplumber 解析，表格转 Markdown
   ▼
页级 Document（含来源、页码）
   │ ② 文档切分      app/rag/splitter.py     按「一、二、三…」章节切、表格不截断、识别患者
   ▼
Chunk（含 来源/页码/章节/患者 元数据）
   │ ③ 向量化        app/rag/embeddings.py   OpenAI embedding 或本地哈希向量
   ▼
向量
   │ ④ 灌入向量数据库 app/rag/vectorstore.py  Chroma 持久化 + BM25 索引
   ▼
data/chroma/
```

### 在线流程（问答）

```
用户问题
   │ ① 问题向量化    同 ③
   ▼
   ├─ ② 向量检索     Chroma cosine 相似度召回
   └─ ② BM25 召回    关键词精确匹配（指标名、专有名词更稳）
          │
          ▼
   ③ 加权融合 + 可选 MMR 去重   app/rag/retriever.py
          ▼
   ④ 填充 Prompt 模板            app/rag/prompt.py
          ▼
   ⑤ 调用 LLM（DeepSeek / OpenAI / 自定义 / 离线模拟）  app/rag/llm.py
          ▼
  回答 + 引用编号 + 原文出处
```
---

## 二、目录结构

```
RAG/
├─ app/
│  ├─ main.py                 # FastAPI 入口（lifespan 初始化服务、挂载静态页）
│  ├─ config.py               # pydantic-settings 配置（读 .env / 环境变量）
│  ├─ db.py                   # PostgreSQL 连接与会话
│  ├─ api/
│  │  ├─ routes.py            # 所有 HTTP 接口
│  │  ├─ agent_routes.py      # Agent 接口（/agent/chat）
│  │  └─ schemas.py           # 请求 / 响应模型
│  ├─ agent/
│  │  ├─ state.py             # 图状态（messages / plan / results / patient）
│  │  ├─ tools.py             # 检索 / 待办 / 时间解析 + 规则规划器
│  │  ├─ graph.py             # LangGraph 编排与重规划
│  │  └─ memory.py            # 会话日志写 PostgreSQL
│  ├─ rag/
│  │  ├─ loader.py            # ① 文档加载（PDF/DOCX/TXT/MD/CSV）
│  │  ├─ splitter.py          # ② 文档切分（结构 + 表格 + 患者）
│  │  ├─ embeddings.py        # ③ 向量化（OpenAI / 本地哈希）
│  │  ├─ vectorstore.py       # ④ 向量数据库（Chroma）
│  │  ├─ bm25.py              # 关键词检索
│  │  ├─ tokenize.py          # 中文友好分词（中英混合）
│  │  ├─ retriever.py         # 混合检索 + MMR 重排 + 轻量 rerank
│  │  ├─ prompt.py            # Prompt 模板与上下文拼装
│  │  ├─ llm.py               # 多模型接入 + 离线模拟模型
│  │  ├─ patient_overrides.py # 语料补丁：修正表格 PDF 的患者归属
│  │  ├─ providers.py         # 模型供应商注册表
│  │  └─ pipeline.py          # 离线 / 在线流程编排（RagService）
│  └─ web/index.html          # 内置聊天界面（无 CDN 依赖）
├─ scripts/
│  ├─ ingest.py               # 命令行灌库
│  ├─ migrate_to_pg.py        # Chroma chunk -> PostgreSQL
│  ├─ export_graph.py         # 导出 LangGraph 流程图（Mermaid）
│  └─ query.py                # 命令行问答
├─ db/
│  └─ init.sql                # PostgreSQL 建表（documents/chunks/tasks/sessions/messages/user_prefs）
├─ evals/
│  ├─ cases.yaml              # 25 条评测集
│  ├─ runner.py               # 跑评测，产出 report.json
│  ├─ report.json             # 最近一次结果
│  └─ REPORT.md               # 指标 + Bad Case 归因
├─ tests/                     # pytest 测试（36 个用例，含 9 个 Agent 场景）
├─ docs/postman/              # Postman 集合（8 请求 + 4 条断言）
├─ input/                     # 健康档案语料（PDF）
├─ data/chroma/               # 向量库落盘目录（可重建，已在 .gitignore）
├─ requirements.txt
├─ .env.example
├─ pytest.ini
├─ run.py                     # 一键启动服务
└─ README.md
```

---

## 三、快速开始

### 1. 环境要求

- Python 3.10+（本项目在 3.13 上验证通过）
- 无需 GPU

### 2. 安装依赖

```bash
cd D:\workspace2\RAG

# 建议先建虚拟环境
python -m venv .venv
.venv\Scripts\activate          # PowerShell

python -m pip install -r requirements.txt

# 国内网络较慢时：
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 配置模型（可选）

```bash
copy .env.example .env
```

编辑 `.env`，至少填一个 Key：

```ini
# 用 DeepSeek 生成
DEEPSEEK_API_KEY=sk-xxxxxxxx
DEFAULT_PROVIDER=deepseek

# 或 OpenAI（同时提供向量化）
OPENAI_API_KEY=sk-xxxxxxxx
EMBED_PROVIDER=openai
```

> 不配置任何 Key 也能跑：会自动降级为 **本地哈希向量 + 离线模拟模型**，
> 用来验证整条链路。真实问答请务必配置 Key。

### 4. 灌库

```bash
python scripts/ingest.py
```

输出示例：

```
文件数量 : 4
页数     : 15
Chunk 数 : 58（平均 385 字符/块）
耗时     : 加载 753ms | 切分 2ms | 向量化 16ms | 入库 57ms
库内总量 : 58 条向量
  · llama2.pdf  27 个片段  第1-5页  患者：-
  · 健康档案(含表格01).pdf  3 个片段  第1-1页  患者：张三九
  · 健康档案(含表格02).pdf  7 个片段  第1-3页  患者：张三九
  · 健康档案.pdf  21 个片段  第1-6页  患者：张三九、李四六、王五
```

### 5. 启动服务

```bash
python run.py
# 或
python -m uvicorn app.main:app --reload --port 8000
```

打开 <http://127.0.0.1:8000/>

- 聊天界面：<http://127.0.0.1:8000/>
- 接口文档（Swagger）：<http://127.0.0.1:8000/docs>

### 6. 命令行问答

```bash
python scripts/query.py -q "张三九的空腹血糖是多少？"
python scripts/query.py -q "李四六对什么过敏？" --patient 李四六 --show-sources
python scripts/query.py -q "王五做过什么手术？" --no-stream
```

### 7. Docker 一键启动（可选）

```bash
docker compose up -d --build
curl -X POST http://127.0.0.1:8000/api/ingest -H "Content-Type: application/json" -d "{}"   # 首次灌库
```

- 克隆下来直接跑即可：`.env` 可选（`env_file` 标了 `required: false`），没有时走离线模拟 + 本地向量
- 要接真实模型：`copy .env.example .env` 填 Key，再 `docker compose up -d`
- 健康检查：`curl http://127.0.0.1:8000/api/health`，`chunks` 应为 58
- 索引落在卷 `rag_data`（挂到 `/app/data`），`docker compose down && docker compose up -d` 后仍在
- 容器内以非 root 用户 `appuser` 运行；`.env` 用 `env_file` 注入，不打进镜像

### 8. PostgreSQL（chunk 元数据 / 会话 / 待办）

`docker compose up -d db` 会起 PostgreSQL 16，首次启动自动执行 `db/init.sql` 建表
（`documents` / `chunks` / `tasks` / `sessions` / `messages` / `user_prefs`）。

把 Chroma 里已有的 58 个 chunk 同步进去（幂等，可重跑）：

```bash
docker compose up -d db
python scripts/migrate_to_pg.py
```

查询验证：

```bash
docker compose exec db psql -U rag -d rag -c "select count(*) from chunks;"
docker compose exec db psql -U rag -d rag -c "EXPLAIN ANALYZE select * from chunks where patient='张三九';"
```

> 中文全文检索 PostgreSQL 不内置分词：`tsvec` 由 `app/rag/tokenize.py` 先切词，
> 再 `to_tsvector('simple', ...)` 写入，`chunks_tsv_idx` 是 GIN 索引。
> 58 行的小表上 PG 仍选择 `Bitmap Index Scan`（患者过滤命中 17 行），索引确实生效。

### 9. Postman 集合（可选）

`docs/postman/` 下有 8 个请求的集合与 `base_url` 环境，`/api/query` 带 4 条断言：

```bash
npx newman run docs/postman/rag-api.postman_collection.json ^
  -e docs/postman/rag-api.postman_environment.json
```

也可以直接在 Postman 里 Import 两个 json 后跑 Collection Runner，详见
[docs/postman/README.md](docs/postman/README.md)。

---

## 四、接口说明

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查（向量库条数、向量模型、是否离线模式） |
| GET | `/api/status` | 知识库状态（文档列表、切片参数、检索配置） |
| GET | `/api/providers` | 可用大模型供应商与 Key 配置情况 |
| POST | `/api/ingest` | 重建知识库（离线流程） |
| POST | `/api/retrieve` | 只检索，不生成（调参用） |
| POST | `/api/query` | RAG 问答 |
| POST | `/api/query/stream` | RAG 问答（SSE 流式） |
| POST | `/api/agent/chat` | Agent 问答（工具调用 / 多轮 / 待办） |
| GET | `/api/documents` | 已入库文档列表（患者、章节、页码） |
| DELETE | `/api/index` | 清空向量库 |

### 问答

```bash
curl -X POST http://127.0.0.1:8000/api/query ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"张三九的空腹血糖是多少？\",\"top_k\":4,\"provider\":\"deepseek\"}"
```

```json
{
  "question": "张三九的空腹血糖是多少？",
  "answer": "张三九最近一次体检（2024-08-15）空腹血糖为 6.8 mmol/L，略高于参考范围 4.0–6.1 [1]……",
  "mode": "llm",
  "provider": {"provider": "deepseek", "model": "deepseek-chat", "mode": "llm"},
  "sources": [
    {
      "rank": 1,
      "score": 0.551,
      "vector_score": 0.215,
      "keyword_score": 0.887,
      "source": "健康档案.pdf",
      "page": 1,
      "section": "四、体检与化验结果",
      "patient": "张三九",
      "content": "【健康档案 · 张三九 · 四、体检与化验结果】……"
    }
  ],
  "timings_ms": {"retrieve": 2.5, "prompt": 0.1, "llm": 1200.4, "total": 1203.0},
  "warnings": []
}
```

### 流式（SSE）

`POST /api/query/stream` 依次返回三类事件：

```
event: retrieval   # 检索结果 + 使用的模型
data: {...}

event: token       # 逐段文本
data: {"text": "张三九"}

event: done        # 结束，附耗时
data: {...}
```

前端 `app/web/index.html` 就是按这个协议渲染的。

---

## 五、多模型接入

所有供应商都通过 OpenAI 兼容协议访问，切换只需改配置：

| 供应商 | 配置项 | 说明 |
| --- | --- | --- |
| DeepSeek | `DEEPSEEK_API_KEY` / `DEEPSEEK_CHAT_MODEL` | 默认生成模型 `deepseek-chat`，也可用 `deepseek-reasoner` |
| OpenAI | `OPENAI_API_KEY` / `OPENAI_CHAT_MODEL` | 生成 + 向量化 |
| 自定义 | `CUSTOM_BASE_URL` / `CUSTOM_CHAT_MODEL` / `CUSTOM_API_KEY` | 通义千问、Moonshot、vLLM、Ollama 等任何兼容服务 |
| 离线模拟 | `ALLOW_MOCK_LLM=true` | 无 Key 时从检索结果抽取句子作答，用于演示 |

**关于向量化**：DeepSeek 官方暂未提供 embedding 接口，所以 `EMBED_PROVIDER` 是独立配置：

```ini
EMBED_PROVIDER=auto    # 有 OPENAI_API_KEY 用 OpenAI，否则用本地哈希向量
EMBED_PROVIDER=openai  # 强制 OpenAI
EMBED_PROVIDER=local   # 强制本地向量（1024 维，无需联网）
```

> 换向量模型后**必须重新灌库**（`python scripts/ingest.py`），否则查询向量和库里的向量不在同一空间。
> 程序会在索引元数据里记录建库时的向量模型，不一致时在响应 `warnings` 里提示。
> Docker 里换模型后执行 `docker compose exec api python scripts/ingest.py`（或 `POST /api/ingest`），
> 否则 `rag_data` 卷里还是旧模型建的 58 条向量。

### 接入硅基流动（一个 Key 同时做 chat + embedding）

```ini
CUSTOM_LABEL=硅基流动
CUSTOM_BASE_URL=https://api.siliconflow.cn/v1
CUSTOM_API_KEY=sk-...
CUSTOM_CHAT_MODEL=deepseek-ai/DeepSeek-V4.1-Flash
CUSTOM_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
EMBED_PROVIDER=custom
DEFAULT_PROVIDER=custom
DEFAULT_MODEL=
```

注意：`OPENAI_API_KEY` 非空时 embedding 会优先走 OpenAI；`DEFAULT_MODEL` 非空会覆盖
`CUSTOM_CHAT_MODEL`。模型 ID 以控制台当前开放的为准，改了模型必须重新灌库。

---

## 六、检索策略

```
score = HYBRID_ALPHA × 向量相似度 + (1 - HYBRID_ALPHA) × BM25 分数
```

- `RETRIEVE_MODE=hybrid`（默认）：向量 + BM25 双路召回融合；
- `RETRIEVE_MODE=vector`：纯向量检索；
- `USE_MMR=true`：开启 MMR（最大边际相关性）重排，减少返回内容重复；
- `patient=张三九`：按患者过滤，只在该患者的记录里检索；
- 轻量规则 rerank：数值题偏好「数值 + 临床单位/日期」的片段，建议题偏好「医生建议」章节。
  这是规则加权，不是 cross-encoder；接真实 reranker 时替换 `_answer_type_bonus` 即可。

实测（本地向量，无 API Key）：

| 问题 | Top-1 命中 |
| --- | --- |
| 张三九最近一次体检的空腹血糖是多少？ | 健康档案.pdf 第1页 · 四、体检与化验结果 ✅ |
| 李四六对哪些药物过敏？ | 健康档案.pdf 第2页 · 李四六 医疗历史 ✅ |
| 王五做过什么手术？ | 健康档案.pdf 第4页 · 王五 医疗历史 ✅ |
| 医生给张三九的饮食建议是什么 | 健康档案.pdf 第2页 · 六、医生建议 ✅ |

---

## 七、Agent 与工具调用（LangGraph）

`POST /api/agent/chat` 把「检索问答」升级成「按需调工具」：先判断意图，再规划工具链，
执行后回看结果，失败就重规划一次。

```mermaid
graph TD;
	__start__([__start__]):::first
	route(route)
	plan(plan)
	act(act)
	observe(observe)
	final(final)
	__end__([__end__]):::last
	__start__ --> route;
	act --> observe;
	observe -.-> final;
	observe -.-> plan;
	plan --> act;
	route -.-> final;
	route -.-> plan;
	final --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

| 工具 | 入参（pydantic 校验） | 作用 |
| --- | --- | --- |
| `search_knowledge` | `query`、`top_k`(1–10)、`patient` | 混合检索，返回带出处的片段 |
| `create_task` | `title`、`due_at` | 往 PostgreSQL `tasks` 写待办 |
| `parse_time` | `text` | 把「明天下午三点」解析成 ISO8601 |

要点：

- **双规划器**：配置了真实 Key 时先让 LLM 输出 JSON 计划；没有 Key 或解析失败时回退规则规划器，
  断网也能跑通整条图。
- **计划校验/纠偏**：LLM 的计划会再过一遍规则——检索词必须用用户原句、患者必须存在或沿用上一轮、
  提醒类必须同时有 `parse_time + create_task`，否则整份计划回退规则规划器。
- **失败重规划**：只读工具失败 → `top_k` 放宽到 8 重跑一次；如果已经创建过待办则不再重跑，避免重复写。
- **相关性门槛**：检索片段里一个实词都对不上问题（去掉患者名后）就判「库里没有」，
  直接回「没查到」，不让无关片段硬答。
- **多轮指代**：LangGraph `MemorySaver` 按 `thread_id` 保存 state，追问「那他的饮食建议呢」会沿用上一轮的患者。
- **会话落库**：每轮问答写入 PostgreSQL 的 `sessions` / `messages`（`tool_calls` 存 JSONB），进程重启后仍可查。
- **最终回答**：纯检索题直接返回带 `[1]` 引用的片段，省一次 LLM 调用；只有带 `create_task` 的“动作类”
  问题才让 LLM 基于检索结果 + 已执行操作写总结（否则模型容易说“无法设置提醒”）。
- **网页双模式**：`app/web/index.html` 的「问答模式」默认普通 RAG（流式），可切 Agent；
  Agent 模式会显示工具链和耗时。
- **离线可测**：没有 API Key 时用离线模拟模型作答，9 个 Agent 用例（检索 / 建待办 / 多轮 / 参数校验 /
  库外拒答 / 库内无此项 / 计划校验 / 中文时间）全过；总测试数 36。

> 配了真实 LLM 后：普通 RAG 约 2–8 s；Agent 纯检索约 4–8 s（一次规划），建待办约 6–12 s
> （规划 + 汇总）。无 Key 走规则 + 片段拼接，毫秒级。
> 评测里两种模式都记录在 [evals/REPORT.md](evals/REPORT.md)。

```bash
curl -X POST http://127.0.0.1:8000/api/agent/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"明天下午三点提醒我复查血糖\"}"
```

```json
{
  "thread_id": "…",
  "answer": "· 已创建待办 #6：复查血糖（2026-11-20T15:00:00）",
  "tool_calls": [
    {"name": "parse_time", "args": {"text": "明天下午三点提醒我复查血糖"}, "ok": true, "summary": "2026-11-20T15:00:00"},
    {"name": "create_task", "args": {"title": "复查血糖"}, "ok": true, "summary": "已创建待办 #6：复查血糖（2026-11-20T15:00:00）"}
  ],
  "sources": [],
  "elapsed_ms": 5710.2
}
```

流程图重新生成：`python scripts/export_graph.py`。

---

## 八、配置项速查

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `INPUT_DIR` | `input` | 文档目录 |
| `PERSIST_DIR` | `data/chroma` | 向量库落盘目录 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `600` / `100` | 切分长度与重叠（字符） |
| `TOP_K` | `4` | 最终送入 Prompt 的片段数 |
| `FETCH_K` | `20` | 每路召回的候选数 |
| `RETRIEVE_MODE` | `hybrid` | `hybrid` / `vector` |
| `HYBRID_ALPHA` | `0.5` | 向量权重（0~1） |
| `USE_MMR` / `MMR_LAMBDA` | `false` / `0.6` | MMR 开关与相关/多样性权重 |
| `MAX_CONTEXT_CHARS` | `6000` | 参考资料总字符预算 |
| `LLM_TEMPERATURE` | `0.2` | 生成温度 |
| `DEFAULT_PROVIDER` | `deepseek` | 默认供应商 |
| `EMBED_PROVIDER` | `auto` | `auto` / `openai` / `local` |
| `ALLOW_MOCK_LLM` | `true` | 无 Key 时是否允许离线模拟 |
| `POSTGRES_DSN` | `postgresql+psycopg://rag:rag@localhost:5432/rag` | PostgreSQL 连接串（容器内由 compose 覆盖为 `@db`） |

完整列表见 `app/config.py` 与 `.env.example`。

---

## 九、测试

```bash
python -m pytest -v
```

覆盖：切分与患者识别、表格保护、页码回溯、语料患者补丁、BM25 排序与过滤、端到端入库检索、
轻量 rerank、HTTP 接口（含 SSE 流式）、空库提问，以及 Agent 的 9 个用例（检索 / 建待办 /
多轮指代 / 参数校验 / 库外拒答 / 库内无此项 / 计划纠偏 / 中文时间）。总 36 个用例，全部使用
本地向量 + 离线模拟模型，**不需要网络和 API Key**；建待办那条需要本地 `db` 容器，
没起 PostgreSQL 时会 skip。

> 测试 fixture 会显式清空 `CUSTOM_* / OPENAI / DEEPSEEK` 配置，避免本机 `.env` 里的真实 Key
> 被测试偷偷调用。

---

## 十、评测体系

`evals/` 下 25 条用例：单轮精确 10 / 多轮指代 5 / 工具组合 5 / 拒答 3 / 参数校验 2。

```bash
python -m evals.runner                            # 规则规划（无 Key 也能跑）
python -m evals.runner --online                   # 真实 LLM 规划
python -m evals.runner --online --out report.online.json
```

| 配置 | 工具 / 参数 | Top-1 / Top-4 | 拒答 | P95 |
| --- | --- | --- | --- | --- |
| 无 Key：本地哈希 + 规则 | 23/23 · 25/25 | 15/15 · 15/15 | 3/3 | 12.3 ms |
| 硅基流动 Qwen3-8B + 规则 | 23/23 · 25/25 | 15/15 · 15/15 | 3/3 | 5.7 s |
| 硅基流动 Qwen3-8B + LLM 规划 | 23/23 · 25/25 | 15/15 · 15/15 | 3/3 | 10.1 s |

> 在线模式里只有带 `create_task` 的动作类问题会多调一次 LLM 写总结，纯检索题直接返回片段；
> 所以 P95 从“每题都总结”的 15.9s 降到 10.1s（平均 6.3s）。

修复前后的对比、3 个 Bad Case 的归因（r03 该拒答却硬答、s07/m01 Top-1 排序错误、
s01 语料自相矛盾）、接入真实模型时修掉的 5 个问题，见 [evals/REPORT.md](evals/REPORT.md)。
CI 里会执行同一套用例，工具准确率低于 0.8 直接失败。

---

## 十一、常见问题

**Q：没有起 PostgreSQL 时 Agent 还能用吗？**
A：能。检索、多轮、拒答不依赖数据库；`create_task` 会返回「待办库不可用」，
会话日志写入失败也只记一条 warning。`pytest` 里建待办那条用例会自动 skip。

**Q：没有 API Key 时界面显示「离线模拟模式」，回答是拼凑的？**
A：是的。离线模拟模型只从检索结果中抽取最相关的句子，用来验证链路。
配置 `DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY` 后即由真实大模型生成。

**Q：想让召回更准？**
A：① 配置 `OPENAI_API_KEY` 并使用 `EMBED_PROVIDER=openai`（语义向量远好于本地哈希）；
② 调大 `TOP_K` / `FETCH_K`；③ 用 `patient` 参数限定患者；④ 调整 `HYBRID_ALPHA`。

**Q：换了 embedding 模型后结果很差？**
A：向量空间不一致，执行 `python scripts/ingest.py` 重建索引。

**Q：PDF 提取不出文字？**
A：扫描版 PDF 需要在 `loader.py` 里接 OCR（如 PaddleOCR / Tesseract），当前只处理文本层 PDF。

**Q：chromadb 安装很慢？**
A：chromadb 依赖 onnxruntime 等较大的包，建议使用国内镜像源。

**Q：端口被占用？**
A：`python run.py --port 8080`。

---

## 十二、可扩展方向

- 接真实 embedding / reranker（`EMBED_PROVIDER=openai` 或本地 bge 模型），替换当前的规则 rerank
- 去重：按「患者 + 章节 + 内容哈希」合并重复档案，减少 Top-1 抖动
- 把与健康档案无关的 `llama2.pdf` 移出 `input/`（占 27/58 chunk），重建索引后重跑评测
- 会话记忆升级：`PostgresSaver` 替代进程内 `MemorySaver`，长期偏好写 `user_prefs`
- 文档增量更新（按文件哈希判断是否需要重新向量化）
- 扫描件 OCR、结构化表格入库
- 权限控制：不同用户只能检索自己的健康档案

---

## 十三、参考

- LangChain：<https://python.langchain.com/>
- FastAPI：<https://fastapi.tiangolo.com/>
- Chroma：<https://docs.trychroma.com/>
- DeepSeek API：<https://platform.deepseek.com/api-docs/>
