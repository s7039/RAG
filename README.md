# 健康档案私有知识库 RAG

基于 **LangChain + FastAPI + OpenAI / DeepSeek + Chroma** 的私有领域知识库（RAG）示例项目。
以 `input/` 目录下的健康档案 PDF 为语料，完整跑通「离线建库 + 在线问答」全流程，
并支持同时接入多种大模型、流式输出、引用溯源与患者级过滤。

> 健康档案存放在 `input/` 目录（原始需求中的 "imput" 为笔误，项目统一使用 `input`）。

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

### 相对「教科书版 RAG」多做的一点事

| 能力 | 说明 |
| --- | --- |
| 表格还原 | PDF 里的表格被识别并转成 Markdown，指标名和数值不会被拆散 |
| 结构切分 | 按章节切，自动带上「文档 · 患者 · 章节」上下文头，避免张三九的资料召回给李四六 |
| 患者识别 | 从「姓名 / 患者姓名为 / XXX，男」等写法里抽取患者名，写入元数据并支持过滤 |
| 混合检索 | 向量 + BM25 双路召回加权融合，离线无 Key 时召回质量也稳定 |
| 引用溯源 | 回答里的 `[1]` 对应前端可展开的原文片段，含文件名、页码、章节 |
| 多模型 | 同一份代码切换 DeepSeek / OpenAI / 任意 OpenAI 兼容服务 |
| 流式输出 | SSE 逐字返回，前端边生成边渲染 |
| 零 Key 可跑 | 没有 API Key 时自动使用「本地哈希向量 + 离线模拟模型」，全链路可演示 |

---

## 二、目录结构

```
RAG/
├─ app/
│  ├─ main.py                 # FastAPI 入口（lifespan 初始化服务、挂载静态页）
│  ├─ config.py               # pydantic-settings 配置（读 .env / 环境变量）
│  ├─ api/
│  │  ├─ routes.py            # 所有 HTTP 接口
│  │  └─ schemas.py           # 请求 / 响应模型
│  ├─ rag/
│  │  ├─ loader.py            # ① 文档加载（PDF/DOCX/TXT/MD/CSV）
│  │  ├─ splitter.py          # ② 文档切分（结构 + 表格 + 患者）
│  │  ├─ embeddings.py        # ③ 向量化（OpenAI / 本地哈希）
│  │  ├─ vectorstore.py       # ④ 向量数据库（Chroma）
│  │  ├─ bm25.py              # 关键词检索
│  │  ├─ tokenize.py          # 中文友好分词（中英混合）
│  │  ├─ retriever.py         # 混合检索 + MMR 重排
│  │  ├─ prompt.py            # Prompt 模板与上下文拼装
│  │  ├─ llm.py               # 多模型接入 + 离线模拟模型
│  │  ├─ providers.py         # 模型供应商注册表
│  │  └─ pipeline.py          # 离线 / 在线流程编排（RagService）
│  └─ web/index.html          # 内置聊天界面（无 CDN 依赖）
├─ scripts/
│  ├─ ingest.py               # 命令行灌库
│  └─ query.py                # 命令行问答
├─ tests/                     # pytest 测试（21 个用例）
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

---

## 六、检索策略

```
score = HYBRID_ALPHA × 向量相似度 + (1 - HYBRID_ALPHA) × BM25 分数
```

- `RETRIEVE_MODE=hybrid`（默认）：向量 + BM25 双路召回融合；
- `RETRIEVE_MODE=vector`：纯向量检索；
- `USE_MMR=true`：开启 MMR（最大边际相关性）重排，减少返回内容重复；
- `patient=张三九`：按患者过滤，只在该患者的记录里检索。

实测（本地向量，无 API Key）：

| 问题 | Top-1 命中 |
| --- | --- |
| 张三九最近一次体检的空腹血糖是多少？ | 健康档案.pdf 第1页 · 四、体检与化验结果 ✅ |
| 李四六对哪些药物过敏？ | 健康档案.pdf 第2页 · 李四六 医疗历史 ✅ |
| 王五做过什么手术？ | 健康档案.pdf 第4页 · 王五 医疗历史 ✅ |
| 医生给张三九的饮食建议是什么 | 健康档案.pdf 第2页 · 六、医生建议 ✅ |

---

## 七、配置项速查

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

完整列表见 `app/config.py` 与 `.env.example`。

---

## 八、测试

```bash
python -m pytest -v
```

覆盖：切分与患者识别、表格保护、页码回溯、BM25 排序与过滤、端到端入库检索、
HTTP 接口（含 SSE 流式）、空库提问等。全部用例使用本地向量 + 离线模拟模型，**不需要网络和 API Key**。

---

## 九、常见问题

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

## 十、可扩展方向

- 接入 Rerank 模型（如 bge-reranker）做二次精排
- 增加会话记忆，支持多轮追问
- 文档增量更新（按文件哈希判断是否需要重新向量化）
- 扫描件 OCR、结构化表格入库
- 接入 LangGraph / Agent，做多步检索与工具调用
- 权限控制：不同用户只能检索自己的健康档案

---

## 十一、参考

- LangChain：<https://python.langchain.com/>
- FastAPI：<https://fastapi.tiangolo.com/>
- Chroma：<https://docs.trychroma.com/>
- DeepSeek API：<https://platform.deepseek.com/api-docs/>
