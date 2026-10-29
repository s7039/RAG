"""接口的请求 / 响应模型（Pydantic）。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: int | None = Field(None, ge=1, le=20, description="召回数量，默认取配置")
    provider: str | None = Field(None, description="deepseek / openai / custom，默认取配置")
    model: str | None = Field(None, description="覆盖默认模型名")
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    use_mmr: bool | None = Field(None, description="是否启用 MMR 重排")
    patient: str | None = Field(None, description="只检索某位患者的记录")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "张三九最近一次体检的空腹血糖是多少？有什么建议？",
                    "top_k": 4,
                    "provider": "deepseek",
                }
            ]
        }
    }


class SourceItem(BaseModel):
    rank: int
    score: float
    content: str
    source: str
    page: int | None = None
    page_end: int | None = None
    section: str = ""
    patient: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    question: str
    answer: str
    mode: str = Field(..., description="llm=真实大模型；mock=离线模拟模型")
    provider: dict[str, Any]
    sources: list[SourceItem]
    timings_ms: dict[str, float]
    context_chars: int = 0
    warnings: list[str] = Field(default_factory=list)


class IngestRequest(BaseModel):
    reset: bool = Field(True, description="是否先清空原有向量库")
    paths: list[str] | None = Field(None, description="只灌入指定文件（相对 input 目录或绝对路径）")
    input_dir: str | None = Field(None, description="自定义文档目录")


class IngestResponse(BaseModel):
    success: bool
    message: str
    stats: dict[str, Any]


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=50)
    use_mmr: bool | None = None
    patient: str | None = None


class RetrieveResponse(BaseModel):
    query: str
    strategy: str
    elapsed_ms: float
    count: int
    chunks: list[SourceItem]
