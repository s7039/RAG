"""多模型供应商注册表。

OpenAI、DeepSeek 以及任何 OpenAI 兼容网关都通过同一套 ``ChatOpenAI``
客户端访问，只是 base_url / api_key / model 不同。这里把这些差异集中管理，
上层（检索、生成、接口）只需要拿到一个 ``ModelProvider``。

- 生成（Chat）：三者都支持。
- 向量化（Embedding）：OpenAI 支持；DeepSeek 官方暂未提供 embedding 接口，
  因此向量化统一交给 ``embeddings.py`` 的供应商解析逻辑（可回落到本地向量）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings


@dataclass(frozen=True)
class ModelProvider:
    """一个 OpenAI 兼容的大模型服务。"""

    key: str
    label: str
    base_url: str
    api_key: str
    chat_model: str
    # 可选模型列表（前端下拉框展示）
    chat_models: tuple[str, ...] = ()
    # 是否支持 embeddings 接口；None 表示模型名，空串表示不支持
    embedding_model: str = ""
    notes: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.base_url)

    @property
    def supports_embedding(self) -> bool:
        return bool(self.embedding_model)

    def describe(self) -> dict:
        """给前端 / 接口用的描述信息（不泄露 key）。"""
        return {
            "key": self.key,
            "label": self.label,
            "base_url": self.base_url,
            "chat_model": self.chat_model,
            "chat_models": list(self.chat_models) or [self.chat_model],
            "embedding_model": self.embedding_model,
            "supports_embedding": self.supports_embedding,
            "available": self.available,
            "notes": self.notes,
        }


def build_providers(settings: Settings) -> dict[str, ModelProvider]:
    """根据配置构造所有可用供应商。"""
    providers: dict[str, ModelProvider] = {
        "deepseek": ModelProvider(
            key="deepseek",
            label="DeepSeek",
            base_url=settings.deepseek_base_url,
            api_key=settings.deepseek_api_key,
            chat_model=settings.deepseek_chat_model,
            chat_models=(
                settings.deepseek_chat_model,
                "deepseek-chat",
                "deepseek-reasoner",
            ),
            embedding_model="",
            notes="DeepSeek 兼容 OpenAI 协议；官方暂不提供 embedding 接口，向量化使用独立配置。",
        ),
        "openai": ModelProvider(
            key="openai",
            label="OpenAI",
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            chat_model=settings.openai_chat_model,
            chat_models=(
                settings.openai_chat_model,
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-4.1-mini",
            ),
            embedding_model=settings.openai_embedding_model,
            notes="同时提供生成与向量化能力。",
        ),
    }

    if settings.custom_base_url and settings.custom_chat_model:
        providers["custom"] = ModelProvider(
            key="custom",
            label=settings.custom_label,
            base_url=settings.custom_base_url,
            api_key=settings.custom_api_key or "not-needed",
            chat_model=settings.custom_chat_model,
            chat_models=(settings.custom_chat_model,),
            embedding_model=settings.custom_embedding_model,
            notes="任意 OpenAI 兼容服务（vLLM / Ollama / 通义千问 / Moonshot 等）。",
        )

    # 离线模拟模型：没有 API Key 时也能跑通整条链路
    providers["mock"] = ModelProvider(
        key="mock",
        label="离线模拟模型",
        base_url="",
        api_key="mock" if settings.allow_mock_llm else "",
        chat_model="offline-extractive",
        chat_models=("offline-extractive",),
        embedding_model="",
        notes="从检索结果中抽取句子拼成回答，不调用网络，仅用于演示与自测。",
    )

    return providers


def resolve_provider(settings: Settings, name: str | None = None) -> ModelProvider:
    """把用户/配置里的供应商名称解析成 ``ModelProvider``。"""
    providers = build_providers(settings)
    key = (name or settings.default_provider or "deepseek").strip().lower()
    if key not in providers:
        raise KeyError(
            f"未知的模型供应商 '{name}'，可选：{', '.join(providers)}"
        )
    return providers[key]
