"""向量化（Embedding）层。

支持两种向量化方式，可通过 ``EMBED_PROVIDER`` 切换：

1. ``openai``  —— 调用 OpenAI（或任何兼容 /embeddings 接口的服务）的 embedding 模型。
2. ``local``   —— 纯本地哈希向量（hashing trick），无需联网、无需 API Key，
   中文按「单字 + 二元组」、英文按「单词 + 双词组」做特征哈希，再 L2 归一化。
   注意：它属于稀疏词法向量，效果不如语义向量，仅用于离线演示 / 无 Key 场景。

``auto`` 时会优先使用 OpenAI，没有 Key 就自动回落到本地向量。
"""

from __future__ import annotations

import hashlib
import math

from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.rag.tokenize import tokenize


class LocalHashingEmbeddings(Embeddings):
    """本地确定性向量：hashing trick + L2 归一化。

    相同文本永远得到相同向量，因此可以直接持久化到向量库；
    查询与文档使用同一套分词规则，保证可比较。
    """

    def __init__(self, dim: int = 1024) -> None:
        if dim <= 0:
            raise ValueError("向量维度必须为正整数")
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        counts: dict[str, int] = {}
        for token in tokenize(text or ""):
            counts[token] = counts.get(token, 0) + 1

        vector = [0.0] * self.dim
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "big")
            index = raw % self.dim
            sign = 1.0 if (raw >> 63) & 1 else -1.0
            # 次线性词频，削弱高频词的影响
            vector[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    # ---- LangChain Embeddings 接口 ----
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def _openai_embeddings(settings: Settings, model: str, base_url: str, api_key: str) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=model,
        api_key=api_key,
        base_url=base_url,
        chunk_size=max(1, settings.embed_batch_size),
        check_embedding_ctx_length=False,
    )


def build_embeddings(settings: Settings) -> tuple[Embeddings, dict]:
    """根据配置构造向量化器，并返回描述信息。

    返回 ``(embeddings, descriptor)``，descriptor 会写进索引元数据，
    用于检测「换了向量模型却没重建索引」的问题。
    """
    choice = (settings.embed_provider or "auto").strip().lower()

    def _local(reason: str) -> tuple[Embeddings, dict]:
        return LocalHashingEmbeddings(dim=settings.local_embed_dim), {
            "provider": "local",
            "model": f"hashing-{settings.local_embed_dim}",
            "dim": settings.local_embed_dim,
            "note": reason,
        }

    def _remote() -> tuple[Embeddings, dict]:
        # 优先 OpenAI；其次 custom；DeepSeek 没有 embedding 接口
        candidates = []
        if settings.openai_api_key:
            candidates.append(
                ("openai", settings.openai_base_url, settings.openai_api_key, settings.openai_embedding_model)
            )
        if settings.custom_base_url and settings.custom_embedding_model:
            candidates.append(
                (
                    "custom",
                    settings.custom_base_url,
                    settings.custom_api_key or "not-needed",
                    settings.custom_embedding_model,
                )
            )
        if not candidates:
            raise RuntimeError(
                "当前 EMBED_PROVIDER 需要向量模型 API Key，但没有配置 "
                "OPENAI_API_KEY / CUSTOM_EMBEDDING_MODEL；"
                "可改为 EMBED_PROVIDER=local 使用本地向量。"
            )
        provider, base_url, api_key, model = candidates[0]
        emb = _openai_embeddings(settings, model=model, base_url=base_url, api_key=api_key)
        return emb, {
            "provider": provider,
            "model": model,
            "dim": None,  # 首次向量化后才知道，见 VectorStore.finalize()
            "note": f"{provider} embedding 接口",
        }

    if choice == "local":
        return _local("强制使用本地哈希向量（EMBED_PROVIDER=local）")
    if choice in {"openai", "custom"}:
        return _remote()

    # auto
    try:
        return _remote()
    except RuntimeError:
        return _local("未检测到 embedding API Key，自动使用本地哈希向量")
