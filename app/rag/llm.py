"""大模型接入层。

同一套接口同时支持：
- OpenAI（gpt-4o-mini / gpt-4o ...）
- DeepSeek（deepseek-chat / deepseek-reasoner）
- 任意 OpenAI 兼容服务（vLLM、Ollama、通义千问、Moonshot ...）
- 离线模拟模型（没有 API Key 时也能跑通全流程，方便演示与自测）

真实模型统一走 ``langchain_openai.ChatOpenAI``，只是 base_url / api_key / model 不同。
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, AsyncIterator, Iterator

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from app.config import Settings
from app.rag.providers import resolve_provider

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[a-zA-Z0-9]+")


# ---------------------------------------------------------------- 离线模拟模型
class MockChatModel(BaseChatModel):
    """离线演示用模型：从检索到的资料里抽句子拼答案。

    它不会「编造」，只是把与问题最相关的句子摘出来，因此可以在完全没有 API Key
    的环境里验证「检索 -> 拼 prompt -> 生成」这条链路是否通畅。
    """

    system_hint: str = ""

    @property
    def _llm_type(self) -> str:
        return "mock-extractive"

    # ---- 工具函数 ----
    @staticmethod
    def _tokens(text: str) -> set[str]:
        lower = text.lower()
        chars = _CJK_RE.findall(lower)
        grams = {chars[i] + chars[i + 1] for i in range(len(chars) - 1)} if len(chars) > 1 else set(chars)
        grams |= set(chars)
        grams |= set(_LATIN_RE.findall(lower))
        return {g for g in grams if g.strip()}

    @staticmethod
    def _parse(messages: list[BaseMessage]) -> tuple[str, str]:
        """从消息里拆出「参考资料」和「用户问题」。"""
        question, context = "", ""
        for message in messages:
            content = message.content if isinstance(message.content, str) else str(message.content)
            if "【参考资料】" in content:
                body = content.split("【参考资料】", 1)[1]
                if "【用户问题】" in body:
                    context, question = body.split("【用户问题】", 1)
                else:
                    context = body
                question = question.replace("请基于上述参考资料回答。", "").strip()
            elif message.type == "human" and not question:
                question = content.strip()
        return question, context.strip()

    def _compose(self, messages: list[BaseMessage]) -> str:
        question, context = self._parse(messages)
        if not context or context.startswith("（未检索到"):
            return (
                "我在健康档案中**没有检索到**与该问题相关的内容。\n\n"
                "可以尝试：\n"
                "1. 换一种问法（例如提到患者姓名、指标名称）；\n"
                "2. 确认相关文档已经灌入知识库（`python scripts/ingest.py`）；\n"
                "3. 在「知识库」页面确认文档列表。"
            )

        q_tokens = self._tokens(question)
        # 拆成「[n] ...」块，逐块找最相关的句子
        blocks = re.split(r"\n(?=\[\d+\])", context)
        scored: list[tuple[float, int, str]] = []
        for block in blocks:
            header, _, body = block.partition("\n")
            cite = re.match(r"\[(\d+)\]", header)
            cite_no = int(cite.group(1)) if cite else 0
            sentences = re.split(r"(?<=[。！？；])|\n+", body)
            for sentence in sentences:
                sentence = sentence.strip(" -\t")
                if len(sentence) < 8:
                    continue
                # 跳过 chunk 自带的「【文档 · 患者 · 章节】」标题行
                if sentence.startswith("【") and sentence.endswith("】"):
                    continue
                overlap = len(q_tokens & self._tokens(sentence))
                if overlap == 0:
                    continue
                score = overlap / (len(sentence) ** 0.5)
                scored.append((score, cite_no, sentence))

        if not scored:
            # 没有完全匹配的句子，退化为展示第一块内容
            first = blocks[0]
            header, _, body = first.partition("\n")
            cite_no = 1
            snippet = body.strip()[:220]
            scored = [(1.0, cite_no, snippet)]

        scored.sort(key=lambda x: (-x[0], x[1]))
        picked = scored[:4]
        picked.sort(key=lambda x: x[1])

        lines = [
            "根据健康档案检索到的资料：",
            "",
        ]
        for _, cite_no, sentence in picked:
            lines.append(f"- {sentence} [{cite_no}]")

        used = sorted({c for _, c, _ in picked})
        lines += [
            "",
            f"（以上内容摘自检索结果 {'、'.join(f'[{c}]' for c in used)}，"
            "由离线模拟模型抽取，仅供参考；如需完整分析请配置真实大模型。）",
        ]
        return "\n".join(lines)

    # ---- BaseChatModel 接口 ----
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self._compose(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        text = self._compose(messages)
        for piece in _chunks(text, 24):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if run_manager:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        text = self._compose(messages)
        for piece in _chunks(text, 24):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if run_manager:
                await run_manager.on_llm_new_token(piece, chunk=chunk)
            await asyncio.sleep(0.01)
            yield chunk


def _chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] or [""]


# ---------------------------------------------------------------- 真实模型
def build_chat_model(
    settings: Settings,
    provider_name: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    streaming: bool = False,
) -> tuple[BaseChatModel, dict]:
    """构造生成模型，返回 ``(model, meta)``。

    meta.mode 为 ``llm`` 表示真实模型，``mock`` 表示离线模拟。
    """
    provider = resolve_provider(settings, provider_name)
    temp = settings.llm_temperature if temperature is None else float(temperature)

    if provider.key == "mock":
        return MockChatModel(), {
            "provider": "mock",
            "provider_label": "离线模拟模型",
            "model": "offline-extractive",
            "mode": "mock",
            "note": "当前使用离线模拟模型，回答由检索结果抽取拼接而成",
        }

    chosen_model = (model or settings.default_model or provider.chat_model).strip()

    meta = {
        "provider": provider.key,
        "provider_label": provider.label,
        "model": chosen_model,
        "mode": "llm",
        "base_url": provider.base_url,
    }

    if not provider.available:
        if settings.allow_mock_llm:
            meta.update(mode="mock", model="offline-extractive", note=f"未配置 {provider.label} API Key，使用离线模拟模型")
            return MockChatModel(), meta
        raise RuntimeError(
            f"{provider.label} 未配置 API Key（{provider.key.upper()}_API_KEY），且未开启离线模拟模式"
        )

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=chosen_model,
        api_key=provider.api_key,
        base_url=provider.base_url,
        temperature=temp,
        max_tokens=settings.llm_max_tokens,
        timeout=90,
        max_retries=2,
        streaming=streaming,
    )
    return llm, meta


def list_provider_status(settings: Settings) -> dict:
    """给 /api/providers 用的状态汇总。"""
    from app.rag.providers import build_providers

    providers = [p.describe() for p in build_providers(settings).values()]
    real = [p for p in providers if p["key"] != "mock"]
    any_available = any(p["available"] for p in real)
    return {
        "providers": providers,
        "default_provider": settings.default_provider,
        "default_model": settings.default_model,
        "allow_mock_llm": settings.allow_mock_llm,
        "mock_active": not any_available and settings.allow_mock_llm,
        "mock_provider": {
            "key": "mock",
            "label": "离线模拟模型",
            "available": settings.allow_mock_llm,
            "notes": "未配置任何 API Key 时启用：从检索结果抽取句子作答，用于验证链路。",
        },
    }
