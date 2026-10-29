"""Prompt 模板层。

在线流程的最后一段：把「检索结果」和「用户问题」填进模板。
模板针对健康档案场景做了约束：只依据资料回答、必须标注引用、资料不足要承认、
不能替代医生做诊断。
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from app.rag.retriever import RetrievedChunk

SYSTEM_PROMPT = """你是「健康档案私有知识库」的智能健康助理，服务对象是档案的本人及其家属。
你必须严格依据下面提供的【参考资料】回答问题。

回答规则：
1. 只使用【参考资料】中的信息，不要编造任何数值、日期、药物或结论。
2. 每个关键结论后标注来源编号，例如 [1]；同一结论可标注多个来源，例如 [1][2]。
3. 如果资料中没有足够信息，请直接说明「健康档案中未记录相关信息」，并建议补充资料或咨询医生。
4. 涉及指标异常时，可结合资料中的参考范围做客观说明，但不要下诊断结论；
   必要时提醒「请以临床医生意见为准」。
5. 用简体中文回答，条理清晰；有多项数据时优先用列表或表格呈现。
6. 回答要简洁，先给结论，再给依据。"""

HUMAN_PROMPT = """【参考资料】
{context}

【用户问题】
{question}

请基于上述参考资料回答。"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ]
)


def build_context(chunks: list[RetrievedChunk], max_chars: int = 6000) -> str:
    """按字符预算拼装上下文，超长时截断最后一块而不是整段丢弃。"""
    if not chunks:
        return "（未检索到相关资料）"

    blocks: list[str] = []
    used = 0
    for i, chunk in enumerate(chunks, start=1):
        head = f"[{i}] 来源：{chunk.source}"
        if chunk.page:
            head += f" 第{chunk.page}页"
        if chunk.patient:
            head += f" | 患者：{chunk.patient}"
        if chunk.section:
            head += f" | 章节：{chunk.section}"
        body = chunk.content.strip()
        block = f"{head}\n{body}"

        remaining = max_chars - used
        if remaining <= 120:
            break
        if len(block) > remaining:
            block = block[: remaining - 20].rstrip() + "\n…（内容已截断）"
            blocks.append(block)
            break
        blocks.append(block)
        used += len(block) + 2

    return "\n\n".join(blocks)


def build_messages(question: str, chunks: list[RetrievedChunk], max_context_chars: int = 6000):
    """返回可直接喂给 LLM 的消息列表。"""
    return RAG_PROMPT.format_messages(
        context=build_context(chunks, max_context_chars),
        question=question.strip(),
    )
