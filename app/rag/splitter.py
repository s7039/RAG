"""文档切分层。

针对健康档案这类「多患者 + 多章节 + 含表格」的中文文档，做了结构化切分：

1. 把同一文件的各页拼成一条文本流，并保留页码标记，切分后仍能回溯页码。
2. 按「一、基本信息 / 二、医疗历史 …」这类标题切出章节；标题形如
   ``1. 既往病史`` 的二级标题用于判断段落归属，不强制切开。
3. 章节内按「段落 / 表格」为最小块打包，保证 Markdown 表格不会被从中间截断。
4. 自动识别患者姓名，写入元数据并拼进 chunk 头部，
   这样「李四六的胆固醇」不会召回张三九的记录。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------- 常量
PAGE_SENTINEL_RE = re.compile(r"\x00P(\d+)\x00")
PAGE_SENTINEL = "\x00P{page}\x00"

# 一级标题：一、基本信息 / 二、医疗历史 / # 标题 / 第X章
H1_RE = re.compile(
    r"^\s*(?:"
    r"[一二三四五六七八九十百]+、"
    r"|第[一二三四五六七八九十百]+[章节篇部分]"
    r"|#{1,3}\s+\S"
    r")"
)
# 二级标题：1. 体检结果 / 2、实验室检查 / （1）xxx
H2_RE = re.compile(r"^\s*(?:\d{1,2}[.、)]|（\d{1,2}）)\s*\S")

RECORD_TITLE_RE = re.compile(r"^\s*(?:健康档案|个人健康档案|居民健康档案)\s*$")

NAME_PATTERNS = [
    # 患者姓名：张三九 / 姓名为张三九 / | 姓名 | 张三九 |
    re.compile(r"(?:患者)?姓\s*名\s*(?:[为是：:]?\s*)[\|｜]?\s*([\u4e00-\u9fa5]{2,4})"),
    # 张三九，男 / 李四六, 女性
    re.compile(r"([\u4e00-\u9fa5]{2,4})[，,]\s*(?:男|女)(?:性)?[，,]?"),
    # 患者为张三九，/ 患者：张三九（
    re.compile(r"患者[为是：:]?\s*([\u4e00-\u9fa5]{2,4})[，,（(]"),
]

# 姓名里几乎不会出现的虚词，用于过滤误匹配
NAME_BAD_CHARS = set("为是的了在与和及或对之其于把被让给也就都还又才")

# 避免把「性别」「年龄」等表头误认成姓名
NAME_STOPWORDS = {
    "姓名", "性别", "年龄", "婚姻", "职业", "血型", "电话", "邮箱", "居住",
    "紧急", "联系", "详细", "项目", "信息", "疾病", "治疗", "结果", "年份",
    "手术", "过敏", "家族", "成员", "情况", "基本", "健康", "档案", "患者",
    "男性", "女性", "已婚", "未婚", "检查", "评估", "参考", "范围", "医院",
}


@dataclass
class Section:
    title: str
    start: int
    end: int
    level: int = 1
    record_index: int = 0


@dataclass
class ChunkRecord:
    text: str
    start: int
    end: int
    section: str
    record_index: int
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- 页码
class PageMap:
    """把字符偏移映射回 PDF 页码。"""

    def __init__(self, marks: list[tuple[int, int]]) -> None:
        self.marks = sorted(marks)

    def page_at(self, offset: int) -> int:
        page = self.marks[0][1] if self.marks else 1
        for pos, page_no in self.marks:
            if pos <= offset:
                page = page_no
            else:
                break
        return page


def _merge_pages(docs: list[Document]) -> tuple[str, PageMap]:
    """把同一文件的页级 Document 拼成一条文本流。"""
    parts: list[str] = []
    marks: list[tuple[int, int]] = []
    cursor = 0
    for doc in docs:
        page = int(doc.metadata.get("page", 1))
        sentinel = PAGE_SENTINEL.format(page=page)
        parts.append(sentinel)
        marks.append((cursor, page))
        cursor += len(sentinel)
        body = doc.page_content.strip("\n")
        parts.append(body)
        cursor += len(body)
        parts.append("\n\n")
        cursor += 2
    return "".join(parts), PageMap(marks)


def _iter_lines(text: str):
    """逐行返回 (行起始偏移, 去换行的行文本)。"""
    offset = 0
    for raw in text.split("\n"):
        yield offset, raw
        offset += len(raw) + 1


# ---------------------------------------------------------------- 结构切分
def _find_sections(text: str) -> list[Section]:
    """按一级标题切章节；同时标记包含「基本信息」的记录起点。"""
    boundaries: list[tuple[int, str]] = []
    for offset, line in _iter_lines(text):
        # 行首可能带着页码标记（\x00P2\x00），先去掉再判断标题
        stripped = PAGE_SENTINEL_RE.sub("", line).strip()
        if not stripped:
            continue
        if H1_RE.match(stripped) or RECORD_TITLE_RE.match(stripped):
            boundaries.append((offset, stripped))

    if not boundaries:
        return [Section(title="正文", start=0, end=len(text), record_index=0)]

    sections: list[Section] = []
    # 第一个标题之前的内容归入「文档开头」
    if boundaries[0][0] > 0 and text[: boundaries[0][0]].strip():
        sections.append(Section(title="文档开头", start=0, end=boundaries[0][0], record_index=0))

    record_index = 0
    for i, (start, title) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        if "基本信息" in title:
            record_index += 1
        sections.append(Section(title=title, start=start, end=end, record_index=record_index))
    # 记录编号从 0 开始更自然
    for sec in sections:
        sec.record_index = max(0, sec.record_index - 1)
    return sections


def _record_spans(text: str, sections: list[Section]) -> dict[int, str]:
    """按 record_index 聚合文本，用于抽取患者姓名。"""
    spans: dict[int, list[str]] = {}
    for sec in sections:
        spans.setdefault(sec.record_index, []).append(text[sec.start : sec.end])
    return {idx: "\n".join(parts) for idx, parts in spans.items()}


def _looks_like_name(candidate: str) -> bool:
    if not (2 <= len(candidate) <= 4):
        return False
    if candidate in NAME_STOPWORDS:
        return False
    return not (set(candidate) & NAME_BAD_CHARS)


def extract_patient_name(record_text: str) -> str:
    cleaned = PAGE_SENTINEL_RE.sub("", record_text)
    for pattern in NAME_PATTERNS:
        for match in pattern.finditer(cleaned):
            candidate = match.group(1).strip()
            if _looks_like_name(candidate):
                return candidate
    return ""


# ---------------------------------------------------------------- 块切分
def _split_into_blocks(text: str) -> list[tuple[int, int]]:
    """把章节文本切成「段落块 / 表格块」（用偏移区间表示）。"""
    blocks: list[tuple[int, int]] = []
    lines = list(_iter_lines(text))

    buf_start: int | None = None
    buf_end = 0

    def flush() -> None:
        nonlocal buf_start, buf_end
        if buf_start is not None and text[buf_start:buf_end].strip():
            blocks.append((buf_start, buf_end))
        buf_start = None

    i = 0
    while i < len(lines):
        offset, line = lines[i]
        stripped = line.strip()
        is_table = stripped.startswith("|") and stripped.count("|") >= 2

        if is_table:
            flush()
            table_start = offset
            table_end = offset + len(line)
            j = i + 1
            while j < len(lines):
                nxt_offset, nxt_line = lines[j]
                nxt_stripped = nxt_line.strip()
                if nxt_stripped.startswith("|") and nxt_stripped.count("|") >= 2:
                    table_end = nxt_offset + len(nxt_line)
                    j += 1
                elif not nxt_stripped:
                    break
                else:
                    break
            blocks.append((table_start, table_end))
            i = j
            continue

        if not stripped:
            flush()
            i += 1
            continue

        if buf_start is None:
            buf_start = offset
        buf_end = offset + len(line)
        i += 1

    flush()
    return blocks


def _overlap_tail(text: str, overlap: int) -> str:
    """取上一块结尾作为下一块的重叠上下文，避免把表格切成半截。"""
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    if tail.count("|") >= 2:  # 落在表格里，放弃重叠
        return ""
    # 从句子边界开始，读起来更自然
    for sep in ("。", "；", "\n", "，", " "):
        idx = tail.find(sep)
        if 0 <= idx < len(tail) - 1:
            return tail[idx + 1 :]
    return tail


def _looks_meaningful(text: str) -> bool:
    """判断短文本是不是有信息量的正文（而不是标题、页眉之类的碎片）。"""
    if any(mark in text for mark in "。！？；，"):
        return True
    if "|" in text:  # 表格片段
        return True
    return len(text) >= 15


def _pack_blocks(
    text: str,
    blocks: list[tuple[int, int]],
    section_start: int,
    section_end: int,
    chunk_size: int,
    overlap: int,
    min_size: int,
) -> list[ChunkRecord]:
    """把块贪心打包成不超过 chunk_size 的 chunk。

    每个 chunk 只由「整块」组成，因此表格不会被截断；
    相邻 chunk 之间用上一块的结尾做少量重叠，提升召回连续性。
    """
    records: list[ChunkRecord] = []
    if not blocks:
        body = PAGE_SENTINEL_RE.sub("", text[section_start:section_end]).strip()
        if body:
            records.append(ChunkRecord(body, section_start, section_end, "", 0))
        return records

    state = {"start": -1, "end": -1, "len": 0}
    previous_text = ""

    def flush() -> None:
        nonlocal previous_text
        start, end = state["start"], state["end"]
        if start < 0:
            return
        state["start"] = -1
        body = PAGE_SENTINEL_RE.sub("", text[start:end]).strip()
        if not body:
            return
        if len(body) < min_size:
            # 太短：能并进上一块就并；否则只在「像正文」时保留，
            # 纯标题行（如封面上的「健康档案」）直接丢弃
            if records:
                prev = records[-1]
                merged = (prev.text + "\n" + body).strip()
                if len(merged) <= chunk_size * 1.5:
                    prev.text = merged
                    prev.end = end
                    previous_text = merged
                    return
            if not _looks_meaningful(body):
                return
        prefix = _overlap_tail(previous_text, overlap) if previous_text else ""
        chunk_text = f"{prefix}\n{body}".strip() if prefix else body
        records.append(ChunkRecord(chunk_text, start, end, "", 0))
        previous_text = body

    for b_start, b_end in blocks:
        block_text = PAGE_SENTINEL_RE.sub("", text[b_start:b_end])
        block_len = len(block_text.strip())
        if state["start"] < 0:
            state.update(start=b_start, end=b_end, len=block_len)
            continue
        if state["len"] + block_len + 1 <= chunk_size:
            state["end"] = b_end
            state["len"] += block_len + 1
        else:
            flush()
            state.update(start=b_start, end=b_end, len=block_len)

    flush()

    # 超长块单独再做句子级切分
    final: list[ChunkRecord] = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "；", "！", "？", "，", "、", " "],
        keep_separator="end",
    )
    for rec in records:
        body = rec.text
        # 表格块即使超长也保留完整（健康档案里的表格都很短）
        if len(body) <= chunk_size or body.lstrip().startswith("|"):
            final.append(rec)
            continue
        for piece in splitter.split_text(body):
            piece = piece.strip()
            if len(piece) >= min_size:
                final.append(ChunkRecord(piece, rec.start, rec.end, "", rec.record_index))
    return final


# ---------------------------------------------------------------- 入口
def split_documents(
    docs: list[Document],
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    min_chunk_size: int = 30,
) -> list[Document]:
    """把页级 Document 切成语义 chunk，并补齐元数据。"""
    if not docs:
        return []

    # 按文件分组，同文件的多页合并后再切，保证跨页章节完整
    grouped: dict[str, list[Document]] = {}
    for doc in docs:
        source = doc.metadata.get("source", doc.metadata.get("doc_title", "unknown"))
        grouped.setdefault(source, []).append(doc)

    chunks: list[Document] = []
    for source, pages in grouped.items():
        pages = sorted(pages, key=lambda d: int(d.metadata.get("page", 1)))
        merged, page_map = _merge_pages(pages)
        sections = _find_sections(merged)
        record_names = {
            idx: extract_patient_name(body) for idx, body in _record_spans(merged, sections).items()
        }

        base_meta = dict(pages[0].metadata)
        global_index = 0

        for section in sections:
            section_text = merged[section.start : section.end]
            if not section_text.strip():
                continue
            title = PAGE_SENTINEL_RE.sub("", section.title).lstrip("#").strip()
            # 正文从标题行之后开始
            newline = merged.find("\n", section.start)
            body_start = newline + 1 if 0 <= newline < section.end else section.end
            if body_start >= section.end:
                body_start = section.start
            blocks = _split_into_blocks(merged[body_start:section.end])
            shifted = [(body_start + s, body_start + e) for s, e in blocks]

            records = _pack_blocks(
                merged,
                shifted,
                body_start,
                section.end,
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                min_size=min_chunk_size,
            )

            patient = record_names.get(section.record_index, "")
            for rec in records:
                body = rec.text.strip()
                if not body:
                    continue
                header_parts = [base_meta.get("doc_title", ""), patient, title]
                header = "【" + " · ".join(p for p in header_parts if p) + "】"
                content = f"{header}\n{body}"
                page_start = page_map.page_at(rec.start)
                page_end = page_map.page_at(max(rec.start, rec.end - 1))
                chunk_id = hashlib.sha1(
                    f"{source}|{page_start}|{global_index}|{content}".encode("utf-8")
                ).hexdigest()[:16]
                metadata = {
                    **base_meta,
                    "page": page_start,
                    "page_end": page_end,
                    "section": title,
                    "patient": patient,
                    "chunk_index": global_index,
                    "char_len": len(body),
                    "chunk_id": chunk_id,
                }
                chunks.append(Document(page_content=content, metadata=metadata))
                global_index += 1

    return chunks
