"""文档加载层。

把 ``input/`` 下的文件读成带元数据的 :class:`langchain_core.documents.Document`。
- PDF：优先用 pdfplumber 还原表格（转成 Markdown 表格），失败时回落到 pypdf。
- DOCX：python-docx，表格同样转 Markdown。
- TXT / MD / CSV：直接读取，CSV 转 Markdown 表格。

每页/每个文件产出若干 Document，元数据包含 source / page / doc_title / doc_type。
"""

from __future__ import annotations

import csv
import io
import logging
import re
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".markdown", ".csv"}


class UnsupportedFileType(Exception):
    pass


# ---------------------------------------------------------------- helpers
def _clean_text(text: str) -> str:
    """去掉行尾空白与多余空行。"""
    if not text:
        return ""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned).strip()


def _rows_to_markdown(rows: list[list[str]]) -> str:
    """把二维表格转成 Markdown。"""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm: list[list[str]] = []
    for row in rows:
        cells = [(c or "").replace("\n", " ").strip() for c in row]
        cells += [""] * (width - len(cells))
        norm.append(cells)
    header, *body = norm
    if not any(header):
        header = [f"列{i + 1}" for i in range(width)]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def _table_to_markdown(table) -> str:
    try:
        rows = table.extract()
    except Exception:  # pragma: no cover - 个别坏表格
        return ""
    rows = [[c for c in row] for row in rows if row]
    # 去掉全空列
    if rows:
        keep = [i for i in range(max(len(r) for r in rows)) if any((r[i] if i < len(r) else "") for r in rows)]
        rows = [[(r[i] if i < len(r) else "") for i in keep] for r in rows]
    return _rows_to_markdown(rows)


# ---------------------------------------------------------------- PDF
# pdfplumber 的 x_tolerance 决定「多大间距算空格」：
# - 中文排版字间距本来就宽，用默认 3.0 才不会把「居住在上海市」拆成一字一空格；
# - 英文论文单词间距窄，需要 1.5，否则 HugoTouvron 会粘在一起。
# 因此按页面字符构成自动选择。
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def _page_x_tolerance(page) -> float:
    chars = [c for c in (page.chars or []) if (c.get("text") or "").strip()]
    if not chars:
        return 1.5
    cjk = sum(1 for c in chars if _CJK_RE.match(c.get("text") or ""))
    return 3.0 if cjk / len(chars) > 0.15 else 1.5


def _extract(page, bbox=None, x_tolerance: float = 1.5) -> str:
    target = page.crop(bbox) if bbox is not None else page
    try:
        return target.extract_text(x_tolerance=x_tolerance, y_tolerance=3) or ""
    except TypeError:  # pragma: no cover - 兼容旧版 pdfplumber
        return target.extract_text() or ""


def _pdf_pages_with_pdfplumber(path: Path) -> list[tuple[int, str]]:
    import pdfplumber

    pages: list[tuple[int, str]] = []
    with pdfplumber.open(str(path)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            tolerance = _page_x_tolerance(page)
            pieces: list[str] = []
            try:
                tables = page.find_tables()
            except Exception:  # pragma: no cover
                tables = []

            boxes = sorted((t.bbox for t in tables), key=lambda b: b[1])
            cursor = page.bbox[1]
            for box in boxes:
                if box[1] > cursor + 1:
                    part = _extract(page, (page.bbox[0], cursor, page.bbox[2], box[1]), tolerance)
                    if part.strip():
                        pieces.append(part)
                # 找到与当前 box 匹配的 table 对象
                table_obj = next((t for t in tables if t.bbox == box), None)
                if table_obj is not None:
                    md = _table_to_markdown(table_obj)
                    if md:
                        pieces.append(md)
                cursor = max(cursor, box[3])
            if cursor < page.bbox[3]:
                part = _extract(page, (page.bbox[0], cursor, page.bbox[2], page.bbox[3]), tolerance)
                if part.strip():
                    pieces.append(part)

            text = _clean_text("\n\n".join(pieces))
            if text:
                pages.append((page_no, text))
    return pages


def _pdf_pages_with_pypdf(path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text(extraction_mode="layout") or ""
        except TypeError:  # 旧版本 pypdf
            text = page.extract_text() or ""
        text = _clean_text(text)
        if text:
            pages.append((page_no, text))
    return pages


def load_pdf(path: Path) -> list[tuple[int, str]]:
    try:
        pages = _pdf_pages_with_pdfplumber(path)
        if pages:
            return pages
        logger.warning("pdfplumber 未提取到文本，回落到 pypdf: %s", path.name)
    except Exception as exc:  # pragma: no cover
        logger.warning("pdfplumber 解析失败(%s)，回落到 pypdf: %s", exc, path.name)
    return _pdf_pages_with_pypdf(path)


# ---------------------------------------------------------------- 其它格式
def load_docx(path: Path) -> list[tuple[int, str]]:
    import docx

    document = docx.Document(str(path))
    blocks: list[str] = []
    body = document.element.body
    table_iter = iter(document.tables)
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            text = "".join(node.text or "" for node in child.iter() if node.tag.endswith("}t"))
            if text.strip():
                blocks.append(text)
        elif tag == "tbl":
            table = next(table_iter, None)
            if table is not None:
                rows = [[cell.text for cell in row.cells] for row in table.rows]
                md = _rows_to_markdown(rows)
                if md:
                    blocks.append(md)
    return [(1, _clean_text("\n\n".join(blocks)))]


def load_text(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(1, _clean_text(text))]


def load_csv(path: Path) -> list[tuple[int, str]]:
    raw = path.read_text(encoding="utf-8-sig", errors="ignore")
    sample = raw[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = [row for row in csv.reader(io.StringIO(raw), dialect=dialect) if any(c.strip() for c in row)]
    md = _rows_to_markdown(rows)
    return [(1, _clean_text(md))]


# ---------------------------------------------------------------- 入口
def load_file(path: Path, base_dir: Path | None = None) -> list[Document]:
    """加载单个文件，返回按页拆分的 Document 列表。"""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileType(f"不支持的文件类型：{suffix}（{path.name}）")

    if suffix == ".pdf":
        pages = load_pdf(path)
    elif suffix == ".docx":
        pages = load_docx(path)
    elif suffix == ".csv":
        pages = load_csv(path)
    else:
        pages = load_text(path)

    try:
        source = str(path.relative_to(base_dir)) if base_dir else path.name
    except ValueError:
        source = path.name

    docs: list[Document] = []
    for page_no, text in pages:
        if not text.strip():
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": source.replace("\\", "/"),
                    "doc_title": path.stem,
                    "doc_type": suffix.lstrip("."),
                    "page": page_no,
                    "file_path": str(path),
                },
            )
        )
    return docs


def discover_files(input_dir: Path) -> list[Path]:
    """递归发现输入目录下的受支持文件（忽略隐藏文件）。"""
    if not input_dir.exists():
        return []
    files = [
        p
        for p in sorted(input_dir.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        and not p.name.startswith(".")
    ]
    return files


def load_documents(input_dir: Path, paths: list[Path] | None = None) -> list[Document]:
    """加载目录或指定文件列表，返回所有页级 Document。"""
    targets = paths if paths else discover_files(input_dir)
    docs: list[Document] = []
    for path in targets:
        try:
            docs.extend(load_file(path, base_dir=input_dir))
        except UnsupportedFileType as exc:
            logger.warning("%s", exc)
        except Exception as exc:  # pragma: no cover
            logger.exception("加载失败 %s: %s", path, exc)
    return docs
