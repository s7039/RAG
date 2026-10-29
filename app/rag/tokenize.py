"""中文友好的轻量分词。

向量化（本地哈希）与 BM25 关键词检索共用同一套分词规则，保证两条召回通道
看到的是同一批「词」：

- 中文：单字 + 相邻二元组（「空腹血糖」-> 空, 腹, 血, 糖, 空腹, 腹血, 血糖 ...）
  中文没有空格，二元组是效果/成本比较平衡的方案；
- 英文/数字：小写单词 + 相邻双词组。
"""

from __future__ import annotations

import re

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[a-z0-9]+")

# 高频虚词单字，作为独立词贡献很低，但二元组仍保留
STOP_CHARS = set("的了是在和与及或就都而及其之为以于")


def tokenize(text: str) -> list[str]:
    lower = (text or "").lower()

    chars = _CJK_RE.findall(lower)
    tokens: list[str] = []
    for i, ch in enumerate(chars):
        if ch not in STOP_CHARS:
            tokens.append(ch)
        if i + 1 < len(chars):
            tokens.append(chars[i] + chars[i + 1])

    words = _LATIN_RE.findall(lower)
    tokens.extend(words)
    tokens.extend(f"{a}_{b}" for a, b in zip(words, words[1:]))
    return tokens
