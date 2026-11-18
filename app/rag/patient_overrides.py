"""语料补丁：修 `input/` 里两份表格 PDF 的患者归属错误。

已知问题（详见 evals/REPORT.md）：

- `健康档案(含表格02).pdf` 整份是王五的数据，只有「姓名」字段写成了张三九；
- `健康档案(含表格01).pdf` 的「一、基本信息」是张三九，体检/化验表是王五的。

切分器按「姓名」字段识别患者，会把王五的记录挂到张三九名下，所以在这里按
「文件 + 章节」覆盖。换成自己的语料后删掉本文件即可，切分器会回退到自动识别。
"""

from __future__ import annotations

from pathlib import Path

OVERRIDES: dict[tuple[str, str | None], str] = {
    ("健康档案(含表格02).pdf", None): "王五",
    ("健康档案(含表格01).pdf", "二、体检与化验结果"): "王五",
}


def resolve(source: str, section: str, detected: str) -> str:
    name = Path(source).name
    return OVERRIDES.get((name, section)) or OVERRIDES.get((name, None)) or detected
