"""切分层单元测试：结构切分、患者识别、表格保护、页码回溯。"""

from __future__ import annotations

from langchain_core.documents import Document

from app.rag.splitter import extract_patient_name, split_documents

PAGE_1 = """健康档案
一、基本信息
张三九，男，出生于1985年5月20日，目前39岁，已婚，职业为IT工程师。
张三九的血型为O型，联系电话为138-1234-5678。

二、医疗历史
张三九在2005年被确诊为高血压，2010年被确诊为二型糖尿病。
张三九对青霉素有过敏反应，曾出现皮疹和呼吸困难。
"""

PAGE_2 = """四、体检与化验结果
张三九最近一次体检是在2024年8月15日，结果如下：
| 项目 | 结果 | 参考范围 | 评估 |
| --- | --- | --- | --- |
| 空腹血糖 | 6.8毫摩尔/升 | 4.0-6.1毫摩尔/升 | 略高 |
| 总胆固醇 | 5.2毫摩尔/升 | <5.2毫摩尔/升 | 正常 |

五、健康风险评估
由于血压偏高和血脂异常，张三九存在较高的心血管疾病风险。

一、基本信息
李四六，女，出生于1990年3月15日，职业为小学教师。
李四六对阿司匹林过敏，曾出现皮疹和轻度呼吸困难。
"""


def make_docs() -> list[Document]:
    meta = {"source": "健康档案.pdf", "doc_title": "健康档案", "doc_type": "pdf"}
    return [
        Document(page_content=PAGE_1, metadata={**meta, "page": 1}),
        Document(page_content=PAGE_2, metadata={**meta, "page": 2}),
    ]


def test_extract_patient_name_variants():
    assert extract_patient_name("张三九，男，出生于1985年5月20日") == "张三九"
    assert extract_patient_name("| 姓名 | 王五 |") == "王五"
    assert extract_patient_name("患者姓名为李四六，女性") == "李四六"
    assert extract_patient_name("这里没有任何姓名信息") == ""


def test_split_keeps_patient_and_section():
    chunks = split_documents(make_docs(), chunk_size=600, chunk_overlap=50)
    assert chunks, "应当切出 chunk"
    patients = {c.metadata["patient"] for c in chunks}
    assert "张三九" in patients
    assert "李四六" in patients

    # 每个 chunk 头部都带上了「文档 · 患者 · 章节」上下文
    for chunk in chunks:
        assert chunk.page_content.startswith("【健康档案 · ")
        assert chunk.page_content.count("\n") >= 1

    sections = {c.metadata["section"] for c in chunks}
    assert "一、基本信息" in sections
    assert "四、体检与化验结果" in sections


def test_page_number_tracking():
    chunks = split_documents(make_docs(), chunk_size=600, chunk_overlap=50)
    zhang = [c for c in chunks if c.metadata["patient"] == "张三九"]
    pages = {c.metadata["section"]: c.metadata["page"] for c in zhang}
    # 第一页的章节
    assert pages["一、基本信息"] == 1
    # 跨页后的章节应指向第 2 页
    assert pages["四、体检与化验结果"] == 2


def test_table_is_not_broken():
    chunks = split_documents(make_docs(), chunk_size=600, chunk_overlap=50)
    table_chunks = [c for c in chunks if "| 空腹血糖 |" in c.page_content]
    assert table_chunks, "表格内容应被保留"
    body = table_chunks[0].page_content
    # 表头、分隔行、两条数据行都在同一个 chunk 里
    assert "| 项目 | 结果 | 参考范围 | 评估 |" in body
    assert "| 总胆固醇 |" in body


def test_chunk_size_respected_for_long_text():
    long_text = "一、基本信息\n" + ("张三九的血糖为6.8毫摩尔/升。" * 200)
    docs = [Document(page_content=long_text, metadata={"source": "long.txt", "doc_title": "long", "page": 1})]
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert max(len(c.page_content) for c in chunks) <= 450
