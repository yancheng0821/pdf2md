import importlib.util
import io
import json
from pathlib import Path

import fitz
import pytest
from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "pdf2md.py"
SPEC = importlib.util.spec_from_file_location("pdf2md", MODULE_PATH)
parse_pdf = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(parse_pdf)


# ── 文本质量检测 ──────────────────────────────────────

def test_is_garbled_detects_high_frequency_cjk():
    garbled = "基基基基基基基基金金金金金金金金" * 5
    assert parse_pdf.is_garbled(garbled) is True


def test_is_garbled_normal_text():
    normal = "本基金成立于2020年1月1日，认缴规模为人民币10亿元，实缴规模为8.5亿元。"
    assert parse_pdf.is_garbled(normal) is False


# ── PyMuPDF本地提取 ──────────────────────────────────

def test_extract_page_to_markdown_basic_text():
    """基本文本提取"""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello World", fontsize=11)
    page.insert_text((72, 100), "This is body text.", fontsize=11)

    result, _ = parse_pdf.extract_page_to_markdown(page)
    doc.close()

    assert "Hello World" in result
    assert "body text" in result


def test_extract_page_to_markdown_heading_detection():
    """大字号应被识别为标题"""
    doc = fitz.open()
    page = doc.new_page()
    # 大标题（字号远大于正文）
    page.insert_text((72, 72), "Big Title", fontsize=24)
    # 正文
    page.insert_text((72, 120), "Normal body text here for sizing.", fontsize=11)
    page.insert_text((72, 140), "More normal text to establish body size.", fontsize=11)

    result, _ = parse_pdf.extract_page_to_markdown(page)
    doc.close()

    assert result.startswith("#")
    assert "Big Title" in result


def test_extract_page_to_markdown_empty_page():
    """空白页应返回[空白页]"""
    doc = fitz.open()
    page = doc.new_page()
    result, _ = parse_pdf.extract_page_to_markdown(page)
    doc.close()

    assert result == "[空白页]"


def test_table_to_markdown():
    """表格转Markdown"""

    class FakeTable:
        def extract(self):
            return [
                ["姓名", "年龄", "城市"],
                ["张三", "30", "北京"],
                ["李四", "25", "上海"],
            ]
        bbox = (0, 0, 100, 100)

    md = parse_pdf._table_to_markup(FakeTable())
    assert "| 姓名 | 年龄 | 城市 |" in md
    assert "| 张三 | 30 | 北京 |" in md
    assert "| --- | --- | --- |" in md


def test_table_to_markup_falls_back_to_html_for_sparse_table():
    """稀疏表格应保留为HTML，而不是静默丢弃"""

    class FakeSparseTable:
        def extract(self):
            return [
                ["项目", "值", "备注"],
                ["资产总额", "", ""],
            ]

        bbox = (0, 0, 100, 100)

    markup = parse_pdf._table_to_markup(FakeSparseTable())
    assert "<table>" in markup
    assert "<td>资产总额</td>" in markup


def test_infer_heading_level():
    """字号→标题级别推断"""
    assert parse_pdf._infer_heading_level(24.0, 11.0) == 1  # 2.18x → h1
    assert parse_pdf._infer_heading_level(16.0, 11.0) == 2  # 1.45x → h2
    assert parse_pdf._infer_heading_level(13.0, 11.0) == 3  # 1.18x → h3
    assert parse_pdf._infer_heading_level(11.0, 11.0) == 0  # 1.0x → 正文


# ── 页面标记处理 ─────────────────────────────────────

def test_strip_page_markers():
    text = "<!-- page:1 method:text -->\n# 标题\n\n<!-- page:2 method:ocr -->\n正文"
    result = parse_pdf.strip_page_markers(text)
    assert "<!-- page:" not in result
    assert "# 标题" in result
    assert "正文" in result


# ── 质量校验 ─────────────────────────────────────────

def test_validate_conversion_detects_missing_numbers():
    # validate_conversion 现在只检查内容长度，不做数字序列比对
    source = "<!-- page:1 method:text -->\n" + "金额 1,234,567.89 " * 50
    result = "金额已省略"
    warnings = parse_pdf.validate_conversion(source, result)
    assert any("内容丢失" in w for w in warnings)


def test_validate_conversion_detects_short_output():
    source = "<!-- page:1 method:text -->\n" + "这是一段很长的正文内容。" * 100
    result = "简短摘要"
    warnings = parse_pdf.validate_conversion(source, result)
    assert any("内容丢失" in w for w in warnings)


def test_validate_conversion_no_warnings_when_good():
    content = "金额 1,234.56 日期 2024-01-01 以及其他内容填充" * 20
    source = f"<!-- page:1 method:text -->\n{content}"
    warnings = parse_pdf.validate_conversion(source, content)
    assert warnings == []


def test_validate_conversion_detects_malformed_years():
    # validate_conversion 现在只检查内容长度和乱码字符，年份检测已移至 _validate_page_output
    source = "<!-- page:1 method:vision -->\n" + "2019 年 2020 年 2021 年 " * 30
    result = "201 年"
    warnings = parse_pdf.validate_conversion(source, result)
    assert any("内容丢失" in w for w in warnings)


def test_validate_conversion_normalizes_line_broken_years():
    source = "<!-- page:1 method:vision -->\n关键财务指标 2023\n年 2024 年各季度"
    result = "关键财务指标 2023 年 2024 年各季度"
    warnings = parse_pdf.validate_conversion(source, result)
    assert not any("年份" in w for w in warnings)


def test_validate_page_output_rejects_empty_table_shell():
    analysis = {
        "has_images": False,
        "has_tables": True,
        "raw_text": "序号 2021.09 爱博分红 50 50",
    }
    markdown = (
        "<!-- page:2 method:native -->\n"
        "<table>\n"
        "  <thead><tr><th>序号</th><th>分配时间</th></tr></thead>\n"
        "  <tbody>\n"
        "  </tbody>\n"
        "</table>\n\n"
        "1\n2021.09\n爱博分红\n50\n50"
    )

    warnings = parse_pdf._validate_page_output(markdown, "native", analysis)
    assert any("表格" in w for w in warnings)


def test_validate_page_output_allows_malformed_visible_year_when_source_is_garbled():
    analysis = {
        "has_images": True,
        "has_tables": False,
        "raw_text": "睡睟睠睨年睡月",
    }
    markdown = (
        "<!-- page:2 method:vision -->\n"
        "![page-002-full](assets/page-002-full.png)\n\n"
        "Neural Galaxy Inc.\n\n"
        "202 年 12 月\n2022 年 9 月"
    )

    warnings = parse_pdf._validate_page_output(markdown, "vision", analysis)
    assert not any("年份" in w for w in warnings)


def test_validate_page_output_allows_ocr_to_omit_header_year_from_raw_text():
    analysis = {
        "has_images": True,
        "has_tables": False,
        "raw_text": "武汉博行问道创业投资合伙企业（有限合伙）\n2024 年度报告\n深圳雅济科技有限公司\n2020 年 12 月\n2021 年 5 月",
        "body_text": "深圳雅济科技有限公司\n2020 年 12 月\n2021 年 5 月",
    }
    markdown = (
        "<!-- page:5 method:vision -->\n"
        "![page-005-full](assets/page-005-full.png)\n\n"
        "深圳雅济科技有限公司\n\n"
        "2020 年 12 月\n2021 年 5 月"
    )

    warnings = parse_pdf._validate_page_output(markdown, "vision", analysis)
    assert not any("年份" in w for w in warnings)


def test_fix_proper_nouns_corrects_inserted_noise():
    text = "上海天天爱动医疗科技股份有限公司"
    fixed = parse_pdf._fix_proper_nouns(text, {"上海博动医疗科技股份有限公司"})
    assert fixed == "上海博动医疗科技股份有限公司"


# ── 阶段2合并 ────────────────────────────────────────

@pytest.mark.asyncio
async def test_stage2_merge_no_cleanup(tmp_path):
    """canonical组装应保留页标记并按manifest顺序拼接"""
    doc_dir = tmp_path / "report"
    pages_dir = doc_dir / "pages"
    pages_dir.mkdir(parents=True)
    (pages_dir / "001.md").write_text("<!-- page:1 -->\n# 标题", encoding="utf-8")
    (pages_dir / "002.md").write_text("<!-- page:2 -->\n正文内容", encoding="utf-8")
    (doc_dir / "manifest.json").write_text(json.dumps({
        "source_pdf": "report.pdf",
        "success": True,
        "pages": [
            {"page_number": 1, "validated": True, "page_file": "pages/001.md"},
            {"page_number": 2, "validated": True, "page_file": "pages/002.md"},
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    out = await parse_pdf.stage2_merge(doc_dir)

    result = out.read_text(encoding="utf-8")
    assert "<!-- page:1 -->" in result
    assert "<!-- page:2 -->" in result
    assert "# 标题" in result
    assert "正文内容" in result


@pytest.mark.asyncio
async def test_stage2_merge_completes_brand_header_from_leading_lines(tmp_path):
    doc_dir = tmp_path / "report"
    pages_dir = doc_dir / "pages"
    pages_dir.mkdir(parents=True)
    (pages_dir / "001.md").write_text(
        "<!-- page:1 -->\n博行资本\nINSIGHT CAPITAL\n\n武汉博行问道创业投资合伙企业（有限合伙）\n2024 年度报告",
        encoding="utf-8",
    )
    (pages_dir / "002.md").write_text(
        "<!-- page:2 -->\n武汉博行问道创业投资合伙企业（有限合伙）\n2024 年度报告\n\n正文内容",
        encoding="utf-8",
    )
    (doc_dir / "manifest.json").write_text(json.dumps({
        "source_pdf": "report.pdf",
        "success": True,
        "pages": [
            {"page_number": 1, "validated": True, "page_file": "pages/001.md"},
            {"page_number": 2, "validated": True, "page_file": "pages/002.md"},
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    out = await parse_pdf.stage2_merge(doc_dir)
    result = out.read_text(encoding="utf-8")
    page_two = result.split("<!-- page:2 -->", 1)[1]
    assert page_two.lstrip().startswith("博行资本\nINSIGHT CAPITAL\n\n武汉博行问道创业投资合伙企业（有限合伙）")


@pytest.mark.asyncio
async def test_stage2_merge_refuses_when_any_page_not_validated(tmp_path):
    doc_dir = tmp_path / "report"
    pages_dir = doc_dir / "pages"
    pages_dir.mkdir(parents=True)
    (pages_dir / "001.md").write_text("<!-- page:1 -->\n内容", encoding="utf-8")
    (doc_dir / "manifest.json").write_text(json.dumps({
        "source_pdf": "report.pdf",
        "success": False,
        "pages": [
            {"page_number": 1, "validated": False, "page_file": "pages/001.md"},
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    with pytest.raises(ValueError):
        await parse_pdf.stage2_merge(doc_dir)

    assert not (doc_dir / "canonical.md").exists()


def test_extract_page_image_creates_placeholder(tmp_path):
    """嵌入图片生成占位符和 pending OCR 任务"""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "正文", fontsize=11)

    img = Image.new("RGB", (16, 16), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(fitz.Rect(72, 120, 144, 192), stream=buf.getvalue())

    result, pending_images = parse_pdf.extract_page_to_markdown(page, page_number=1, assets_dir=tmp_path)
    doc.close()

    assert "<!-- IMG_OCR:" in result
    assert len(pending_images) == 1


def test_extract_page_large_image_becomes_pending(tmp_path):
    """大图进入 pending OCR 队列"""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "正文", fontsize=11)

    img = Image.new("RGB", (400, 400), color=(0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(fitz.Rect(72, 120, 400, 400), stream=buf.getvalue())

    result, pending_images = parse_pdf.extract_page_to_markdown(page, page_number=1, assets_dir=tmp_path)
    doc.close()

    assert "<!-- IMG_OCR:" in result
    assert len(pending_images) == 1


# ── 集成测试 ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_process_pdf_fails_when_ocr_fails(monkeypatch, tmp_path):
    async def fake_stage1(*args, **kwargs):
        return None

    monkeypatch.setattr(parse_pdf, "stage1_extract", fake_stage1)

    ok = await parse_pdf.process_pdf(
        pdf_path=tmp_path / "sample.pdf",
        output_dir=tmp_path / "out",
        provider="openai",
        vision_provider=None,
    )

    assert ok is False
    assert not (tmp_path / "out" / "sample" / "canonical.md").exists()


@pytest.mark.asyncio
async def test_stage1_extract_falls_back_to_vision_when_native_table_is_invalid(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()

    monkeypatch.setattr(parse_pdf, "_analyze_page", lambda _page, repeated_texts=None: {
        "has_images": False,
        "has_tables": True,
        "raw_text": "序号 2021.09 爱博分红 50 50",
        "native_candidate": True,
        "raw_text_length": 20,
        "table_count": 1,
        "image_blocks": 0,
        "has_native_text": True,
        "garbled": False,
    })
    monkeypatch.setattr(
        parse_pdf,
        "extract_page_to_markdown",
        lambda *args, **kwargs: (
            "<table><thead><tr><th>序号</th></tr></thead><tbody></tbody></table>\n\n"
            "1\n2021.09\n爱博分红\n50\n50",
            [],  # no pending images
        ),
    )
    monkeypatch.setattr(parse_pdf, "get_async_client", lambda _provider: (object(), "fake-model"))

    async def fake_convert(client, model, page_number, image_bytes, semaphore, progress, use_high_detail=False):
        del client, model, image_bytes, semaphore
        progress["done"] += 1
        return page_number, "| 序号 | 分配时间 |\n| --- | --- |\n| 1 | 2021.09 |"

    monkeypatch.setattr(parse_pdf, "convert_single_page", fake_convert)

    doc_dir = await parse_pdf.stage1_extract(
        pdf_path=pdf_path,
        output_dir=tmp_path / "out",
        provider="openai",
        vision_provider=None,
        ocr_concurrency=1,
    )

    assert doc_dir is not None
    manifest = json.loads((doc_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["success"] is True
    assert manifest["pages"][0]["method"] == "vision"


@pytest.mark.asyncio
async def test_stage1_extract_vision_output_has_no_image_ref(monkeypatch, tmp_path):
    """vision 页输出不应包含图片引用，内容已被 OCR 转为文字"""
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()

    monkeypatch.setattr(parse_pdf, "_analyze_page", lambda _page, repeated_texts=None: {
        "has_images": False,
        "has_tables": False,
        "raw_text": "",
        "body_text": "",
        "native_candidate": False,
        "raw_text_length": 0,
        "table_count": 0,
        "image_blocks": 0,
        "has_native_text": False,
        "garbled": False,
    })
    monkeypatch.setattr(parse_pdf, "get_async_client", lambda _provider: (object(), "fake-model"))

    async def fake_convert(client, model, page_number, image_bytes, semaphore, progress, use_high_detail=False):
        del client, model, image_bytes, semaphore
        progress["done"] += 1
        return page_number, "识别正文"

    monkeypatch.setattr(parse_pdf, "convert_single_page", fake_convert)

    doc_dir = await parse_pdf.stage1_extract(
        pdf_path=pdf_path,
        output_dir=tmp_path / "out",
        provider="openai",
        vision_provider=None,
        ocr_concurrency=1,
    )

    assert doc_dir is not None
    page_md = (doc_dir / "pages" / "001.md").read_text(encoding="utf-8")
    assert "![" not in page_md
    assert "识别正文" in page_md


@pytest.mark.asyncio
async def test_process_batch_passes_concurrency(monkeypatch, tmp_path):
    seen = []

    async def fake_process_pdf(pdf_path, output_dir, provider, vision_provider=None,
                               file_sem=None, ocr_concurrency=None, chunk_chars=None,
                               stage="all"):
        seen.append({
            "pdf": Path(pdf_path).name,
            "ocr_concurrency": ocr_concurrency,
        })
        return True

    monkeypatch.setattr(parse_pdf, "process_pdf", fake_process_pdf)

    pdfs = [tmp_path / "a.pdf", tmp_path / "b.pdf"]
    for pdf in pdfs:
        pdf.write_text("stub", encoding="utf-8")

    await parse_pdf.process_batch(
        pdf_files=pdfs,
        output_dir=tmp_path / "out",
        provider="openai",
        vision_provider=None,
        file_concurrency=7,
        ocr_concurrency=2,
        chunk_chars=1234,
    )

    assert [item["ocr_concurrency"] for item in seen] == [2, 2]
