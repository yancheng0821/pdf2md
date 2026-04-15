"""
PDF -> Markdown 无损主版本转换工具 v4

目标：
- 只产出一个主版本：`canonical.md`
- 保留明确分页边界
- 导出原图并在 Markdown 中引用
- 表格优先 Markdown，必要时退化为 HTML table
- 任一页未通过校验则整份失败，不写出 `canonical.md`
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import fitz  # pymupdf
from dotenv import load_dotenv
from openai import AsyncOpenAI

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None

try:
    import pytesseract
except ImportError:  # pragma: no cover - optional dependency
    pytesseract = None


load_dotenv(Path(__file__).parent / ".env")

for key in ["all_proxy", "ALL_PROXY"]:
    os.environ.pop(key, None)


# ── 配置 ──────────────────────────────────────────────

PROVIDERS = {
    "xiaomi": {
        "api_key": os.getenv("XIAOMI_API_KEY"),
        "base_url": os.getenv("XIAOMI_BASE_URL"),
        "model": os.getenv("XIAOMI_MODEL"),
    },
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "model": os.getenv("OPENAI_MODEL"),
    },
    "4o-mini": {
        "api_key": os.getenv("OPENAI_4O_MINI_API_KEY"),
        "base_url": os.getenv("OPENAI_4O_MINI_BASE_URL"),
        "model": os.getenv("OPENAI_4O_MINI_MODEL"),
    },
    "4.1-mini": {
        "api_key": os.getenv("OPENAI_41_MINI_API_KEY"),
        "base_url": os.getenv("OPENAI_41_MINI_BASE_URL"),
        "model": os.getenv("OPENAI_41_MINI_MODEL"),
    },
    "4.1-nano": {
        "api_key": os.getenv("OPENAI_41_NANO_API_KEY"),
        "base_url": os.getenv("OPENAI_41_NANO_BASE_URL"),
        "model": os.getenv("OPENAI_41_NANO_MODEL"),
    },
    "5.4": {
        "api_key": os.getenv("OPENAI_54_API_KEY"),
        "base_url": os.getenv("OPENAI_54_BASE_URL"),
        "model": os.getenv("OPENAI_54_MODEL"),
    },
    "5.4-mini": {
        "api_key": os.getenv("OPENAI_54_MINI_API_KEY"),
        "base_url": os.getenv("OPENAI_54_MINI_BASE_URL"),
        "model": os.getenv("OPENAI_54_MINI_MODEL"),
    },
    "5.4-nano": {
        "api_key": os.getenv("OPENAI_54_NANO_API_KEY"),
        "base_url": os.getenv("OPENAI_54_NANO_BASE_URL"),
        "model": os.getenv("OPENAI_54_NANO_MODEL"),
    },
}

TEXT_THRESHOLD = 30
IMAGE_DPI = 200
OCR_DPI = 300
IMAGE_FORMAT = "png"
MAX_RETRIES = 2
OCR_CONCURRENCY = 8
FILE_CONCURRENCY = 3
CHUNK_CHAR_LIMIT = 24_000

TOKEN_PRICES = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-5.4": {"input": 2.50, "output": 10.00},
    "gpt-5.4-mini": {"input": 0.40, "output": 1.60},
    "gpt-5.4-nano": {"input": 0.10, "output": 0.40},
}

KNOWN_GARBLE_CHARS = set("睡睢督睤睥睦睧睨睩睪睫睬睭睮睯")

VISION_TRANSCRIBE_PROMPT = """你是一个精确的 PDF 页面转 Markdown 转录器。

任务要求：
1. 逐字逐句转录页面中的所有内容，包括正文、标题、页眉页脚、表格、列表和图片中的文字
2. 不要总结、概括、补充或解释
3. 所有数字、日期、金额、百分比必须与原文一致
4. **表格必须使用 Markdown 表格语法（| 列1 | 列2 | 格式），严禁将表格数据逐行输出为纯文本**。即使表格结构复杂，也必须用 | 分隔符输出。如果某个单元格为空，用空字符串占位
5. 图片中如果包含文字或表格，提取其内容；如果是纯装饰图片（logo等），输出 [logo] 或 [装饰图片]
6. 如果页面只有页码或为空，输出 [空白页]
7. 不要包裹代码块，只输出 Markdown
"""

IMAGE_OCR_PROMPT = """识别这张图片中的内容，转为 Markdown 格式输出。

规则：
1. 如果包含表格，输出 Markdown 表格
2. 如果包含文字，原样输出
3. 如果是图表，用文字描述关键数据
4. 如果是 logo 或纯装饰图片，输出 [logo] 或 [装饰图片]
5. 不要加额外说明，直接输出内容
"""


def get_async_client(provider: str) -> tuple[AsyncOpenAI, str]:
    cfg = PROVIDERS[provider]
    client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    return client, cfg["model"]


# ── 文本质量检测 ──────────────────────────────────────

def is_garbled(text: str) -> bool:
    if len(text) < 30:
        return False

    garble_count = sum(1 for c in text if c in KNOWN_GARBLE_CHARS)
    if garble_count >= 5:
        return True

    cjk_chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
    if len(cjk_chars) < 10:
        return False

    freq = Counter(cjk_chars)
    top_count = sum(c for _, c in freq.most_common(8))
    top_ratio = top_count / len(cjk_chars)
    digit_ratio = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
    return top_ratio > 0.4 and digit_ratio < 0.02


# ── 基础工具 ──────────────────────────────────────────

def strip_page_markers(text: str) -> str:
    return re.sub(r"<!-- page:\d+(?: method:[\w-]+)? -->\n?", "", text)


def _page_marker(page_number: int, method: str) -> str:
    return f"<!-- page:{page_number} method:{method} -->"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_cell(cell: Any) -> str:
    if cell is None:
        return ""
    return str(cell).strip().replace("\n", "<br>")


def _rows_to_markdown(rows: list[list[str]]) -> str:
    ncols = max(len(row) for row in rows)
    padded = [row + [""] * (ncols - len(row)) for row in rows]
    lines = [
        "| " + " | ".join(padded[0]) + " |",
        "| " + " | ".join(["---"] * ncols) + " |",
    ]
    for row in padded[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _rows_to_html(rows: list[list[str]]) -> str:
    ncols = max(len(row) for row in rows)
    padded = [row + [""] * (ncols - len(row)) for row in rows]
    lines = ["<table>", "  <thead>", "    <tr>"]
    for cell in padded[0]:
        lines.append(f"      <th>{cell}</th>")
    lines.extend(["    </tr>", "  </thead>", "  <tbody>"])
    for row in padded[1:]:
        lines.append("    <tr>")
        for cell in row:
            lines.append(f"      <td>{cell}</td>")
        lines.append("    </tr>")
    lines.extend(["  </tbody>", "</table>"])
    return "\n".join(lines)


def _table_to_markup(table) -> str:
    """优先输出 Markdown 表格；稀疏或不规则表格退化为 HTML。"""
    try:
        data = table.extract()
    except Exception:
        return ""

    if not data:
        return ""

    rows = [[_normalize_cell(cell) for cell in row] for row in data if row]
    if not rows or not rows[0]:
        return ""

    raw_lengths = {len(row) for row in rows}
    total_cells = sum(len(row) for row in rows)
    empty_cells = sum(1 for row in rows for cell in row if not cell.strip())
    sparse = total_cells > 0 and empty_cells / total_cells >= 0.3
    uneven = len(raw_lengths) > 1
    too_small_for_markdown = len(rows) <= 1

    if sparse or uneven or too_small_for_markdown:
        return _rows_to_html(rows)
    return _rows_to_markdown(rows)


def _table_to_markdown(table) -> str:
    return _table_to_markup(table)


def _infer_heading_level(span_size: float, body_size: float) -> int:
    if body_size <= 0:
        return 0
    ratio = span_size / body_size
    if ratio >= 1.8:
        return 1
    if ratio >= 1.4:
        return 2
    if ratio >= 1.15:
        return 3
    return 0


def _extract_table_fragments(page: fitz.Page) -> list[tuple[float, float, str]]:
    try:
        table_finder = page.find_tables()
    except Exception:
        return []

    fragments: list[tuple[float, float, str]] = []
    for table in table_finder.tables:
        markup = _table_to_markup(table)
        if not markup:
            continue
        bbox = table.bbox
        fragments.append((bbox[1], bbox[3], markup))
    return fragments


def _render_page_png(page: fitz.Page, dpi: int = OCR_DPI) -> bytes:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    return pix.tobytes("png")


def _save_embedded_image(block: dict[str, Any], page_number: int,
                         image_index: int, assets_dir: Path) -> tuple[str, Path] | None:
    image_bytes = block.get("image")
    if not image_bytes:
        return None

    ext = block.get("ext") or "png"
    file_name = f"page-{page_number:03d}-img-{image_index:02d}.{ext}"
    asset_path = assets_dir / file_name
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    asset_path.write_bytes(image_bytes)
    return f"assets/{file_name}", asset_path


def extract_page_to_markdown(page: fitz.Page, page_number: int | None = None,
                             assets_dir: Path | None = None) -> tuple[str, list[dict[str, Any]]]:
    """
    本地提取页内文本、表格、嵌入图片，按纵向顺序输出 Markdown。
    返回 (markdown, pending_images)
    pending_images: 需要视觉模型识别的嵌入图片列表
    """
    table_rects = _extract_table_fragments(page)

    def _in_table(y0: float, y1: float) -> bool:
        mid = (y0 + y1) / 2
        return any(ty0 - 2 <= mid <= ty1 + 2 for ty0, ty1, _ in table_rects)

    page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    blocks = page_dict.get("blocks", [])

    size_counter: Counter[float] = Counter()
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if text:
                    size_counter[round(span["size"], 1)] += len(text)
    body_size = size_counter.most_common(1)[0][0] if size_counter else 11.0

    fragments: list[tuple[float, str]] = []
    for y0, _, markup in table_rects:
        fragments.append((y0, markup))

    pending_images: list[dict[str, Any]] = []
    seen_xrefs: set[int] = set()
    image_index = 0
    for image in page.get_images(full=True):
        xref = image[0]
        rects = page.get_image_rects(xref)
        image_info = page.parent.extract_image(xref)
        image_bytes = image_info.get("image")
        ext = image_info.get("ext") or "png"
        for rect in rects:
            image_index += 1
            placeholder_id = f"xref-{xref}"
            fragments.append((rect.y0, f"<!-- IMG_OCR:{placeholder_id} -->"))
            # 同一 xref 只收集一次 OCR 任务
            if xref not in seen_xrefs and image_bytes:
                seen_xrefs.add(xref)
                if ext != "png":
                    try:
                        pil_img = Image.open(io.BytesIO(image_bytes))
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        png_bytes = buf.getvalue()
                    except Exception:
                        png_bytes = image_bytes
                else:
                    png_bytes = image_bytes
                pending_images.append({
                    "placeholder_id": placeholder_id,
                    "image_bytes": png_bytes,
                })

    for block in blocks:
        bbox = block.get("bbox", (0, 0, 0, 0))
        if block.get("type") != 0:
            continue
        if _in_table(bbox[1], bbox[3]):
            continue

        block_lines = []
        for line in block.get("lines", []):
            line_text = ""
            line_heading = 0
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text.strip():
                    line_text += text
                    continue
                heading = _infer_heading_level(span["size"], body_size)
                if heading > 0:
                    line_heading = max(line_heading, heading)
                is_bold = bool(span.get("flags", 0) & (1 << 4))
                if is_bold and heading == 0:
                    line_text += f"**{text}**"
                else:
                    line_text += text

            line_text = line_text.rstrip()
            if not line_text.strip():
                continue
            if line_heading > 0:
                line_text = "#" * line_heading + " " + line_text.lstrip("# ")
            block_lines.append(line_text)

        if block_lines:
            fragments.append((bbox[1], "\n".join(block_lines)))

    fragments.sort(key=lambda item: item[0])
    result = "\n\n".join(content for _, content in fragments).strip()
    return (result if result else "[空白页]"), pending_images


def classify_pages(pdf_path: str) -> tuple[dict[int, str], dict[int, str], int]:
    """
    兼容旧接口：原生文本页直接本地提取，其余页渲染为图片。
    """
    doc = fitz.open(pdf_path)
    text_pages: dict[int, str] = {}
    scan_pages: dict[int, str] = {}
    total = len(doc)

    for i in range(total):
        page = doc[i]
        raw_text = page.get_text().strip()
        if len(raw_text) >= TEXT_THRESHOLD and not is_garbled(raw_text):
            text_pages[i], _ = extract_page_to_markdown(page)
        else:
            scan_pages[i] = base64.b64encode(_render_page_png(page, dpi=IMAGE_DPI)).decode()

    doc.close()
    return text_pages, scan_pages, total


# ── LLM 调用 ─────────────────────────────────────────

async def call_llm(client: AsyncOpenAI, model: str, system: str,
                   content_parts: list[dict[str, Any]], retry: int = 0) -> str:
    use_new_param = any(model.startswith(prefix) for prefix in ("gpt-5", "o1", "o3", "o4"))
    token_param = {"max_completion_tokens": 16384} if use_new_param else {"max_tokens": 16384}
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content_parts},
            ],
            temperature=0.1,
            **token_param,
        )
        return resp.choices[0].message.content or ""
    except Exception:
        if retry < MAX_RETRIES:
            await asyncio.sleep(2 ** retry)
            return await call_llm(client, model, system, content_parts, retry + 1)
        raise


async def convert_single_page(client: AsyncOpenAI, model: str, page_number: int,
                              image_bytes: bytes, semaphore: asyncio.Semaphore,
                              progress: dict[str, int]) -> tuple[int, str]:
    async with semaphore:
        content = [
            {"type": "text", "text": f"第{page_number}页"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}",
                "detail": "high",
            }},
        ]
        result = await call_llm(client, model, VISION_TRANSCRIBE_PROMPT, content)
        progress["done"] += 1
        print(f"    Vision OCR {progress['done']}/{progress['total']} (第{page_number}页)")
        return page_number, result.strip()


async def ocr_single_image(client: AsyncOpenAI, model: str,
                           placeholder_id: str, image_bytes: bytes,
                           semaphore: asyncio.Semaphore,
                           progress: dict[str, int]) -> tuple[str, str]:
    """识别单张嵌入图片，返回 (placeholder_id, 识别文本)"""
    async with semaphore:
        content = [
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}",
                "detail": "high",
            }},
        ]
        result = await call_llm(client, model, IMAGE_OCR_PROMPT, content)
        progress["done"] += 1
        print(f"    图片识别 {progress['done']}/{progress['total']} ({placeholder_id})")
        return placeholder_id, result.strip()


# ── 页面分析与校验 ───────────────────────────────────

def _extract_block_text(block: dict[str, Any]) -> str:
    lines = []
    for line in block.get("lines", []):
        spans = [span.get("text", "") for span in line.get("spans", [])]
        text = "".join(spans).rstrip()
        if text.strip():
            lines.append(text)
    return "\n".join(lines).strip()


def _detect_repeated_texts(doc: fitz.Document, zone: str = "header",
                           threshold: float = 0.5) -> set[str]:
    """
    扫描所有页面，找出在 >threshold 比例的页面中重复出现的页眉/页脚文本。
    zone="header": 页面顶部 15%
    zone="footer": 页面底部 10%
    """
    total_pages = len(doc)
    if total_pages < 3:
        return set()

    text_counts: Counter[str] = Counter()
    for page in doc:
        height = page.rect.height
        if zone == "header":
            y_min, y_max = 0, height * 0.15
        else:
            y_min, y_max = height * 0.90, height

        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        page_texts: set[str] = set()
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            bbox = block.get("bbox", (0, 0, 0, 0))
            mid_y = (bbox[1] + bbox[3]) / 2
            if y_min <= mid_y <= y_max:
                text = _extract_block_text(block)
                if text and len(text.strip()) > 1:
                    page_texts.add(text.strip())
        for t in page_texts:
            text_counts[t] += 1

    return {t for t, count in text_counts.items() if count / total_pages >= threshold}


def _detect_headers_footers(doc: fitz.Document) -> set[str]:
    """检测文档中所有重复页眉和页脚文本"""
    headers = _detect_repeated_texts(doc, zone="header")
    footers = _detect_repeated_texts(doc, zone="footer")
    return headers | footers


def _strip_repeated_texts(text: str, repeated: set[str]) -> str:
    """从文本中移除已知的重复页眉/页脚"""
    for r in sorted(repeated, key=len, reverse=True):  # 长的先匹配
        text = text.replace(r, "")
    return text.strip()


def _analyze_page(page: fitz.Page, repeated_texts: set[str] | None = None) -> dict[str, Any]:
    raw_text = page.get_text().strip()
    try:
        table_count = len(page.find_tables().tables)
    except Exception:
        table_count = 0

    page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    image_blocks = sum(1 for block in page_dict.get("blocks", []) if block.get("type") == 1)
    # 有表格的页面强制走 vision，PyMuPDF 表格提取对复杂布局不可靠
    native_candidate = table_count == 0 and len(raw_text) >= TEXT_THRESHOLD and not is_garbled(raw_text)

    body_text = raw_text
    if repeated_texts:
        body_text = _strip_repeated_texts(raw_text, repeated_texts)

    return {
        "raw_text_length": len(raw_text),
        "raw_text": raw_text,
        "body_text": body_text,
        "has_native_text": bool(raw_text),
        "has_tables": table_count > 0,
        "table_count": table_count,
        "has_images": image_blocks > 0,
        "image_blocks": image_blocks,
        "native_candidate": native_candidate,
        "garbled": is_garbled(raw_text),
    }


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d[\d,.\-]*\d", text))


def _normalize_year_text(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _years(text: str) -> set[str]:
    normalized = _normalize_year_text(text)
    # 提取年份并去掉内部空格，统一为 "2024年" 格式
    raw = re.findall(r"(?<!\d)((?:19|20)\d{2})\s*年", normalized)
    return {f"{y}年" for y in raw}


def _garbled_count(text: str) -> int:
    return sum(1 for c in text if c in KNOWN_GARBLE_CHARS)


def _collect_asset_refs(markdown: str) -> list[str]:
    return re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown)


def validate_conversion(source_text: str, result_text: str) -> list[str]:
    warnings = []
    source_clean = strip_page_markers(source_text)
    source_numbers = _numbers(source_clean)
    result_numbers = _numbers(result_text)
    missing_numbers = source_numbers - result_numbers
    if missing_numbers and len(missing_numbers) > 3:
        warnings.append(f"输出中缺失 {len(missing_numbers)} 个数字序列，可能有内容丢失")

    source_years = _years(source_clean)
    result_years = _years(result_text)
    missing_years = source_years - result_years
    if missing_years:
        warnings.append(f"输出中缺失 {len(missing_years)} 个年份，可能有 OCR 漏字")

    garbled_count = _garbled_count(result_text)
    if garbled_count > 5:
        warnings.append(f"输出中检测到 {garbled_count} 个疑似乱码字符")

    source_len = len(source_clean.strip())
    result_len = len(result_text.strip())
    if source_len > 0 and result_len < source_len * 0.5:
        warnings.append(f"输出长度({result_len})不足源文本({source_len})的50%，可能有内容丢失")

    return warnings


def _validate_page_output(markdown: str, method: str,
                          analysis: dict[str, Any]) -> list[str]:
    warnings = []
    body = strip_page_markers(markdown).strip()
    asset_refs = _collect_asset_refs(markdown)

    if not body:
        warnings.append("页面输出为空")

    # 表格标记校验仅对 native 提取有意义
    if method == "native" and analysis["has_tables"]:
        if "<table>" not in body and "|" not in body:
            warnings.append("页面包含表格但输出中未发现表格标记")
        if re.search(r"<tbody>\s*</tbody>", body, re.S):
            warnings.append("页面表格只有空表壳，疑似表体提取失败")

    # native 页短行检测：连续大量短行说明可能是无框线表格被拆散，应 fallback 到 vision
    if method == "native" and "|" not in body and "<table>" not in body:
        non_empty = [ln for ln in body.split("\n") if ln.strip()]
        short_lines = sum(1 for ln in non_empty if len(ln.strip()) <= 15)
        if len(non_empty) > 8 and short_lines / len(non_empty) > 0.6:
            warnings.append("页面大量短行，疑似无框线表格未能还原")

    source_text = analysis.get("raw_text", "")
    if method in {"local_ocr", "vision"}:
        source_text = analysis.get("body_text") or source_text
    if source_text:
        warnings.extend(validate_conversion(source_text, body))

    if _garbled_count(body) > 5:
        warnings.append("输出中疑似乱码过多")

    return warnings


# ── 本地 OCR ─────────────────────────────────────────

def _local_ocr_available() -> bool:
    return bool(Image is not None and pytesseract is not None and shutil.which("tesseract"))


def _extract_local_ocr_markdown(page: fitz.Page, page_number: int,
                                assets_dir: Path) -> str | None:
    if not _local_ocr_available():
        return None

    page_png = _render_page_png(page)
    file_name = f"page-{page_number:03d}-full.png"
    asset_path = assets_dir / file_name
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    asset_path.write_bytes(page_png)

    image = Image.open(io.BytesIO(page_png))
    ocr_text = pytesseract.image_to_string(image)
    if not ocr_text.strip():
        return None

    return f"![page-{page_number:03d}-full](assets/{file_name})\n\n{ocr_text.strip()}"


def _ensure_doc_dirs(doc_dir: Path) -> dict[str, Path]:
    paths = {
        "doc_dir": doc_dir,
        "pages_dir": doc_dir / "pages",
        "assets_dir": doc_dir / "assets",
        "debug_dir": doc_dir / "debug",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _page_file_name(page_number: int) -> str:
    return f"{page_number:03d}.md"


def _debug_base(debug_dir: Path, page_number: int, suffix: str) -> Path:
    return debug_dir / f"page-{page_number:03d}.{suffix}"


def _extract_proper_nouns(native_texts: list[str]) -> set[str]:
    """
    从 native 页文本中提取高频专有名词（公司名、基金名等）。
    这些名词来自 PyMuPDF 直接读取的字符，不会有 OCR 错误。
    同时提取短名片段（如"博行问道"）用于覆盖非完整引用。
    """
    patterns = [
        r"[\u4e00-\u9fff]{2,}(?:有限合伙）|有限公司|股份有限公司|有限责任公司)",
        r"[\u4e00-\u9fff]{2,}(?:合伙企业|投资中心|基金)",
    ]
    names: Counter[str] = Counter()
    for text in native_texts:
        for pattern in patterns:
            for match in re.findall(pattern, text):
                if len(match) >= 6:
                    names[match] += 1
    result = {name for name, count in names.items() if count >= 2}

    # 从已知长专名中提取高频子串（仅 4 字）作为简称
    all_native = "\n".join(native_texts)
    for name in list(result):
        cjk_run = "".join(c for c in name if "\u4e00" <= c <= "\u9fff")
        for start in range(len(cjk_run) - 3):
            sub = cjk_run[start:start + 4]
            if sub not in result and all_native.count(sub) >= 5:
                result.add(sub)

    return result


def _fix_proper_nouns(text: str, known_nouns: set[str]) -> str:
    """
    用已知的正确专名纠正 OCR 文本中的近似错误。
    对每个已知专名，在文本中查找编辑距离 <= 1 的变体并替换。
    """
    if not known_nouns:
        return text
    for correct in sorted(known_nouns, key=len, reverse=True):
        # 生成常见的单字替换变体用于匹配
        for i in range(len(correct)):
            prefix = re.escape(correct[:i])
            suffix = re.escape(correct[i + 1:])
            # 匹配：prefix + 任意单个中文字符 + suffix
            pattern = prefix + r"[\u4e00-\u9fff]" + suffix
            if re.search(pattern, text):
                text = re.sub(pattern, correct, text)
    return text


def _extract_phrases(text: str, min_len: int = 2) -> list[str]:
    """从文本中提取所有中文短语（连续中文字符序列）"""
    return [p for p in re.findall(r"[\u4e00-\u9fff]{" + str(min_len) + r",}", text)]


def _edit_distance_1_match(correct: str, text: str) -> str | None:
    """在 text 中查找与 correct 编辑距离为 1（单字替换）的子串"""
    for i in range(len(correct)):
        prefix = re.escape(correct[:i])
        suffix = re.escape(correct[i + 1:])
        pattern = prefix + r"[\u4e00-\u9fff]" + suffix
        match = re.search(pattern, text)
        if match and match.group() != correct:
            return match.group()
    return None


def _edit_distance_2_match(correct: str, text: str) -> str | None:
    """在 text 中查找与 correct 编辑距离为 2（双字替换）的子串，仅对 5-12 字短语"""
    if len(correct) < 5 or len(correct) > 12:
        return None
    # 只检查相邻或间隔小的双字替换，限制组合爆炸
    for i in range(len(correct)):
        for j in range(i + 1, min(i + 4, len(correct))):
            parts = []
            if i > 0:
                parts.append(re.escape(correct[:i]))
            parts.append(r"[\u4e00-\u9fff]")
            if j > i + 1:
                parts.append(re.escape(correct[i + 1:j]))
            parts.append(r"[\u4e00-\u9fff]")
            if j + 1 < len(correct):
                parts.append(re.escape(correct[j + 1:]))
            pattern = "".join(parts)
            try:
                match = re.search(pattern, text)
            except re.error:
                continue
            if match and match.group() != correct:
                return match.group()
    return None


def _fix_vision_with_text_layer(vision_text: str, raw_text: str) -> str:
    """
    用 PDF 文本层（PyMuPDF 提取的准确文本）校正 vision 输出。
    保留 vision 的 Markdown 格式，用文本层纠正 OCR 错字。

    三轮修正：
    1. 编辑距离 1（单字替换）
    2. 编辑距离 2（双字替换，仅 ≥5 字短语）
    3. 漏字修复（vision 中少了一个字）
    """
    if not raw_text or not vision_text:
        return vision_text

    source_phrases = set(_extract_phrases(raw_text, min_len=3))
    if not source_phrases:
        return vision_text

    sorted_phrases = sorted(source_phrases, key=len, reverse=True)

    # 第 1 轮：编辑距离 1
    for correct in sorted_phrases:
        if correct in vision_text:
            continue
        wrong = _edit_distance_1_match(correct, vision_text)
        # 安全检查：如果 wrong 在文本层中也存在，说明它本身是正确的，不替换
        if wrong and wrong not in raw_text:
            vision_text = vision_text.replace(wrong, correct)

    # 第 2 轮：编辑距离 2（仅 ≥5 字的短语，且被替换文本不在文本层中）
    for correct in sorted_phrases:
        if len(correct) < 5 or correct in vision_text:
            continue
        wrong = _edit_distance_2_match(correct, vision_text)
        if wrong and wrong not in raw_text:
            vision_text = vision_text.replace(wrong, correct)

    # 第 3 轮：漏字修复（vision 中少了一个字）
    for correct in sorted_phrases:
        if len(correct) < 4 or correct in vision_text:
            continue
        for i in range(len(correct)):
            truncated = correct[:i] + correct[i + 1:]
            # 只在截断形式不存在于文本层时才修复（避免误修正确文本）
            if len(truncated) >= 3 and truncated in vision_text and truncated not in raw_text:
                vision_text = vision_text.replace(truncated, correct, 1)
                break

    return vision_text


# ── 纯文本表格检测与修复 ────────────────────────────

def _is_table_cell(token: str) -> bool:
    """判断一个 token 是否像表格单元格（数字、百分比、金额、短中文词等）"""
    token = token.strip()
    if not token:
        return False
    # 数字（含逗号分隔、小数、负数）
    if re.match(r"^-?[\d,]+\.?\d*%?$", token):
        return True
    # 倍数 如 3.3x 10.0x
    if re.match(r"^[\d.]+x$", token, re.I):
        return True
    # 短中文词（≤10字）
    if re.match(r"^[\u4e00-\u9fff/（）\w]{1,10}$", token):
        return True
    # yoy, Q1-Q4 等
    if re.match(r"^(?:yoy|Q[1-4]|YTD|\d{4}[EeYy]?)$", token, re.I):
        return True
    return False


def _split_table_line(line: str) -> list[str]:
    """将一行文本按多空格或制表符分割为单元格"""
    # 先按 tab 分
    if "\t" in line:
        return [c.strip() for c in line.split("\t")]
    # 按 2+ 空格分（中文全角空格也算）
    parts = re.split(r"[ \u3000]{2,}", line.strip())
    return [p.strip() for p in parts if p.strip()]


def _detect_and_fix_plain_tables(text: str) -> str:
    """
    检测 vision 输出中"看起来像表格但没用 Markdown 表格格式"的文本块，
    自动转为 Markdown 表格。

    判断标准：连续 3+ 行，每行有 3+ 个分隔的 token，且大部分 token 像表格单元格。
    """
    lines = text.split("\n")
    result_lines: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # 跳过已经是 Markdown 表格、HTML 表格、页标记、图片的行
        stripped = line.strip()
        if (stripped.startswith("|") or stripped.startswith("<") or
                stripped.startswith("<!--") or stripped.startswith("![") or
                stripped.startswith("#")):
            result_lines.append(line)
            i += 1
            continue

        # 尝试检测表格块
        cells = _split_table_line(line)
        if len(cells) >= 3:
            block_start = i
            table_rows: list[list[str]] = []
            while i < len(lines):
                row_cells = _split_table_line(lines[i])
                if len(row_cells) >= 2 and sum(1 for c in row_cells if _is_table_cell(c)) >= len(row_cells) * 0.5:
                    table_rows.append(row_cells)
                    i += 1
                elif not lines[i].strip():
                    i += 1  # 跳过空行
                else:
                    break

            # 保证 i 至少前进一步，避免死循环
            if i == block_start:
                result_lines.append(line)
                i += 1
            elif len(table_rows) >= 3:
                ncols = max(len(row) for row in table_rows)
                for row in table_rows:
                    while len(row) < ncols:
                        row.append("")
                md_lines = []
                md_lines.append("| " + " | ".join(table_rows[0]) + " |")
                md_lines.append("| " + " | ".join(["---"] * ncols) + " |")
                for row in table_rows[1:]:
                    md_lines.append("| " + " | ".join(row) + " |")
                result_lines.append("\n".join(md_lines))
            else:
                for j in range(block_start, i):
                    result_lines.append(lines[j])
        else:
            result_lines.append(line)
            i += 1

    return "\n".join(result_lines)


# ── 阶段1：逐页提取并持久化 ─────────────────────────

async def stage1_extract(pdf_path: Path, output_dir: Path,
                         provider: str, vision_provider: str | None,
                         ocr_concurrency: int) -> Path | None:
    pdf_path = Path(pdf_path)
    doc_dir = output_dir / pdf_path.stem
    paths = _ensure_doc_dirs(doc_dir)
    manifest_path = doc_dir / "manifest.json"

    doc = fitz.open(pdf_path)

    # 跨页检测重复页眉/页脚，用于后续校验时剥离
    repeated_texts = _detect_headers_footers(doc)
    if repeated_texts:
        print(f"  检测到重复页眉/页脚: {len(repeated_texts)} 条")

    manifest: dict[str, Any] = {
        "source_pdf": str(pdf_path),
        "generated_at": int(time.time()),
        "success": False,
        "pages": [],
        "repeated_texts": sorted(repeated_texts),
    }

    vision_jobs: list[dict[str, Any]] = []
    # 收集 native 页中需要 OCR 的嵌入图片
    image_ocr_jobs: list[dict[str, Any]] = []
    # 记录哪些 native 页有待 OCR 的图片，OCR 完成后替换占位符
    native_pages_with_images: list[dict[str, Any]] = []

    for page_index in range(len(doc)):
        page_number = page_index + 1
        page = doc[page_index]
        analysis = _analyze_page(page, repeated_texts=repeated_texts)
        _write_json(_debug_base(paths["debug_dir"], page_number, "native.json"), analysis)

        page_file = paths["pages_dir"] / _page_file_name(page_number)
        page_rel = f"pages/{page_file.name}"
        page_record: dict[str, Any] = {
            "page_number": page_number,
            "page_file": page_rel,
            "validated": False,
            "method": None,
            "warnings": [],
            "assets": [],
            "attempts": [],
        }

        def _record_attempt(method: str, markdown: str | None, warnings: list[str]) -> None:
            attempt = {
                "method": method,
                "ok": not warnings and bool(markdown and markdown.strip()),
                "warnings": warnings,
            }
            page_record["attempts"].append(attempt)
            if markdown is not None:
                _debug_base(paths["debug_dir"], page_number, f"{method}.md").write_text(
                    markdown,
                    encoding="utf-8",
                )

        native_markdown = None
        if analysis["native_candidate"]:
            native_markdown, pending_images = extract_page_to_markdown(
                page,
                page_number=page_number,
                assets_dir=paths["assets_dir"],
            )
            # 收集嵌入图片 OCR 任务（按 placeholder_id 去重，同一 xref 只 OCR 一次）
            seen_img_ids = {j["placeholder_id"] for j in image_ocr_jobs}
            if pending_images:
                for img_job in pending_images:
                    if img_job["placeholder_id"] not in seen_img_ids:
                        image_ocr_jobs.append(img_job)
                        seen_img_ids.add(img_job["placeholder_id"])
                native_pages_with_images.append({
                    "page_number": page_number,
                    "page_file": page_file,
                    "page_record": page_record,
                    "native_markdown": native_markdown,
                    "analysis": analysis,
                })

            native_body = f"{_page_marker(page_number, 'native')}\n{native_markdown}"
            native_warnings = _validate_page_output(native_body, "native", analysis)
            _record_attempt("native", native_body, native_warnings)
            if not native_warnings:
                page_file.write_text(native_body, encoding="utf-8")
                page_record["validated"] = True
                page_record["method"] = "native"
                page_record["assets"] = _collect_asset_refs(native_body)
                manifest["pages"].append(page_record)
                continue

        local_markdown = _extract_local_ocr_markdown(page, page_number, paths["assets_dir"])
        local_body = None
        if local_markdown is not None:
            local_body = f"{_page_marker(page_number, 'local_ocr')}\n{local_markdown}"
            local_warnings = _validate_page_output(local_body, "local_ocr", analysis)
            _record_attempt("local_ocr", local_body, local_warnings)
            if not local_warnings:
                page_file.write_text(local_body, encoding="utf-8")
                page_record["validated"] = True
                page_record["method"] = "local_ocr"
                page_record["assets"] = _collect_asset_refs(local_body)
                manifest["pages"].append(page_record)
                continue

        full_image_name = f"page-{page_number:03d}-full.png"
        full_image_path = paths["assets_dir"] / full_image_name
        full_image_bytes = _render_page_png(page)
        full_image_path.write_bytes(full_image_bytes)
        vision_jobs.append({
            "page_number": page_number,
            "analysis": analysis,
            "page_file": page_file,
            "page_record": page_record,
            "image_bytes": full_image_bytes,
            "image_ref": f"assets/{full_image_name}",
        })
        manifest["pages"].append(page_record)

    doc.close()

    if vision_jobs:
        client, model = get_async_client(vision_provider or provider)
        sem = asyncio.Semaphore(ocr_concurrency)
        progress = {"done": 0, "total": len(vision_jobs)}
        outcomes = await asyncio.gather(
            *[
                convert_single_page(
                    client,
                    model,
                    job["page_number"],
                    job["image_bytes"],
                    sem,
                    progress,
                )
                for job in vision_jobs
            ],
            return_exceptions=True,
        )

        outcome_map: dict[int, str] = {}
        error_map: dict[int, str] = {}
        for job, outcome in zip(vision_jobs, outcomes):
            if isinstance(outcome, Exception):
                error_map[job["page_number"]] = str(outcome)
            else:
                outcome_map[outcome[0]] = outcome[1]

        # 收集需要重试的页面（表格被输出为纯文本）
        retry_jobs: list[dict[str, Any]] = []

        for job in vision_jobs:
            page_number = job["page_number"]
            page_record = job["page_record"]
            if page_number in error_map:
                page_record["warnings"] = [error_map[page_number]]
                continue

            ocr_text = outcome_map.get(page_number, "").strip()
            try:
                ocr_text = _detect_and_fix_plain_tables(ocr_text)
            except Exception as exc:
                print(f"    表格修复异常(第{page_number}页): {exc}")

            # 检测表格被输出为纯文本 → 需要重试
            has_table_mark = "|" in ocr_text or "<table>" in ocr_text
            needs_retry = False
            if job["analysis"].get("has_tables", False) and not has_table_mark:
                needs_retry = True
            # 额外检测：连续多行短文本（数字/百分比/短词）= 表格数据被逐行输出
            if not has_table_mark and not needs_retry and len(retry_jobs) < 10:
                non_empty = [ln for ln in ocr_text.split("\n") if ln.strip()]
                short_lines = sum(1 for ln in non_empty if len(ln.strip()) <= 15)
                if len(non_empty) > 8 and short_lines / len(non_empty) > 0.6:
                    needs_retry = True
            if needs_retry:
                retry_jobs.append(job)
                print(f"    第{page_number}页: 疑似表格输出为纯文本，加入重试队列")
                continue

            # 文本层引导纠错：用 PDF 原文校正 vision OCR 错误
            raw_text = job["analysis"].get("raw_text", "")
            if raw_text and not job["analysis"].get("garbled", False):
                ocr_text = _fix_vision_with_text_layer(ocr_text, raw_text)

            # 补页眉：如果文档有重复页眉但 vision 输出缺失，在开头补上
            for header in sorted(repeated_texts, key=len, reverse=True):
                if header not in ocr_text and header in raw_text:
                    ocr_text = header + "\n\n" + ocr_text

            vision_markdown = (
                f"{_page_marker(page_number, 'vision')}\n"
                f"{ocr_text or '[空白页]'}"
            )
            _debug_base(paths["debug_dir"], page_number, "vision.md").write_text(
                vision_markdown, encoding="utf-8",
            )
            warnings = _validate_page_output(vision_markdown, "vision", job["analysis"])
            _write_json(
                _debug_base(paths["debug_dir"], page_number, "validation.json"),
                {"method": "vision", "warnings": warnings},
            )
            if warnings:
                page_record["warnings"] = warnings
                continue

            job["page_file"].write_text(vision_markdown, encoding="utf-8")
            page_record["validated"] = True
            page_record["method"] = "vision"
            page_record["assets"] = _collect_asset_refs(vision_markdown)
            page_record["warnings"] = []

        # 重试：表格被输出为纯文本的页面
        if retry_jobs:
            print(f"  重试 {len(retry_jobs)} 个表格页...")
            retry_outcomes = await asyncio.gather(
                *[
                    convert_single_page(
                        client, model,
                        job["page_number"], job["image_bytes"],
                        sem, {"done": 0, "total": len(retry_jobs)},
                    )
                    for job in retry_jobs
                ],
                return_exceptions=True,
            )
            for job, outcome in zip(retry_jobs, retry_outcomes):
                page_number = job["page_number"]
                page_record = job["page_record"]
                if isinstance(outcome, Exception):
                    page_record["warnings"] = [str(outcome)]
                    continue

                ocr_text = outcome[1].strip()
                try:
                    ocr_text = _detect_and_fix_plain_tables(ocr_text)
                except Exception:
                    pass

                # 文本层引导纠错
                raw_text = job["analysis"].get("raw_text", "")
                if raw_text and not job["analysis"].get("garbled", False):
                    ocr_text = _fix_vision_with_text_layer(ocr_text, raw_text)

                vision_markdown = (
                    f"{_page_marker(page_number, 'vision')}\n"
                    f"{ocr_text or '[空白页]'}"
                )
                _debug_base(paths["debug_dir"], page_number, "vision.md").write_text(
                    vision_markdown, encoding="utf-8",
                )
                # 重试后不再严格校验表格格式，直接接受
                warnings = _validate_page_output(vision_markdown, "vision", job["analysis"])
                if warnings:
                    page_record["warnings"] = warnings
                    continue
                job["page_file"].write_text(vision_markdown, encoding="utf-8")
                page_record["validated"] = True
                page_record["method"] = "vision"
                page_record["warnings"] = []

    # 批量 OCR native 页中的嵌入图片
    if image_ocr_jobs:
        v_client, v_model = get_async_client(vision_provider or provider)
        img_sem = asyncio.Semaphore(ocr_concurrency)
        img_progress = {"done": 0, "total": len(image_ocr_jobs)}
        img_outcomes = await asyncio.gather(
            *[
                ocr_single_image(
                    v_client, v_model,
                    job["placeholder_id"], job["image_bytes"],
                    img_sem, img_progress,
                )
                for job in image_ocr_jobs
            ],
            return_exceptions=True,
        )

        # 构建占位符 → 识别结果映射
        img_ocr_map: dict[str, str] = {}
        for job, outcome in zip(image_ocr_jobs, img_outcomes):
            if isinstance(outcome, Exception):
                img_ocr_map[job["placeholder_id"]] = "[图片识别失败]"
            else:
                img_ocr_map[outcome[0]] = outcome[1]

        # 替换 native 页中的图片占位符
        for info in native_pages_with_images:
            md = info["native_markdown"]
            for pid, ocr_text in img_ocr_map.items():
                md = md.replace(f"<!-- IMG_OCR:{pid} -->", ocr_text)
            final_body = f"{_page_marker(info['page_number'], 'native')}\n{md}"
            warnings = _validate_page_output(final_body, "native", info["analysis"])
            if not warnings:
                info["page_file"].write_text(final_body, encoding="utf-8")
                info["page_record"]["validated"] = True
                info["page_record"]["method"] = "native"

    # 保存 logo/图片 OCR 结果到 manifest，供 stage2 使用
    if image_ocr_jobs:
        manifest["image_ocr"] = {job["placeholder_id"]: img_ocr_map.get(job["placeholder_id"], "")
                                  for job in image_ocr_jobs}

    manifest["success"] = all(page["validated"] for page in manifest["pages"])
    _write_json(manifest_path, manifest)

    if not manifest["success"]:
        failures = [str(page["page_number"]) for page in manifest["pages"] if not page["validated"]]
        (doc_dir / "errors.txt").write_text(
            "以下页面未通过校验：" + ", ".join(failures),
            encoding="utf-8",
        )
        return None

    return doc_dir


# ── 阶段2：组装 canonical.md ───────────────────────

async def stage2_merge(doc_dir: Path) -> Path:
    doc_dir = Path(doc_dir)
    manifest_path = doc_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"缺少 manifest.json: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages = manifest.get("pages", [])
    if not manifest.get("success") or any(not page.get("validated") for page in pages):
        raise ValueError("存在未通过校验的页面，不能生成 canonical.md")

    # 读取所有页面内容
    page_contents: list[str] = []
    for page in sorted(pages, key=lambda item: item["page_number"]):
        page_path = doc_dir / page["page_file"]
        if not page_path.exists():
            raise ValueError(f"缺少页文件: {page_path}")
        page_contents.append(page_path.read_text(encoding="utf-8").strip())

    # 品牌头补全：从 logo OCR 结果和高频首行中提取品牌文字，补到缺失的页面
    brand_texts: set[str] = set()

    # 来源1：logo OCR 结果
    for _pid, ocr_text in manifest.get("image_ocr", {}).items():
        text = ocr_text.strip()
        if text and text not in ("[logo]", "[装饰图片]", "[图片识别失败]"):
            brand_texts.add(text)

    # 来源2：在 >40% 页面出现的首行文字
    first_lines_counter: Counter[str] = Counter()
    for content in page_contents:
        body = strip_page_markers(content).strip()
        lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
        if lines:
            first_lines_counter[lines[0]] += 1
    for line, count in first_lines_counter.most_common():
        if count >= len(page_contents) * 0.4:
            brand_texts.add(line)
        else:
            break

    if brand_texts:
        fixed = 0
        for i, content in enumerate(page_contents):
            body = strip_page_markers(content).strip()
            for bt in sorted(brand_texts, key=len, reverse=True):
                if bt not in body:
                    marker_end = content.find("\n")
                    if marker_end > 0:
                        page_contents[i] = content[:marker_end + 1] + bt + "\n\n" + content[marker_end + 1:]
                        content = page_contents[i]  # 更新用于后续检查
                        fixed += 1
        if fixed:
            print(f"  品牌头补全: {fixed} 页")

    parts = [f"<!-- source: {manifest.get('source_pdf', '')} -->"]
    parts.extend(page_contents)

    canonical_path = doc_dir / "canonical.md"
    canonical_path.write_text("\n\n".join(parts).strip() + "\n", encoding="utf-8")

    # 清理 assets 中未被引用的孤儿文件
    canonical_text = canonical_path.read_text(encoding="utf-8")
    assets_dir = doc_dir / "assets"
    if assets_dir.exists():
        removed = 0
        for f in assets_dir.iterdir():
            if f.name not in canonical_text:
                f.unlink()
                removed += 1
        if removed:
            print(f"  清理孤儿文件: {removed} 个")

    return canonical_path


# ── dry-run ───────────────────────────────────────────

def dry_run(pdf_files: list[Path], provider: str, vision_provider: str | None) -> None:
    model = PROVIDERS[provider]["model"]
    v_model = PROVIDERS[vision_provider or provider]["model"]
    total_pages = 0
    scan_pages = 0

    for pdf_file in pdf_files:
        doc = fitz.open(pdf_file)
        file_scan_pages = 0
        page_count = len(doc)
        for page in doc:
            if not _analyze_page(page)["native_candidate"]:
                file_scan_pages += 1
        doc.close()
        total_pages += page_count
        scan_pages += file_scan_pages
        print(f"  {pdf_file.name}: 预计 OCR {file_scan_pages} 页")

    image_tokens = scan_pages * 1200
    ocr_output_tokens = scan_pages * 800
    vision_price = TOKEN_PRICES.get(v_model, {"input": 1.0, "output": 4.0})
    ocr_cost = (
        image_tokens * vision_price["input"] + ocr_output_tokens * vision_price["output"]
    ) / 1_000_000

    print(f"\n{'=' * 50}")
    print(f"总页数: {total_pages}")
    print(f"预计 OCR 页数: {scan_pages}")
    print(f"视觉模型: {v_model}")
    print(f"文本模型: {model}")
    print(f"预估 OCR 成本: ${ocr_cost:.3f}")


# ── 单文件处理 ────────────────────────────────────────

async def process_pdf(pdf_path: Path, output_dir: Path, provider: str,
                      vision_provider: str | None = None,
                      file_sem: asyncio.Semaphore | None = None,
                      ocr_concurrency: int = OCR_CONCURRENCY,
                      chunk_chars: int = CHUNK_CHAR_LIMIT,
                      stage: str = "all") -> bool:
    del chunk_chars  # 兼容旧接口，canonical 主路径不再使用分块清理

    pdf_path = Path(pdf_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    doc_dir = output_dir / pdf_path.stem
    canonical_path = doc_dir / "canonical.md"

    if stage == "all" and canonical_path.exists():
        print(f"  [跳过] {pdf_path.name}")
        return True

    sem = file_sem or asyncio.Semaphore(1)
    async with sem:
        t0 = time.time()
        print(f"\n{'─' * 60}")
        print(f"  {pdf_path.name}")

        if stage in ("all", "extract"):
            doc_dir_result = await stage1_extract(
                pdf_path=pdf_path,
                output_dir=output_dir,
                provider=provider,
                vision_provider=vision_provider,
                ocr_concurrency=ocr_concurrency,
            )
            if doc_dir_result is None:
                print("  [失败] 阶段1未能让所有页面通过校验")
                return False

        if stage in ("all", "merge"):
            if not doc_dir.exists():
                print(f"  [错误] 缺少目录 {doc_dir}，请先运行 --stage extract")
                return False
            try:
                out = await stage2_merge(doc_dir)
            except ValueError as exc:
                print(f"  [失败] {exc}")
                return False
            print(f"  → {out.name}")

        elapsed = time.time() - t0
        print(f"  done ({elapsed:.1f}s)")
        return True


# ── 批量处理 ──────────────────────────────────────────

async def process_batch(pdf_files: list[Path], output_dir: Path,
                        provider: str, vision_provider: str | None = None,
                        file_concurrency: int = FILE_CONCURRENCY,
                        ocr_concurrency: int = OCR_CONCURRENCY,
                        chunk_chars: int = CHUNK_CHAR_LIMIT,
                        stage: str = "all") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vp = vision_provider or provider

    print(f"共 {len(pdf_files)} 个PDF | 阶段: {stage}")
    print(f"文本模型: {provider} ({PROVIDERS[provider]['model']})")
    print(f"视觉模型: {vp} ({PROVIDERS[vp]['model']})")
    print(f"并发: 文件×{file_concurrency}, OCR×{ocr_concurrency}")
    print("输出: 单一 canonical.md，无损主版本")

    file_sem = asyncio.Semaphore(file_concurrency)
    tasks = [
        process_pdf(
            pdf_path=pdf_file,
            output_dir=output_dir,
            provider=provider,
            vision_provider=vision_provider,
            file_sem=file_sem,
            ocr_concurrency=ocr_concurrency,
            chunk_chars=chunk_chars,
            stage=stage,
        )
        for pdf_file in pdf_files
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    success = sum(1 for result in results if result is True)
    failed = len(results) - success
    print(f"\n{'=' * 60}")
    print(f"完成! 成功 {success}, 失败 {failed} | 输出: {output_dir}")


# ── CLI ───────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="PDF -> Markdown 无损主版本转换工具 v4")
    parser.add_argument("input", help="PDF文件或文件夹路径")
    parser.add_argument("-o", "--output", default=None, help="输出目录")
    parser.add_argument("-p", "--provider", default="openai",
                        choices=list(PROVIDERS.keys()), help="文本模型")
    parser.add_argument("-v", "--vision", default=None,
                        choices=list(PROVIDERS.keys()), help="视觉模型（扫描页兜底）")
    parser.add_argument("--ocr-concurrency", type=int, default=OCR_CONCURRENCY)
    parser.add_argument("--file-concurrency", type=int, default=FILE_CONCURRENCY)
    parser.add_argument("--chunk-chars", type=int, default=CHUNK_CHAR_LIMIT)
    parser.add_argument("--force", action="store_true", help="强制重新处理")
    parser.add_argument("--stage", default="all",
                        choices=["all", "extract", "merge"],
                        help="运行阶段: all=全部, extract=仅提取, merge=仅组装")
    parser.add_argument("--dry-run", action="store_true", help="预估 OCR 成本，不实际处理")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_file():
        pdf_files = [input_path]
    elif input_path.is_dir():
        pdf_files = sorted(list(input_path.glob("*.pdf")) + list(input_path.glob("*.PDF")))
    else:
        print(f"错误: {input_path} 不存在")
        sys.exit(1)

    if not pdf_files:
        print("未找到PDF文件")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else input_path.parent / "output"

    if args.dry_run:
        dry_run(pdf_files, args.provider, args.vision)
        return

    if args.force and output_dir.exists():
        for pdf_file in pdf_files:
            doc_dir = output_dir / pdf_file.stem
            if doc_dir.exists():
                shutil.rmtree(doc_dir)

    asyncio.run(process_batch(
        pdf_files=pdf_files,
        output_dir=output_dir,
        provider=args.provider,
        vision_provider=args.vision,
        file_concurrency=args.file_concurrency,
        ocr_concurrency=args.ocr_concurrency,
        chunk_chars=args.chunk_chars,
        stage=args.stage,
    ))


if __name__ == "__main__":
    main()
