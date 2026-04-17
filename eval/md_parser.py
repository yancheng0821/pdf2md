"""从 pdf2md 产出的 pages/*.md 解析出纯文本和表格，供评估使用。"""

from __future__ import annotations

import re
from pathlib import Path

from eval.normalize import normalize_text

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`]*`")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_TABLE_BLOCK = re.compile(
    r"(?:^\|[^\n]*\|\s*\n)+",
    re.MULTILINE,
)
_HTML_TABLE = re.compile(r"<table\b.*?</table>", re.DOTALL | re.IGNORECASE)
_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^>\s?", re.MULTILINE)
_LIST_MARKER = re.compile(r"^(\s*)(?:[-*+]|\d+\.)\s+", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*\*|__|\*|_)(.+?)\1")
_HR = re.compile(r"^-{3,}$", re.MULTILINE)


def strip_markdown(md: str) -> str:
    """剥离 Markdown 标记返回归一化后的纯文本。表格块整块移除（走 parse_md_tables）。"""
    if not md:
        return ""
    text = md
    text = _CODE_FENCE.sub(" ", text)
    text = _HTML_TABLE.sub(" ", text)
    text = _TABLE_BLOCK.sub(" ", text)
    text = _IMAGE.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _INLINE_CODE.sub(" ", text)
    text = _HEADING.sub("", text)
    text = _BLOCKQUOTE.sub("", text)
    text = _LIST_MARKER.sub(r"\1", text)
    text = _EMPHASIS.sub(r"\2", text)
    text = _HR.sub(" ", text)
    return normalize_text(text)


_MD_TABLE_ROW = re.compile(r"^\|(.+)\|\s*$")
_SEPARATOR_ROW = re.compile(r"^\s*[:\-| ]+\s*$")


def _parse_markdown_tables(md: str) -> list[list[list[str]]]:
    """连续的 `| ... |` 行构成一张表，跳过分隔行 `|---|---|`。"""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in md.splitlines():
        m = _MD_TABLE_ROW.match(line)
        if m:
            inner = m.group(1)
            if _SEPARATOR_ROW.match(inner.replace("|", "")):
                continue
            cells = [c.strip() for c in inner.split("|")]
            current.append(cells)
        else:
            if current:
                tables.append(current)
                current = []
    if current:
        tables.append(current)
    return tables


_HTML_TR = re.compile(r"<tr\b.*?</tr>", re.DOTALL | re.IGNORECASE)
_HTML_TD_TH = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)
_HTML_TAGS = re.compile(r"<[^>]+>")


def _parse_html_tables(md: str) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    for tbl in _HTML_TABLE.findall(md):
        rows: list[list[str]] = []
        for tr in _HTML_TR.findall(tbl):
            cells = [_HTML_TAGS.sub("", c).strip() for c in _HTML_TD_TH.findall(tr)]
            if cells:
                rows.append(cells)
        if rows:
            tables.append(rows)
    return tables


def parse_md_tables(md: str) -> list[list[list[str]]]:
    """从 MD 里抽出所有表格（Markdown pipe 表 + HTML table）。单元格已 strip。"""
    if not md:
        return []
    return _parse_html_tables(md) + _parse_markdown_tables(md)


def read_page_md(md_dir: Path, page_number: int) -> str | None:
    """读取 pages/NNN.md 的原始内容。缺失返回 None；utf-8 失败回退 gbk。"""
    path = md_dir / "pages" / f"{page_number:03d}.md"
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gbk")


def parse_page(md_dir: Path, page_number: int) -> tuple[str, list[list[list[str]]]] | None:
    """返回 (纯文本, 表格列表)；若该页文件不存在，返回 None。"""
    raw = read_page_md(md_dir, page_number)
    if raw is None:
        return None
    return strip_markdown(raw), parse_md_tables(raw)
