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
