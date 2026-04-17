"""从 pdf2md 产出的 pages/*.md 读取并剥 Markdown 标记成纯文本。"""

from __future__ import annotations

import re
from pathlib import Path

from eval.normalize import normalize_text

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`]*`")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_INLINE_HTML_TAG = re.compile(r"<[^>]+>")
_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^>\s?", re.MULTILINE)
_LIST_MARKER = re.compile(r"^(\s*)(?:[-*+]|\d+\.)\s+", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*\*|__|\*|_)(.+?)\1")
_HR = re.compile(r"^-{3,}$", re.MULTILINE)
_TABLE_SEP_LINE = re.compile(r"^\s*\|?[\s:\-|]+\|?\s*$", re.MULTILINE)
_TABLE_PIPE = re.compile(r"\|")


def strip_markdown(md: str) -> str:
    """剥 Markdown 标记，保留单元格文字（和 PyMuPDF 基线口径一致）。
    返回归一化后的纯文本。"""
    if not md:
        return ""
    text = md
    text = _CODE_FENCE.sub(" ", text)
    text = _HTML_COMMENT.sub(" ", text)
    # HTML table 不整块删——单元格文字是真实内容（pdf2md 偶尔输出 HTML table）。
    # 交给下面的 _INLINE_HTML_TAG 一起处理标签剥除。
    text = _IMAGE.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _INLINE_CODE.sub(" ", text)
    text = _HEADING.sub("", text)
    text = _BLOCKQUOTE.sub("", text)
    text = _LIST_MARKER.sub(r"\1", text)
    text = _EMPHASIS.sub(r"\2", text)
    text = _HR.sub(" ", text)
    text = _TABLE_SEP_LINE.sub(" ", text)
    text = _TABLE_PIPE.sub(" ", text)
    text = _INLINE_HTML_TAG.sub(" ", text)
    return normalize_text(text)


def read_page_md(md_dir: Path, page_number: int) -> str | None:
    """读 pages/NNN.md 原始内容。缺失返回 None；utf-8 失败回退 gbk。"""
    path = md_dir / "pages" / f"{page_number:03d}.md"
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gbk")
