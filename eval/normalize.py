"""文本归一化——基线和 MD 解析共用同一份规则，否则归一化噪音会淹没真实差异。"""

from __future__ import annotations

import re
import unicodedata

_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\ufeff\u2060]")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = _WHITESPACE.sub(" ", text)
    text = text.strip()
    text = text.lower()
    return text
