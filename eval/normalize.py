"""文本归一化——基线和 MD 解析共用同一份规则，否则归一化噪音会淹没真实差异。"""

from __future__ import annotations

import re
import unicodedata

_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\ufeff\u2060]")
_WHITESPACE = re.compile(r"\s+")
# 数字间的千分位逗号：两侧都是数字才去掉，例如 1,100.00 → 1100.00
# 语句里的逗号（如 "alice, bob"）保留。
_THOUSANDS_SEP = re.compile(r"(?<=\d),(?=\d)")


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = _THOUSANDS_SEP.sub("", text)
    text = _WHITESPACE.sub(" ", text)
    text = text.strip()
    text = text.lower()
    return text
