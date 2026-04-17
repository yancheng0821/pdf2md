"""从源 PDF 抽取基线文本（PyMuPDF），供召回率计算用。"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

from eval.normalize import normalize_text

logger = logging.getLogger(__name__)


def extract_baseline_text(pdf_path: Path) -> list[str]:
    """返回每页归一化后的纯文本。基线不完美（CID 乱码/扫描件空文本）是已知的，
    和 MD 做 len 比即可——这是 README 用的"召回比"口径。"""
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    out: list[str] = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            try:
                raw = page.get_text("text")
            except Exception as e:  # noqa: BLE001
                logger.warning("PyMuPDF get_text failed on page %d: %s", page.number + 1, e)
                raw = ""
            out.append(normalize_text(raw))
    return out
