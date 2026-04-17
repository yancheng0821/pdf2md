"""从源 PDF 抽取基线数据（文本 + 表格）。"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

from eval.normalize import normalize_text

logger = logging.getLogger(__name__)

SCANNED_THRESHOLD = 20


def extract_baseline_text(pdf_path: Path) -> list[str]:
    """返回每页归一化后的纯文本，长度 == PDF 页数。"""
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


def is_scanned_page(normalized_text: str) -> bool:
    """归一化后文本长度 < 阈值 → 扫描页（基线不可信）。"""
    return len(normalized_text.strip()) < SCANNED_THRESHOLD


def extract_baseline_tables(pdf_path: Path) -> list[list[list[list[str]]]]:
    """返回每页的表格列表。

    外层：页 → 中层：页内的表 → 内层：行 → 单元格。
    某页 pdfplumber 抛异常时，该页对应 `[]`（空 list）并记 warning。
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    out: list[list[list[list[str]]]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                raw_tables = page.extract_tables() or []
            except Exception as e:  # noqa: BLE001
                logger.warning("pdfplumber extract_tables failed on page %d: %s", i + 1, e)
                raw_tables = []
            page_tables: list[list[list[str]]] = []
            for t in raw_tables:
                normalized = [
                    [("" if cell is None else str(cell).strip()) for cell in row]
                    for row in t
                ]
                page_tables.append(normalized)
            out.append(page_tables)
    return out
