"""报告生成：页级评估、聚合、多种格式输出。"""

from __future__ import annotations

import csv
import json
from collections import Counter
from difflib import unified_diff
from pathlib import Path
from typing import Any

from eval.baseline import is_scanned_page
from eval.metrics_table import cell_multiset, page_table_metrics
from eval.metrics_text import char_metrics, edit_ratio, word_metrics


def evaluate_page(
    baseline_text: str,
    md_text: str,
    baseline_tables: list[list[list[str]]],
    md_tables: list[list[list[str]]],
    lang: str,
) -> dict[str, Any]:
    """给单页算出所有指标。status：ok / scanned / table_missing / page_missing。"""
    scanned = is_scanned_page(baseline_text)
    table_base_empty = len(baseline_tables) == 0
    table_md_nonempty = len(md_tables) > 0

    status = "ok"
    if scanned:
        status = "scanned"
    elif table_base_empty and table_md_nonempty:
        status = "table_missing"

    if scanned:
        text_m: dict[str, Any] = {"skipped": True, "reason": "scanned"}
        edit: float | None = None
    else:
        text_m = char_metrics(baseline_text, md_text)
        edit = edit_ratio(baseline_text, md_text)

    table_m = page_table_metrics(md_tables=md_tables, baseline_tables=baseline_tables)

    result: dict[str, Any] = {
        "status": status,
        "baseline_text_len": len(baseline_text),
        "md_text_len": len(md_text),
        "text": text_m,
        "text_edit_ratio": edit,
        "baseline_tables": len(baseline_tables),
        "md_tables": len(md_tables),
        "table": table_m,
    }
    if lang == "en" and not scanned:
        result["word"] = word_metrics(baseline_text, md_text)
    return result
