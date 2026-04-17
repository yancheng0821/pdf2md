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
    result["_text_tp"] = text_m.get("_tp", 0)
    result["_text_fp"] = text_m.get("_fp", 0)
    result["_text_fn"] = text_m.get("_fn", 0)
    return result


def aggregate_pdf(pages: list[dict[str, Any]], lang: str) -> dict[str, Any]:
    """对一份 PDF 的页级结果做 micro-average 聚合。"""
    pages_total = len(pages)
    pages_scanned = sum(1 for p in pages if p["status"] == "scanned")
    pages_table_missing = sum(1 for p in pages if p["status"] == "table_missing")
    pages_missing = sum(1 for p in pages if p["status"] == "page_missing")

    text_tp = sum(p.get("_text_tp", 0) for p in pages if not p["text"].get("skipped"))
    text_fp = sum(p.get("_text_fp", 0) for p in pages if not p["text"].get("skipped"))
    text_fn = sum(p.get("_text_fn", 0) for p in pages if not p["text"].get("skipped"))
    text_P = text_tp / (text_tp + text_fp) if (text_tp + text_fp) > 0 else 0.0
    text_R = text_tp / (text_tp + text_fn) if (text_tp + text_fn) > 0 else 0.0
    text_F1 = 2 * text_P * text_R / (text_P + text_R) if (text_P + text_R) > 0 else 0.0

    tp_t = sum(p["table"].get("tp", 0) for p in pages if not p["table"].get("skipped"))
    fp_t = sum(p["table"].get("fp", 0) for p in pages if not p["table"].get("skipped"))
    fn_t = sum(p["table"].get("fn", 0) for p in pages if not p["table"].get("skipped"))
    table_P = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0.0
    table_R = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
    table_F1 = 2 * table_P * table_R / (table_P + table_R) if (table_P + table_R) > 0 else 0.0

    tables_total = sum(p["table"].get("tables_total_baseline", 0) for p in pages)
    tables_matched = sum(p["table"].get("tables_matched", 0) for p in pages)

    edit_values = [p["text_edit_ratio"] for p in pages if p.get("text_edit_ratio") is not None]
    text_edit_ratio = sum(edit_values) / len(edit_values) if edit_values else 0.0

    agg: dict[str, Any] = {
        "lang": lang,
        "pages_total": pages_total,
        "pages_scanned": pages_scanned,
        "pages_table_missing": pages_table_missing,
        "pages_missing": pages_missing,
        "text_P": text_P,
        "text_R": text_R,
        "text_F1": text_F1,
        "text_edit_ratio": text_edit_ratio,
        "table_P": table_P,
        "table_R": table_R,
        "table_F1": table_F1,
        "tables_total": tables_total,
        "tables_matched": tables_matched,
    }

    if lang == "en":
        w_tp = sum(p.get("word", {}).get("_tp", 0) for p in pages
                   if p.get("word") and not p["word"].get("skipped"))
        w_fp = sum(p.get("word", {}).get("_fp", 0) for p in pages
                   if p.get("word") and not p["word"].get("skipped"))
        w_fn = sum(p.get("word", {}).get("_fn", 0) for p in pages
                   if p.get("word") and not p["word"].get("skipped"))
        agg["text_P_word"] = w_tp / (w_tp + w_fp) if (w_tp + w_fp) > 0 else 0.0
        agg["text_R_word"] = w_tp / (w_tp + w_fn) if (w_tp + w_fn) > 0 else 0.0
        agg["text_F1_word"] = (
            2 * agg["text_P_word"] * agg["text_R_word"] / (agg["text_P_word"] + agg["text_R_word"])
            if (agg["text_P_word"] + agg["text_R_word"]) > 0 else 0.0
        )
    else:
        agg["text_P_word"] = agg["text_R_word"] = agg["text_F1_word"] = None

    return agg
