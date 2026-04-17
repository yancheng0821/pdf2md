"""pdf2md 转换质量评估入口。

用法：
    python -m eval.run_eval [--config eval/config.yaml]

从 config.yaml 读取 5 份 PDF，逐份跑评估，产物写到 output_dir。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

from eval.baseline import extract_baseline_tables, extract_baseline_text
from eval.md_parser import parse_page
from eval.report import (
    aggregate_pdf,
    evaluate_page,
    write_bad_cases,
    write_pdf_report,
    write_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("eval")


def _load_manifest(md_dir: Path) -> dict[str, Any]:
    mpath = md_dir / "manifest.json"
    if not mpath.exists():
        return {}
    try:
        return json.loads(mpath.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning("manifest.json unreadable in %s: %s", md_dir, e)
        return {}


def _method_for_page(manifest: dict[str, Any], page_num: int) -> str:
    for p in manifest.get("pages", []):
        if p.get("page_number") == page_num:
            return p.get("method", "")
    return ""


def evaluate_pdf(item: dict[str, Any]) -> dict[str, Any]:
    """评估一份 PDF，返回 {name, lang, agg, pages}。"""
    name = item["name"]
    pdf = Path(item["pdf"])
    md_dir = Path(item["md_dir"])
    lang = item.get("lang", "zh")
    logger.info("Evaluating %s (lang=%s)", name, lang)

    baseline_texts = extract_baseline_text(pdf)
    baseline_tables = extract_baseline_tables(pdf)
    total_pages = max(len(baseline_texts), len(baseline_tables))

    manifest = _load_manifest(md_dir)
    pages: list[dict[str, Any]] = []
    for i in range(total_pages):
        page_num = i + 1
        b_text = baseline_texts[i] if i < len(baseline_texts) else ""
        b_tables = baseline_tables[i] if i < len(baseline_tables) else []
        parsed = parse_page(md_dir, page_num)
        if parsed is None:
            pages.append({
                "page": page_num,
                "status": "page_missing",
                "baseline_text_len": len(b_text),
                "md_text_len": 0,
                "text": {"skipped": True, "reason": "page_missing"},
                "text_edit_ratio": None,
                "baseline_tables": len(b_tables),
                "md_tables": 0,
                "table": {"skipped": True, "reason": "page_missing"},
                "md_method": _method_for_page(manifest, page_num),
            })
            continue
        md_text, md_tables = parsed
        page_result = evaluate_page(
            baseline_text=b_text,
            md_text=md_text,
            baseline_tables=b_tables,
            md_tables=md_tables,
            lang=lang,
        )
        page_result["page"] = page_num
        page_result["md_method"] = _method_for_page(manifest, page_num)
        page_result["_baseline_text"] = b_text
        page_result["_md_text"] = md_text
        page_result["_baseline_tables"] = b_tables
        page_result["_md_tables"] = md_tables
        pages.append(page_result)

    agg = aggregate_pdf(pages, lang=lang)
    logger.info(
        "  %s: pages=%d scanned=%d text_F1=%.3f table_F1=%.3f",
        name, agg["pages_total"], agg["pages_scanned"], agg["text_F1"], agg["table_F1"],
    )
    return {"name": name, "lang": lang, "agg": agg, "pages": pages}


def run_evaluation(items: list[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    """主流程：逐份评估，产出所有报告文件。某份挂掉不影响其他。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for item in items:
        try:
            result = evaluate_pdf(item)
            write_pdf_report(result, output_dir)
            write_bad_cases(result, output_dir)
            results.append(result)
        except Exception as e:  # noqa: BLE001
            logger.error("FAILED %s: %s", item.get("name"), e)
            logger.error(traceback.format_exc())
            results.append({
                "name": item.get("name", "unknown"),
                "lang": item.get("lang", "zh"),
                "agg": {
                    "lang": item.get("lang", "zh"),
                    "pages_total": 0, "pages_scanned": 0,
                    "pages_table_missing": 0, "pages_missing": 0,
                    "text_P": 0.0, "text_R": 0.0, "text_F1": 0.0,
                    "text_edit_ratio": 0.0,
                    "table_P": 0.0, "table_R": 0.0, "table_F1": 0.0,
                    "tables_total": 0, "tables_matched": 0,
                    "text_P_word": None, "text_R_word": None, "text_F1_word": None,
                    "_failed": str(e),
                },
                "pages": [],
            })
    write_summary(results, output_dir)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="pdf2md 转换质量评估")
    parser.add_argument("--config", default="eval/config.yaml")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error("Config not found: %s", cfg_path)
        return 2
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    items = cfg.get("items", [])
    output_dir = Path(cfg.get("output_dir", "eval_output"))
    missing = []
    for it in items:
        if not Path(it["pdf"]).exists():
            missing.append(f"PDF missing: {it['pdf']}")
        if not Path(it["md_dir"]).exists():
            missing.append(f"md_dir missing: {it['md_dir']}")
    if missing:
        logger.error("Fail-fast: config references missing paths:\n%s", "\n".join(missing))
        return 3

    run_evaluation(items, output_dir)
    logger.info("Done. Reports at %s", output_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
