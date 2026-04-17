"""pdf2md 召回率 + 字段级准确率评估。

两套指标：
  - 召回率：len(strip_markdown(MD)) / len(PyMuPDF 基线)
  - 准确率：每份 PDF 按 golden.yaml 里的字段查 MD 是否包含该值，命中率

用法：
  python -m eval.run_eval [--config eval/config.yaml] [--golden eval/golden.yaml]

准确率的 golden.yaml 是手工标注的。没标就只跑召回率。
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from eval.baseline import extract_baseline_text
from eval.md_parser import read_page_md, strip_markdown
from eval.normalize import normalize_text

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("eval")

# 首次用脚本测得的基线（2026-04-17）。后续改 pdf2md 对比这个，Δ < 0 = 退步。
# 昆山峰瑞：字体乱码，PyMuPDF 基线是乱码字符，召回比 >1 是因为 vision 修复后字符变多
# 广州越秀：扫描件，无 PyMuPDF 基线 → None
RECALL_BASELINE: dict[str, float | None] = {
    "天津真格": 1.101,
    "昆山峰瑞": 1.028,
    "广州越秀": None,
    "苏州济峰": 0.994,
    "IDG": 0.999,
}

# 字段级准确率基线（2026-04-17）。金标从 eval/golden.yaml 读取——
# 昆山峰瑞 / 广州越秀的 golden 目前是空，需用户人眼看 PDF 手填后才会有数字。
ACCURACY_BASELINE: dict[str, float | None] = {
    "天津真格": 1.00,
    "昆山峰瑞": None,
    "广州越秀": None,
    "苏州济峰": 1.00,
    "IDG": 1.00,
}


# ──────────────────────────── 召回率 ────────────────────────────


def recall_of_pdf(pdf_path: Path, md_dir: Path) -> tuple[float | None, int, int]:
    """返回 (recall, md_char_len, baseline_char_len)。baseline=0 → recall=None。"""
    base_pages = extract_baseline_text(pdf_path)
    baseline_len = sum(len(t) for t in base_pages)
    md_len = 0
    for i in range(len(base_pages)):
        raw = read_page_md(md_dir, i + 1)
        if raw is None:
            continue
        md_len += len(strip_markdown(raw))
    if baseline_len == 0:
        return None, md_len, 0
    return md_len / baseline_len, md_len, baseline_len


# ──────────────────────────── 准确率（字段级） ────────────────────────────


@dataclass
class FieldCheck:
    field: str
    expected: str
    hit: bool


def _load_md_whole(md_dir: Path, total_pages: int) -> str:
    """把所有 pages/*.md 拼接起来做一次归一化。字段匹配就在这个大字符串上找。"""
    parts: list[str] = []
    for i in range(total_pages):
        raw = read_page_md(md_dir, i + 1)
        if raw is None:
            continue
        parts.append(strip_markdown(raw))
    return " ".join(parts)


def check_fields(md_whole_normalized: str, fields: dict[str, str]) -> list[FieldCheck]:
    """对每个 (字段名, 期望值)，归一化期望值后在 MD 里做 substring 查找。

    跳过占位符（形如 "<填...>"）——这些是模板里的默认值，没真标过。
    """
    checks: list[FieldCheck] = []
    for field, expected in fields.items():
        if not expected or expected.startswith("<填"):
            continue
        needle = normalize_text(str(expected))
        if not needle:
            continue
        checks.append(FieldCheck(
            field=field,
            expected=str(expected),
            hit=needle in md_whole_normalized,
        ))
    return checks


def accuracy_of_pdf(
    md_dir: Path,
    total_pages: int,
    fields: dict[str, str],
) -> tuple[float | None, list[FieldCheck]]:
    """返回 (accuracy, checks)。若没有有效字段（全是 <填...> 占位），accuracy=None。"""
    md_whole = _load_md_whole(md_dir, total_pages)
    checks = check_fields(md_whole, fields)
    if not checks:
        return None, []
    hits = sum(1 for c in checks if c.hit)
    return hits / len(checks), checks


# ──────────────────────────── 入口 ────────────────────────────


def _fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _delta_str(current: float | None, baseline: float | None) -> str:
    if current is None or baseline is None:
        return "(无基线)"
    delta_pp = (current - baseline) * 100
    arrow = "↑" if delta_pp > 0.5 else "↓" if delta_pp < -0.5 else "="
    sign = "+" if delta_pp >= 0 else ""
    return f"{sign}{delta_pp:5.1f}pp {arrow}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="pdf2md 召回率 + 准确率评估")
    parser.add_argument("--config", default="eval/config.yaml")
    parser.add_argument("--golden", default="eval/golden.yaml")
    args = parser.parse_args(argv)

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: config not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = cfg.get("items", [])

    # golden 可选：没有就只跑召回
    golden_path = Path(args.golden)
    golden_by_name: dict[str, dict[str, str]] = {}
    if golden_path.exists():
        gdoc = yaml.safe_load(golden_path.read_text(encoding="utf-8")) or {}
        for g in gdoc.get("items", []):
            golden_by_name[g["name"]] = g.get("fields", {}) or {}
    else:
        print(f"提示：{golden_path} 不存在，只跑召回率。复制 golden.yaml.example 填入标注即可启用准确率。")

    # ── 主表 ──
    print()
    print(
        f"{'PDF':<12s}  {'召回':>8s}  {'召回基线':>10s}  {'Δ召回':>12s}  "
        f"{'准确':>8s}  {'准确基线':>10s}  {'Δ准确':>12s}  {'字段':>6s}"
    )
    print("─" * 100)
    miss_report: list[tuple[str, list[FieldCheck]]] = []

    for it in items:
        name = it["name"]
        pdf = Path(it["pdf"])
        md_dir = Path(it["md_dir"])
        if not pdf.exists():
            print(f"{name:<12s}  (PDF 不存在：{pdf})")
            continue
        if not md_dir.exists():
            print(f"{name:<12s}  (MD 目录不存在：{md_dir})")
            continue

        try:
            recall, md_len, base_len = recall_of_pdf(pdf, md_dir)
        except Exception as e:  # noqa: BLE001
            print(f"{name:<12s}  FAILED: {e}")
            continue

        # 准确率
        acc: float | None = None
        checks: list[FieldCheck] = []
        fields = golden_by_name.get(name, {})
        if fields:
            # total_pages 取 PDF 页数
            import fitz
            with fitz.open(pdf) as doc:
                total_pages = len(doc)
            acc, checks = accuracy_of_pdf(md_dir, total_pages, fields)
            if checks and not all(c.hit for c in checks):
                miss_report.append((name, [c for c in checks if not c.hit]))

        recall_baseline = RECALL_BASELINE.get(name)
        acc_baseline = ACCURACY_BASELINE.get(name)
        n_fields = len(checks)
        print(
            f"{name:<12s}  "
            f"{_fmt_pct(recall):>8s}  {_fmt_pct(recall_baseline):>10s}  {_delta_str(recall, recall_baseline):>12s}  "
            f"{_fmt_pct(acc):>8s}  {_fmt_pct(acc_baseline):>10s}  {_delta_str(acc, acc_baseline):>12s}  "
            f"{n_fields:>6d}"
        )

    # ── 准确率命中详情（只列未命中的字段）──
    if miss_report:
        print()
        print("未命中字段（MD 里没找到期望值）：")
        for name, misses in miss_report:
            print(f"  [{name}]")
            for c in misses:
                print(f"    ✗ {c.field}: 期望包含 {c.expected!r}")

    print()
    print("提示：字段级准确率是基于 golden.yaml 的人工标注，覆盖你肉眼最关心的 15-25 个字段；")
    print("      召回率是字符长度比，作为整体完整性的代理信号。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
