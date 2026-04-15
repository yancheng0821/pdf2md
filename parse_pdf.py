"""
PDF基金报告解析工具 v2
- 阶段1（提取）：逐页分类 → 文本直取 / OCR并发 → 页级原文持久化
- 阶段2（汇总）：读页级原文 → 分块并发 → 结构化MD输出
- 两阶段解耦：换prompt/换模型只需重跑阶段2
- 质量校验：关键财务指标交叉验证
- dry-run：预估token和成本
"""

import asyncio
import base64
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import fitz  # pymupdf
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).parent / ".env")

for key in ["all_proxy", "ALL_PROXY"]:
    os.environ.pop(key, None)

# ── 配置 ──────────────────────────────────────────────

PROVIDERS = {
    "xiaomi": {
        "api_key": os.getenv("XIAOMI_API_KEY"),
        "base_url": os.getenv("XIAOMI_BASE_URL"),
        "model": os.getenv("XIAOMI_MODEL"),
    },
    "openai": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "model": os.getenv("OPENAI_MODEL"),
    },
    "4o-mini": {
        "api_key": os.getenv("OPENAI_4O_MINI_API_KEY"),
        "base_url": os.getenv("OPENAI_4O_MINI_BASE_URL"),
        "model": os.getenv("OPENAI_4O_MINI_MODEL"),
    },
    "4.1-mini": {
        "api_key": os.getenv("OPENAI_41_MINI_API_KEY"),
        "base_url": os.getenv("OPENAI_41_MINI_BASE_URL"),
        "model": os.getenv("OPENAI_41_MINI_MODEL"),
    },
    "4.1-nano": {
        "api_key": os.getenv("OPENAI_41_NANO_API_KEY"),
        "base_url": os.getenv("OPENAI_41_NANO_BASE_URL"),
        "model": os.getenv("OPENAI_41_NANO_MODEL"),
    },
    "5.4": {
        "api_key": os.getenv("OPENAI_54_API_KEY"),
        "base_url": os.getenv("OPENAI_54_BASE_URL"),
        "model": os.getenv("OPENAI_54_MODEL"),
    },
    "5.4-mini": {
        "api_key": os.getenv("OPENAI_54_MINI_API_KEY"),
        "base_url": os.getenv("OPENAI_54_MINI_BASE_URL"),
        "model": os.getenv("OPENAI_54_MINI_MODEL"),
    },
    "5.4-nano": {
        "api_key": os.getenv("OPENAI_54_NANO_API_KEY"),
        "base_url": os.getenv("OPENAI_54_NANO_BASE_URL"),
        "model": os.getenv("OPENAI_54_NANO_MODEL"),
    },
}

TEXT_THRESHOLD = 30
IMAGE_DPI = 150
IMAGE_FORMAT = "jpeg"
IMAGE_QUALITY = 85
MAX_RETRIES = 2
OCR_CONCURRENCY = 8
FILE_CONCURRENCY = 3
CHUNK_CHAR_LIMIT = 24_000

# 各模型每百万token价格（美元），用于dry-run估算
TOKEN_PRICES = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-5.4": {"input": 2.50, "output": 10.00},
    "gpt-5.4-mini": {"input": 0.40, "output": 1.60},
    "gpt-5.4-nano": {"input": 0.10, "output": 0.40},
}


def get_async_client(provider: str) -> tuple[AsyncOpenAI, str]:
    cfg = PROVIDERS[provider]
    client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    return client, cfg["model"]


# ── 文本质量检测 ──────────────────────────────────────

def is_garbled(text: str) -> bool:
    if len(text) < 30:
        return False
    cjk_chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
    if len(cjk_chars) < 10:
        return False
    freq = Counter(cjk_chars)
    top_count = sum(c for _, c in freq.most_common(8))
    top_ratio = top_count / len(cjk_chars)
    digit_ratio = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
    return top_ratio > 0.4 and digit_ratio < 0.02


# ── 阶段1：提取 ──────────────────────────────────────

def classify_pages(pdf_path: str) -> tuple[dict[int, str], dict[int, str], int]:
    """
    逐页分类，返回:
    - text_pages: {page_idx: text}
    - image_pages: {page_idx: base64_jpeg}
    - total_pages: 总页数
    """
    doc = fitz.open(pdf_path)
    text_pages = {}
    image_pages = {}
    total = len(doc)

    for i in range(total):
        page = doc[i]
        text = page.get_text().strip()
        if len(text) >= TEXT_THRESHOLD and not is_garbled(text):
            text_pages[i] = text
        else:
            pix = page.get_pixmap(dpi=IMAGE_DPI)
            img_bytes = pix.tobytes(IMAGE_FORMAT, jpg_quality=IMAGE_QUALITY)
            image_pages[i] = base64.b64encode(img_bytes).decode()

    doc.close()
    return text_pages, image_pages, total


# ── LLM 调用 ─────────────────────────────────────────

OCR_PROMPT = """请将这个扫描页面中的所有文字和表格内容提取出来，用纯文本输出。
要求：
1. 所有数字必须精确，不要猜测或四舍五入
2. 表格用 markdown 表格格式还原
3. 保持原文的段落和层次结构
4. 如果是封面、目录、免责声明等无数据页，输出"[无关键数据]"即可
直接输出内容，不要加解释。"""

CHUNK_SUMMARY_PROMPT = """你在做基金报告的分块抽取。下面只是一部分页面，不是整份报告。

请只基于当前分块中实际出现的内容，提取关键事实，使用 Markdown 输出。
要求：
1. 不要补全未出现的信息，不要猜测
2. 所有数字、日期、单位保持原文精确值
3. 表格数据必须完整保留，用 markdown 表格输出，不要省略任何行
4. 如果同一字段有多个版本，保留原值并注明对应页码
5. 没有信息的栏目写"未提及"

输出结构：
## 基本信息
## 财务指标
## 投资组合
## 现金分配
## 费用与合规
## 其他重要信息
"""

REDUCE_SUMMARY_PROMPT = """你将收到若干个分块提取结果，请合并为更紧凑的中间摘要。

要求：
1. 合并重复项，但不要丢数字、日期、单位
2. 投资组合和现金分配的表格必须完整保留每一行，不要省略
3. 有冲突的数据并列保留，不要自行裁决
4. 不要生成最终结论，只做信息归并
"""

FINAL_SUMMARY_PROMPT = """你是一个专业的基金报告分析助手。以下是从一份基金报告PDF中提取的内容。
请将这些数据整理成一份结构化的汇总报告，使用Markdown格式。

要求的输出结构：

# [基金名称]

## 一、基金基本信息
- 基金全称、成立日期、注册地址、组织形式、存续期限
- 管理人/GP、托管人等

## 二、主要财务指标
- 认缴出资额、实缴出资额
- 基金资产总额、净资产
- DPI、IRR、TVPI/MOIC 等回报指标

## 三、投资组合概况
用表格展示所有被投企业：
| 企业名称 | 行业 | 投资日期 | 投资金额 | 最新估值/退出金额 | 持股比例 | 回报倍数 | 状态 |

分为：已退出项目、未退出项目。必须列出所有项目，不要省略。

## 四、现金分配记录
| 序号 | 分配时间 | 分配事项 | 本次分配金额 | 累计分配金额 |

## 五、其他重要信息
- 基金费用、管理费、业绩报酬等
- 风险提示、合规信息等
- 任何其他值得注意的数据

注意：
- 所有数字保持原文精确值
- 如果某些信息在报告中未提及，标注"未提及"
- 金额单位统一标注清楚（万元/元/美元等）
"""


async def call_llm(client: AsyncOpenAI, model: str, system: str,
                   content_parts: list, retry: int = 0) -> str:
    use_new_param = any(model.startswith(p) for p in ("gpt-5", "o1", "o3", "o4"))
    token_param = {"max_completion_tokens": 16384} if use_new_param else {"max_tokens": 16384}
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content_parts},
            ],
            temperature=0.1,
            **token_param,
        )
        return resp.choices[0].message.content or ""
    except Exception:
        if retry < MAX_RETRIES:
            await asyncio.sleep(2 ** retry)
            return await call_llm(client, model, system, content_parts, retry + 1)
        raise


async def ocr_single_page(client: AsyncOpenAI, model: str, page_idx: int,
                          b64_img: str, semaphore: asyncio.Semaphore,
                          progress: dict) -> tuple[int, str]:
    async with semaphore:
        content = [
            {"type": "text", "text": f"第{page_idx + 1}页："},
            {"type": "image_url", "image_url": {
                "url": f"data:image/{IMAGE_FORMAT};base64,{b64_img}",
                "detail": "high",
            }},
        ]
        result = await call_llm(client, model, OCR_PROMPT, content)
        progress["done"] += 1
        print(f"    OCR {progress['done']}/{progress['total']} (第{page_idx + 1}页)")
        return page_idx, result


# ── 阶段1：提取并持久化 ──────────────────────────────

async def stage1_extract(pdf_path: Path, output_dir: Path,
                         provider: str, vision_provider: str | None,
                         ocr_concurrency: int) -> Path | None:
    """
    阶段1：提取全部页面文字，持久化到 .pages.md
    如果 .pages.md 已存在则跳过（复用之前的提取结果）
    返回 pages_path 或 None（失败时）
    """
    pages_path = output_dir / (pdf_path.stem + ".pages.md")

    if pages_path.exists():
        print(f"  [阶段1跳过] {pdf_path.name} (复用已有提取)")
        return pages_path

    text_pages, image_pages, total_pages = classify_pages(str(pdf_path))
    print(f"  共 {total_pages} 页 | 文本 {len(text_pages)} | 扫描 {len(image_pages)}")

    # 并发OCR扫描页
    ocr_results: dict[int, str] = {}
    failed_pages: dict[int, str] = {}

    if image_pages:
        v_client, v_model = get_async_client(vision_provider or provider)
        ocr_sem = asyncio.Semaphore(ocr_concurrency)
        progress = {"done": 0, "total": len(image_pages)}

        tasks = [
            ocr_single_page(v_client, v_model, idx, b64, ocr_sem, progress)
            for idx, b64 in image_pages.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for (page_idx, _b64), result in zip(image_pages.items(), results):
            if isinstance(result, Exception):
                failed_pages[page_idx] = str(result)
                print(f"    OCR失败: 第{page_idx + 1}页 | {result}")
            else:
                ocr_results[result[0]] = result[1]

    if failed_pages:
        err_path = output_dir / (pdf_path.stem + ".errors.txt")
        err_msg = "以下页面提取失败：" + ", ".join(f"第{k+1}页" for k in sorted(failed_pages))
        err_path.write_text(err_msg, encoding="utf-8")
        print(f"  [失败] {err_msg}")
        return None

    # 按页码顺序合并，持久化
    lines = []
    for i in range(total_pages):
        if i in text_pages:
            lines.append(f"<!-- page:{i+1} method:text -->\n{text_pages[i]}")
        elif i in ocr_results:
            lines.append(f"<!-- page:{i+1} method:ocr -->\n{ocr_results[i]}")

    pages_path.write_text("\n\n".join(lines), encoding="utf-8")
    print(f"  阶段1完成: {len(lines)} 页 → {pages_path.name} ({len(''.join(lines))} 字符)")
    return pages_path


# ── 阶段2：汇总 ──────────────────────────────────────

def chunk_text(text: str, max_chars: int = CHUNK_CHAR_LIMIT) -> list[str]:
    """按页标记分块"""
    pages = re.split(r"(?=<!-- page:\d+)", text)
    pages = [p for p in pages if p.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for page in pages:
        plen = len(page)
        if current and current_len + plen > max_chars:
            chunks.append("\n\n".join(current))
            current = [page]
            current_len = plen
        else:
            current.append(page)
            current_len += plen

    if current:
        chunks.append("\n\n".join(current))
    return chunks


async def stage2_summarize(pages_path: Path, output_dir: Path,
                           provider: str, chunk_chars: int) -> Path:
    """
    阶段2：读取 .pages.md → 分块并发提取 → 汇总 → 输出 .md
    """
    out_path = output_dir / (pages_path.stem.replace(".pages", "") + ".md")
    raw_text = pages_path.read_text(encoding="utf-8")

    client, model = get_async_client(provider)
    chunks = chunk_text(raw_text, max_chars=chunk_chars)

    if len(chunks) == 1:
        print(f"  单块直接汇总 ({len(raw_text)} 字符)...")
        summary = await call_llm(client, model, FINAL_SUMMARY_PROMPT,
                                 [{"type": "text", "text": raw_text}])
    else:
        # 并发分块提取
        print(f"  并发抽取 {len(chunks)} 个分块...")

        async def extract_chunk(idx: int, chunk: str) -> str:
            note = await call_llm(client, model, CHUNK_SUMMARY_PROMPT,
                                  [{"type": "text", "text": chunk}])
            print(f"    分块 {idx}/{len(chunks)} 完成")
            return note

        chunk_notes = list(await asyncio.gather(
            *[extract_chunk(i, c) for i, c in enumerate(chunks, 1)]
        ))

        # 只在合并后超限时才归并，否则直接拼接进最终汇总
        merged = "\n\n".join(chunk_notes)
        while len(merged) > chunk_chars and len(chunk_notes) > 1:
            pairs = [chunk_notes[i:i+2] for i in range(0, len(chunk_notes), 2)]
            print(f"  并发归并 {len(pairs)} 组...")

            async def reduce_pair(group: list[str]) -> str:
                if len(group) == 1:
                    return group[0]
                return await call_llm(client, model, REDUCE_SUMMARY_PROMPT,
                                      [{"type": "text", "text": "\n\n---\n\n".join(group)}])

            chunk_notes = list(await asyncio.gather(*[reduce_pair(p) for p in pairs]))
            merged = "\n\n".join(chunk_notes)

        # 最终汇总
        print("  生成最终报告...")
        summary = await call_llm(client, model, FINAL_SUMMARY_PROMPT,
                                 [{"type": "text", "text": chunk_notes[0]}])

    # 质量校验
    warnings = validate_report(summary)
    if warnings:
        summary += "\n\n## 解析质量校验\n"
        for w in warnings:
            summary += f"- ⚠ {w}\n"

    out_path.write_text(summary, encoding="utf-8")
    return out_path


# ── 质量校验 ──────────────────────────────────────────

def _extract_number(text: str, pattern: str) -> float | None:
    """从报告文本中按关键词提取数字"""
    match = re.search(pattern + r"[：:]\s*([\d,]+\.?\d*)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def validate_report(report_text: str) -> list[str]:
    """交叉验证关键财务指标"""
    warnings = []

    committed = _extract_number(report_text, r"认缴出资额")
    paid_in = _extract_number(report_text, r"实缴出资额")
    total_assets = _extract_number(report_text, r"基金资产总额")
    net_assets = _extract_number(report_text, r"净资产")

    if committed and paid_in and paid_in > committed * 1.01:
        warnings.append(f"实缴({paid_in:,.2f})大于认缴({committed:,.2f})，请核实")

    if total_assets and net_assets and net_assets > total_assets * 1.01:
        warnings.append(f"净资产({net_assets:,.2f})大于总资产({total_assets:,.2f})，请核实")

    # DPI交叉验证：DPI = 累计分配 / 实缴
    dpi_match = re.search(r"DPI[：:]\s*([\d.]+)%?", report_text)
    if dpi_match and paid_in:
        dpi_val = float(dpi_match.group(1))
        # 尝试提取累计分配金额
        dist_match = re.search(r"累计分配金额[）)]*\s*\n.*?\|\s*([\d,]+\.?\d*)", report_text)
        if dist_match:
            total_dist = float(dist_match.group(1).replace(",", ""))
            calc_dpi = total_dist / paid_in * 100
            if abs(calc_dpi - dpi_val) > 5:
                warnings.append(
                    f"DPI({dpi_val}%) vs 计算值(累计分配{total_dist:,.0f}/实缴{paid_in:,.0f}={calc_dpi:.1f}%)，偏差>5%，请核实"
                )

    # 检查乱码残留
    garbled_chars = set("睡睢督睤睥睦睧睨")
    garbled_count = sum(1 for c in report_text if c in garbled_chars)
    if garbled_count > 5:
        warnings.append(f"输出中检测到{garbled_count}个疑似乱码字符，部分数据可能不准确")

    return warnings


# ── dry-run ───────────────────────────────────────────

def dry_run(pdf_files: list[Path], provider: str, vision_provider: str | None):
    """预估处理成本，不实际调用API"""
    model = PROVIDERS[provider]["model"]
    v_model = PROVIDERS[vision_provider or provider]["model"]

    total_pages = 0
    total_text = 0
    total_scan = 0

    for f in pdf_files:
        text_pages, image_pages, n_pages = classify_pages(str(f))
        total_pages += n_pages
        total_text += len(text_pages)
        total_scan += len(image_pages)
        text_chars = sum(len(t) for t in text_pages.values())
        print(f"  {f.name}: {n_pages}页 (文本{len(text_pages)}, 扫描{len(image_pages)}, 文字{text_chars}字符)")

    # 估算token
    # 文本页：~1字符≈0.5token(中文)，扫描页：~1000 token/页(图片)
    text_tokens = total_text * 400  # 平均每页400字符 * 0.5
    image_tokens = total_scan * 1100  # 图片token + prompt
    ocr_output_tokens = total_scan * 500  # OCR输出
    summary_input = (text_tokens + ocr_output_tokens)  # 汇总输入
    summary_output = 3000  # 汇总输出

    total_input = text_tokens + image_tokens + summary_input
    total_output = ocr_output_tokens + summary_output

    text_price = TOKEN_PRICES.get(model, {"input": 1.0, "output": 4.0})
    vision_price = TOKEN_PRICES.get(v_model, text_price)

    # OCR成本用视觉模型价格，汇总用文本模型价格
    ocr_cost = (image_tokens * vision_price["input"] + ocr_output_tokens * vision_price["output"]) / 1_000_000
    summary_cost = (summary_input * text_price["input"] + summary_output * text_price["output"]) / 1_000_000
    total_cost = ocr_cost + summary_cost

    print(f"\n{'='*50}")
    print(f"汇总:")
    print(f"  文件: {len(pdf_files)} 个")
    print(f"  总页数: {total_pages} (文本 {total_text}, 扫描 {total_scan})")
    print(f"  预估token: ~{total_input/1000:.0f}K input + ~{total_output/1000:.0f}K output")
    print(f"  文本模型: {model}")
    print(f"  视觉模型: {v_model}")
    print(f"  预估成本: ${total_cost:.3f}")
    print(f"    OCR: ${ocr_cost:.3f}")
    print(f"    汇总: ${summary_cost:.3f}")


# ── 单文件处理 ────────────────────────────────────────

async def process_pdf(pdf_path: Path, output_dir: Path, provider: str,
                      vision_provider: str | None = None,
                      file_sem: asyncio.Semaphore | None = None,
                      ocr_concurrency: int = OCR_CONCURRENCY,
                      chunk_chars: int = CHUNK_CHAR_LIMIT,
                      stage: str = "all") -> bool:
    """处理单个PDF，返回是否成功"""
    pdf_path = Path(pdf_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / (pdf_path.stem + ".md")
    pages_path = output_dir / (pdf_path.stem + ".pages.md")

    # 断点续跑
    if stage == "all" and out_path.exists():
        print(f"  [跳过] {pdf_path.name}")
        return True

    sem = file_sem or asyncio.Semaphore(1)
    async with sem:
        t0 = time.time()
        print(f"\n{'─'*60}")
        print(f"  {pdf_path.name}")

        # 阶段1
        if stage in ("all", "extract"):
            pages_path = await stage1_extract(
                pdf_path, output_dir, provider, vision_provider, ocr_concurrency
            )
            if pages_path is None:
                return False

        # 阶段2
        if stage in ("all", "summarize"):
            if not pages_path.exists():
                print(f"  [错误] 缺少 {pages_path.name}，请先运行 --stage extract")
                return False
            out = await stage2_summarize(pages_path, output_dir, provider, chunk_chars)
            print(f"  → {out.name}")

        elapsed = time.time() - t0
        print(f"  done ({elapsed:.1f}s)")
        return True


# ── 批量处理 ──────────────────────────────────────────

async def process_batch(pdf_files: list[Path], output_dir: Path,
                        provider: str, vision_provider: str | None = None,
                        file_concurrency: int = FILE_CONCURRENCY,
                        ocr_concurrency: int = OCR_CONCURRENCY,
                        chunk_chars: int = CHUNK_CHAR_LIMIT,
                        stage: str = "all"):
    output_dir.mkdir(parents=True, exist_ok=True)
    vp = vision_provider or provider

    print(f"共 {len(pdf_files)} 个PDF | 阶段: {stage}")
    print(f"文本模型: {provider} ({PROVIDERS[provider]['model']})")
    print(f"视觉模型: {vp} ({PROVIDERS[vp]['model']})")
    print(f"并发: 文件×{file_concurrency}, OCR×{ocr_concurrency}")

    file_sem = asyncio.Semaphore(file_concurrency)
    tasks = [
        process_pdf(f, output_dir, provider, vision_provider,
                    file_sem, ocr_concurrency, chunk_chars, stage)
        for f in pdf_files
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    success = sum(1 for r in results if r is True)
    failed = len(results) - success
    print(f"\n{'='*60}")
    print(f"完成! 成功 {success}, 失败 {failed} | 输出: {output_dir}")


# ── CLI ───────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PDF基金报告解析工具 v2")
    parser.add_argument("input", help="PDF文件或文件夹路径")
    parser.add_argument("-o", "--output", default=None, help="输出目录")
    parser.add_argument("-p", "--provider", default="openai",
                        choices=list(PROVIDERS.keys()), help="文本模型")
    parser.add_argument("-v", "--vision", default=None,
                        choices=list(PROVIDERS.keys()), help="视觉模型")
    parser.add_argument("--ocr-concurrency", type=int, default=OCR_CONCURRENCY)
    parser.add_argument("--file-concurrency", type=int, default=FILE_CONCURRENCY)
    parser.add_argument("--chunk-chars", type=int, default=CHUNK_CHAR_LIMIT)
    parser.add_argument("--force", action="store_true", help="强制重新处理")
    parser.add_argument("--stage", default="all",
                        choices=["all", "extract", "summarize"],
                        help="运行阶段: all=全部, extract=仅提取, summarize=仅汇总")
    parser.add_argument("--dry-run", action="store_true", help="预估成本，不实际处理")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_file():
        pdf_files = [input_path]
    elif input_path.is_dir():
        pdf_files = sorted(list(input_path.glob("*.pdf")) + list(input_path.glob("*.PDF")))
    else:
        print(f"错误: {input_path} 不存在")
        sys.exit(1)

    if not pdf_files:
        print("未找到PDF文件")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else input_path.parent / "output"

    if args.dry_run:
        dry_run(pdf_files, args.provider, args.vision)
        return

    if args.force:
        output_dir.mkdir(parents=True, exist_ok=True)
        for f in pdf_files:
            if args.stage in ("all", "extract"):
                for suffix in (".pages.md", ".errors.txt"):
                    t = output_dir / f"{f.stem}{suffix}"
                    if t.exists():
                        t.unlink()
            if args.stage in ("all", "summarize"):
                t = output_dir / f"{f.stem}.md"
                if t.exists():
                    t.unlink()

    asyncio.run(process_batch(
        pdf_files, output_dir, args.provider, args.vision,
        args.file_concurrency, args.ocr_concurrency,
        args.chunk_chars, args.stage,
    ))


if __name__ == "__main__":
    main()
