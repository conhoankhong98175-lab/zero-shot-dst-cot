"""
compare_ablations.py
聚合多个 prompt 变体的评测结果，输出论文表格所需的横向对比。

产出：
    results/ablation_summary_{tag}.json   —  结构化数据
    results/ablation_summary_{tag}.md     —  Markdown 表格（直接贴到论文）
    results/ablation_summary_{tag}.csv    —  CSV（喂给 matplotlib / pandas）

关键对比维度：
    - JGA / Slot F1 / P / R           （整体性能）
    - hallucinated_slot               （假阳性，对应 TransferQA 37.54%）
    - missing_slot + wrong_value      （假阴性 + 值错）
    - 各领域 slot accuracy             （观察 taxi 目标域）
    - Δ 相对 cot_full                 （消融影响量）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from prompt_builder import ALL_DOMAINS, VARIANTS
from utils import ablation_summary_path, eval_report_path

# 固定的列顺序，保证论文表格每次重跑结构一致
CORE_METRICS = ["JGA", "Slot_F1", "Slot_Precision", "Slot_Recall"]
ERROR_KEYS   = ["hallucinated_slot", "missing_slot", "wrong_value"]


# ── 数据装载 ─────────────────────────────────────────────────
def _load_summary_from_disk(variant: str, tag: str) -> dict | None:
    """从 eval_report 文件读取单变体 summary；缺失返回 None。"""
    p = eval_report_path(variant, tag)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return None
    summary = data.get("summary", {})
    if summary:
        summary.setdefault("variant", variant)
        summary.setdefault("description", VARIANTS.get(variant, {}).get("description", ""))
    return summary or None


def _collect(tag: str, only: list[str] | None = None) -> dict[str, dict]:
    """扫描 results/ 下所有 eval_*_tag.json，构造 {variant: summary}。"""
    targets = only if only else list(VARIANTS)
    summaries: dict[str, dict] = {}
    for variant in targets:
        s = _load_summary_from_disk(variant, tag)
        if s:
            summaries[variant] = s
    return summaries


# ── 计算派生指标 ─────────────────────────────────────────────
def _enrich(summary: dict) -> dict:
    """为每个 variant 补充复合指标，便于制表。"""
    eb = summary.get("error_breakdown", {})
    total = summary.get("total_samples", 0)
    halluc = eb.get("hallucinated_slot", 0)
    miss   = eb.get("missing_slot", 0)
    wrong  = eb.get("wrong_value", 0)

    # 每 100 轮对话的错误数，便于跨规模对比
    per100 = (lambda x: round(x / total * 100, 2)) if total else (lambda x: 0.0)

    summary["halluc_per_100"]   = per100(halluc)
    summary["missing_per_100"]  = per100(miss)
    summary["wrong_per_100"]    = per100(wrong)
    summary["error_total"]      = halluc + miss + wrong
    return summary


def _delta_vs_baseline(summaries: dict[str, dict],
                       baseline: str = "cot_full") -> dict[str, dict]:
    """对每个 variant 计算相对 baseline 的差值。"""
    if baseline not in summaries:
        return {}
    base = summaries[baseline]
    deltas: dict[str, dict] = {}
    for variant, s in summaries.items():
        if variant == baseline:
            continue
        d: dict = {}
        for m in CORE_METRICS:
            d[f"Δ_{m}"] = round(s.get(m, 0) - base.get(m, 0), 2)
        for k in ERROR_KEYS:
            d[f"Δ_{k}"] = (
                s.get("error_breakdown", {}).get(k, 0)
                - base.get("error_breakdown", {}).get(k, 0)
            )
        deltas[variant] = d
    return deltas


# ── 输出：Markdown 表格 ─────────────────────────────────────
def _render_markdown(summaries: dict[str, dict],
                     deltas: dict[str, dict],
                     baseline: str) -> str:
    ordered_variants = [v for v in VARIANTS if v in summaries]

    lines = ["# Ablation Comparison\n"]

    # --- 主指标表 ---
    lines.append("## Core Metrics\n")
    header = ["Variant", "Description", "N", *CORE_METRICS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for v in ordered_variants:
        s = summaries[v]
        row = [
            v,
            s.get("description", ""),
            str(s.get("total_samples", 0)),
            *(f"{s.get(m, 0):.2f}" for m in CORE_METRICS),
        ]
        lines.append("| " + " | ".join(row) + " |")

    # --- 错误分解 ---
    lines.append("\n## Error Breakdown (per 100 turns)\n")
    err_header = ["Variant", "halluc / 100", "missing / 100", "wrong_value / 100", "total_errors"]
    lines.append("| " + " | ".join(err_header) + " |")
    lines.append("|" + "|".join(["---"] * len(err_header)) + "|")
    for v in ordered_variants:
        s = summaries[v]
        lines.append("| " + " | ".join([
            v,
            f"{s.get('halluc_per_100', 0):.2f}",
            f"{s.get('missing_per_100', 0):.2f}",
            f"{s.get('wrong_per_100', 0):.2f}",
            str(s.get("error_total", 0)),
        ]) + " |")

    # --- Δ 对比 ---
    if deltas and baseline in summaries:
        lines.append(f"\n## Delta vs. `{baseline}` baseline\n")
        dh = ["Variant", *(f"Δ {m}" for m in CORE_METRICS),
              "Δ halluc", "Δ missing", "Δ wrong_value"]
        lines.append("| " + " | ".join(dh) + " |")
        lines.append("|" + "|".join(["---"] * len(dh)) + "|")
        for v in ordered_variants:
            if v == baseline:
                continue
            d = deltas.get(v, {})
            lines.append("| " + " | ".join([
                v,
                *(f"{d.get(f'Δ_{m}', 0):+.2f}" for m in CORE_METRICS),
                f"{d.get('Δ_hallucinated_slot', 0):+d}",
                f"{d.get('Δ_missing_slot', 0):+d}",
                f"{d.get('Δ_wrong_value', 0):+d}",
            ]) + " |")

    # --- 各领域 slot accuracy ---
    lines.append("\n## Per-domain Slot Accuracy (%)\n")
    dh = ["Variant", *ALL_DOMAINS]
    lines.append("| " + " | ".join(dh) + " |")
    lines.append("|" + "|".join(["---"] * len(dh)) + "|")
    for v in ordered_variants:
        s = summaries[v]
        dom = s.get("domain_slot_accuracy", {})
        row = [v] + [f"{dom.get(d, 0):.1f}" if d in dom else "—"
                     for d in ALL_DOMAINS]
        lines.append("| " + " | ".join(row) + " |")

    # --- 各领域 hallucinated_slot 计数（S1 消融的关键证据） ---
    has_domain_err = any(
        s.get("domain_error_breakdown") for s in summaries.values()
    )
    if has_domain_err:
        lines.append("\n## Per-domain Hallucinated Slots (count)\n")
        lines.append("| " + " | ".join(["Variant", *ALL_DOMAINS, "total"]) + " |")
        lines.append("|" + "|".join(["---"] * (len(ALL_DOMAINS) + 2)) + "|")
        for v in ordered_variants:
            deb = summaries[v].get("domain_error_breakdown", {}) or {}
            counts = [deb.get(d, {}).get("hallucinated_slot", 0) for d in ALL_DOMAINS]
            row = [v, *(str(c) for c in counts), str(sum(counts))]
            lines.append("| " + " | ".join(row) + " |")

        if baseline in summaries:
            lines.append(f"\n## Δ Per-domain Hallucination vs. `{baseline}`\n")
            base_deb = summaries[baseline].get("domain_error_breakdown", {}) or {}
            lines.append("| " + " | ".join(["Variant", *ALL_DOMAINS]) + " |")
            lines.append("|" + "|".join(["---"] * (len(ALL_DOMAINS) + 1)) + "|")
            for v in ordered_variants:
                if v == baseline:
                    continue
                deb = summaries[v].get("domain_error_breakdown", {}) or {}
                deltas_d = [
                    deb.get(d, {}).get("hallucinated_slot", 0)
                    - base_deb.get(d, {}).get("hallucinated_slot", 0)
                    for d in ALL_DOMAINS
                ]
                row = [v] + [f"{x:+d}" for x in deltas_d]
                lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


# ── 输出：CSV（宽表）─────────────────────────────────────────
def _render_csv(summaries: dict[str, dict], csv_path: Path) -> None:
    fields = ["variant", "description", "total_samples",
              *CORE_METRICS,
              "hallucinated_slot", "missing_slot", "wrong_value",
              "halluc_per_100", "missing_per_100", "wrong_per_100",
              *(f"domain_{d}" for d in ALL_DOMAINS),
              *(f"halluc_{d}" for d in ALL_DOMAINS)]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for v in VARIANTS:
            if v not in summaries:
                continue
            s = summaries[v]
            eb = s.get("error_breakdown", {})
            dom = s.get("domain_slot_accuracy", {})
            deb = s.get("domain_error_breakdown", {}) or {}
            row = {
                "variant":        v,
                "description":    s.get("description", ""),
                "total_samples":  s.get("total_samples", 0),
                **{m: s.get(m, 0) for m in CORE_METRICS},
                "hallucinated_slot": eb.get("hallucinated_slot", 0),
                "missing_slot":      eb.get("missing_slot", 0),
                "wrong_value":       eb.get("wrong_value", 0),
                "halluc_per_100":    s.get("halluc_per_100", 0),
                "missing_per_100":   s.get("missing_per_100", 0),
                "wrong_per_100":     s.get("wrong_per_100", 0),
                **{f"domain_{d}": dom.get(d, 0) for d in ALL_DOMAINS},
                **{f"halluc_{d}": deb.get(d, {}).get("hallucinated_slot", 0)
                   for d in ALL_DOMAINS},
            }
            w.writerow(row)


# ── 顶层入口 ─────────────────────────────────────────────────
def build_comparison(summaries: dict[str, dict] | None = None,
                     *, tag: str = "full",
                     baseline: str = "cot_full") -> Path:
    """
    生成三份对比产物。若未传入 summaries，则从磁盘读取。
    返回 JSON 文件路径（调用者可据此继续处理）。
    """
    if summaries is None:
        summaries = _collect(tag=tag)
    if not summaries:
        raise RuntimeError(
            f"找不到任何 eval 报告 (tag={tag!r})。请先运行 ablation_runner.py。"
        )

    summaries = {v: _enrich(s) for v, s in summaries.items()}
    deltas = _delta_vs_baseline(summaries, baseline=baseline)

    json_path = ablation_summary_path(tag)
    md_path   = json_path.with_suffix(".md")
    csv_path  = json_path.with_suffix(".csv")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "tag":        tag,
            "baseline":   baseline,
            "variants":   summaries,
            "deltas":     deltas,
        }, f, ensure_ascii=False, indent=2)

    md = _render_markdown(summaries, deltas, baseline)
    md_path.write_text(md, encoding="utf-8")
    _render_csv(summaries, csv_path)

    print("\n" + "=" * 72)
    print("📊 Ablation Comparison")
    print("=" * 72)
    print(md)
    print(f"📁 JSON: {json_path}")
    print(f"📁 MD  : {md_path}")
    print(f"📁 CSV : {csv_path}")
    return json_path


# ── CLI ─────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full",
                    help="读取 eval_{variant}_{tag}.json 中的 tag")
    ap.add_argument("--baseline", default="cot_full",
                    help="作为差值基准的 variant（默认 cot_full）")
    args = ap.parse_args()
    build_comparison(tag=args.tag, baseline=args.baseline)
