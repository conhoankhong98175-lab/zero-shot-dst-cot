"""
build_step_error_mapping.py
从 ablation_summary_{tag}.json 自动生成论文 §5.3 的步骤↔错误类型映射表。

输出:
    results/step_error_mapping_{tag}.md

每个消融变体的"靶向错误类型"已知:
    ab_no_s1  →  跨域 hallucinated_slot（taxi 域以外的假阳性）
    ab_no_s2  →  wrong_value（值提取错误）
    ab_no_s3  →  missing_slot（隐含漏判）
    ab_no_s4  →  hallucinated_slot（总幻觉，None 验证缺失）

脚本读取 baseline (cot_full) vs 每个 ab_no_s* 的差值，
对靶向方向打"✓ 印证"或"✗ 反预期"标记。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import RESULTS_DIR, ablation_summary_path


TARGETS: dict[str, tuple[str, str]] = {
    # variant -> (主要错误类型, 设计印证文本)
    "ab_no_s1": ("hallucinated_slot", "Step 1 过滤跨域溢出 (Domain Activation Check)"),
    "ab_no_s2": ("wrong_value",       "Step 2 减少值提取错误 (Explicit Search)"),
    "ab_no_s3": ("missing_slot",      "Step 3 补隐含漏判 (Implicit Inference)"),
    "ab_no_s4": ("hallucinated_slot", "Step 4 抑制幻觉 (None Decision)"),
}

ERR_KEYS = ["hallucinated_slot", "missing_slot", "wrong_value"]


def _fmt_delta(d: int) -> str:
    if d > 0:
        return f"+{d} ↑"
    if d < 0:
        return f"{d} ↓"
    return "0 —"


def _verdict(delta_target: int, delta_others: list[int]) -> str:
    """判断"靶向方向 ↑ 而其他方向相对持平"是否成立。"""
    if delta_target <= 0:
        return "✗ 反预期（未上升）"
    # 用相对幅度判断：靶向错误的 Δ 应大于其他错误平均 Δ 的 1.5×
    avg_other = (sum(max(0, d) for d in delta_others) / len(delta_others)) if delta_others else 0
    if avg_other == 0 or delta_target >= 1.5 * avg_other:
        return "✓ 印证设计"
    return "△ 方向正确但选择性偏弱"


def build_mapping(tag: str) -> Path:
    summary_path = ablation_summary_path(tag)
    if not summary_path.exists():
        raise FileNotFoundError(
            f"找不到 {summary_path}。请先运行 ablation_runner 或 compare_ablations。"
        )

    with open(summary_path, encoding="utf-8") as f:
        data = json.load(f)

    baseline = data.get("baseline", "cot_full")
    variants = data.get("variants", {})

    if baseline not in variants:
        raise RuntimeError(f"baseline {baseline!r} 不在变体集合中")

    base_eb = variants[baseline].get("error_breakdown", {})
    base_jga = variants[baseline].get("JGA", 0)
    base_f1  = variants[baseline].get("Slot_F1", 0)

    lines = [
        f"# §5.3 步骤↔错误类型映射表（tag = `{tag}`）\n",
        f"> baseline: `{baseline}`  |  JGA = {base_jga:.2f}%  |  Slot F1 = {base_f1:.2f}%\n",
        "## 主表：消融对错误类型的差异化影响\n",
        "| 消融变体 | Δhallucinated | Δmissing | Δwrong_value | ΔJGA | 靶向错误 | 印证判定 |",
        "|---|---|---|---|---|---|---|",
    ]

    for variant, (target_err, design_text) in TARGETS.items():
        if variant not in variants:
            lines.append(
                f"| `{variant}` | — | — | — | — | {target_err} | (缺失) |"
            )
            continue
        eb = variants[variant].get("error_breakdown", {})
        deltas = {k: eb.get(k, 0) - base_eb.get(k, 0) for k in ERR_KEYS}
        d_jga  = variants[variant].get("JGA", 0) - base_jga

        d_target = deltas[target_err]
        d_others = [v for k, v in deltas.items() if k != target_err]
        verdict = _verdict(d_target, d_others)

        lines.append(
            f"| `{variant}` "
            f"| {_fmt_delta(deltas['hallucinated_slot'])} "
            f"| {_fmt_delta(deltas['missing_slot'])} "
            f"| {_fmt_delta(deltas['wrong_value'])} "
            f"| {d_jga:+.2f}% "
            f"| `{target_err}` "
            f"| {verdict} |"
        )

    # ── ab_no_s1 专项：domain_error_breakdown 看 taxi 域是否单独飙升 ──
    lines.append("\n## ab_no_s1 跨域溢出专项（taxi 域 hallucinated_slot 计数）\n")
    deb_base = variants[baseline].get("domain_error_breakdown", {}) or {}
    deb_s1   = variants.get("ab_no_s1", {}).get("domain_error_breakdown", {}) or {}
    domains  = ["hotel", "restaurant", "attraction", "taxi", "train"]
    if deb_base and deb_s1:
        lines.append("| 域 | cot_full | ab_no_s1 | Δ |")
        lines.append("|---|---|---|---|")
        for d in domains:
            b = deb_base.get(d, {}).get("hallucinated_slot", 0)
            a = deb_s1.get(d, {}).get("hallucinated_slot", 0)
            lines.append(f"| {d} | {b} | {a} | {a - b:+d} |")
        lines.append(
            "\n> **解读规则**：Step 1 设计目标是阻止非 taxi 对话回填 taxi 槽位；"
            "去掉 Step 1 后，预期 *非 taxi 域* 的 hallucinated_slot 显著上升。"
        )
    else:
        lines.append("(domain_error_breakdown 数据缺失)")

    lines.append("\n## 解读模板（论文写作用）\n")
    lines.append(
        "- **设计精准性**：每个消融只在其靶向错误维度上显著恶化，其他维度变化幅度小，"
        "说明四步设计的选择性 (selectivity) 成立。\n"
        "- **若某变体显示 '✗ 反预期'**：在论文中如实呈现，分析可能原因 "
        "（如 LLM 与 T5 在该错误类型上的天然分布差异，详见 §5.1）。\n"
        "- **若 ab_no_s4 的 Δhallucinated 不显著**：呼应 §5.1 的发现——"
        "GPT-4o-mini 在 standard 下幻觉本就少，Step 4 的边际收益相应较小，"
        "Step 4 的价值在于 \"保险\" 而非 \"抑制\"。"
    )

    out_path = RESULTS_DIR / f"step_error_mapping_{tag}.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✅ 步骤映射表已写入: {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="taxi_full",
                    help="ablation_summary 标签，默认 taxi_full")
    args = ap.parse_args()
    build_mapping(args.tag)
