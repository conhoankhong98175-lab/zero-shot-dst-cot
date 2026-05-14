"""
select_case_studies.py
从全量预测中自动筛选 §5.5 所需的 4 类典型案例。

类别:
    A. CoT 抑制幻觉:  standard 填了一个值 / cot_full 正确输出 none
    B. CoT 修复漏判:  standard 缺一个 slot / cot_full 正确填补
    C. 虚假推理侥幸:  faithfulness=False 且 jga=True  (依赖 T3.2 标注 csv)
    D. 跨域溢出:       ab_no_s1 填了 taxi 槽 / cot_full 正确输出 none

输出:
    results/case_studies_{tag}.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from utils import RESULTS_DIR, TEST_TAXI_PATH, prediction_path
from evaluator import normalize_belief, filter_by_domain


def _load_preds(variant: str, tag: str) -> dict:
    """Return {(dial_id, turn_id): pred_record}."""
    p = prediction_path(variant, tag)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return {(d["dial_id"], d["turn_id"]): d for d in data}


def _load_history_index() -> dict:
    with open(TEST_TAXI_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {(d["dial_id"], d["turn_id"]): d for d in data}


def _format_history(history: list) -> str:
    out = []
    t = 0
    for role, text in history:
        if role == "user":
            t += 1
            out.append(f"[turn {t}] User: {text}")
        else:
            out.append(f"           System: {text}")
    return "\n".join(out)


def _taxi_dict(belief: dict) -> dict:
    """归一化 + 仅 taxi 域 + 剔除空/none。"""
    return filter_by_domain(normalize_belief(belief), "taxi")


def find_class_A(std_preds: dict, cot_preds: dict, max_n: int = 3) -> list:
    """A. standard 在 taxi 槽填值 / cot_full 正确 none / gold 不含该 slot"""
    hits = []
    for key, cot in cot_preds.items():
        std = std_preds.get(key)
        if not std:
            continue
        gold = _taxi_dict(cot["gold_belief"])
        cot_p = _taxi_dict(cot["pred_belief"])
        std_p = _taxi_dict(std["pred_belief"])
        # standard 假阳性的 taxi 槽
        halluc = {s: v for s, v in std_p.items() if s not in gold}
        if not halluc:
            continue
        # cot_full 在这些槽上 = none
        cot_corrects = [s for s in halluc if s not in cot_p]
        if cot_corrects:
            hits.append({
                "key": key, "evidence_slots": cot_corrects,
                "gold": gold, "std_pred": std_p, "cot_pred": cot_p,
                "cot_reasoning": cot.get("reasoning", ""),
            })
            if len(hits) >= max_n:
                break
    return hits


def find_class_B(std_preds: dict, cot_preds: dict, max_n: int = 3) -> list:
    """B. standard 缺槽 / cot_full 正确填 / gold 含此槽"""
    hits = []
    for key, cot in cot_preds.items():
        std = std_preds.get(key)
        if not std:
            continue
        gold = _taxi_dict(cot["gold_belief"])
        cot_p = _taxi_dict(cot["pred_belief"])
        std_p = _taxi_dict(std["pred_belief"])
        # gold 中存在、standard 漏掉的 slot
        std_missing = {s: v for s, v in gold.items() if s not in std_p}
        if not std_missing:
            continue
        # cot_full 在这些槽上填对了
        cot_correct = [s for s, v in std_missing.items() if cot_p.get(s) == v]
        if cot_correct:
            hits.append({
                "key": key, "evidence_slots": cot_correct,
                "gold": gold, "std_pred": std_p, "cot_pred": cot_p,
                "cot_reasoning": cot.get("reasoning", ""),
            })
            if len(hits) >= max_n:
                break
    return hits


def find_class_D(s1_preds: dict, cot_preds: dict, max_n: int = 3) -> list:
    """D. ab_no_s1 假阳性 taxi 槽 / cot_full 正确 none / gold 不含该槽"""
    hits = []
    for key, cot in cot_preds.items():
        s1 = s1_preds.get(key)
        if not s1:
            continue
        gold = _taxi_dict(cot["gold_belief"])
        cot_p = _taxi_dict(cot["pred_belief"])
        s1_p = _taxi_dict(s1["pred_belief"])
        halluc = {s: v for s, v in s1_p.items() if s not in gold and s not in cot_p}
        if halluc:
            hits.append({
                "key": key, "evidence_slots": list(halluc),
                "gold": gold, "s1_pred": s1_p, "cot_pred": cot_p,
                "cot_reasoning": cot.get("reasoning", ""),
            })
            if len(hits) >= max_n:
                break
    return hits


def find_class_C(annotation_csv: Path, cot_preds: dict, max_n: int = 3) -> list:
    """C. faithfulness=False 且 jga=True. 依赖 T3.2 csv. 缺失返回空。"""
    if not annotation_csv.exists():
        return []
    df = pd.read_csv(annotation_csv, encoding="utf-8-sig")
    df["reasoning_faithful"] = (
        df["reasoning_faithful"].astype(str).str.upper().str.strip()
    )
    unfaithful = df[df["reasoning_faithful"] == "FALSE"]
    hits = []
    for _, r in unfaithful.head(max_n).iterrows():
        key = (r["dial_id"], int(r["turn_id"]))
        cot = cot_preds.get(key)
        if not cot:
            continue
        hits.append({
            "key": key, "note": r.get("note", ""),
            "gold": _taxi_dict(cot["gold_belief"]),
            "cot_pred": _taxi_dict(cot["pred_belief"]),
            "cot_reasoning": cot.get("reasoning", ""),
        })
    return hits


# ── Markdown 渲染 ───────────────────────────────────────────
def _render_case_block(title: str, h: dict, hist_idx: dict,
                       *, extra_pred_key: str | None = None) -> str:
    sample = hist_idx.get(h["key"], {})
    history = _format_history(sample.get("history", []))
    out = [f"### {title}\n",
           f"- `dial_id` = `{h['key'][0]}`  |  `turn_id` = {h['key'][1]}",
           f"- 关键证据槽位: `{h.get('evidence_slots', [])}`\n",
           "**对话历史**\n",
           "```",
           history,
           "```\n",
           "**Belief 对比**\n",
           f"- gold:       `{json.dumps(h['gold'], ensure_ascii=False)}`",
           f"- cot_full:   `{json.dumps(h['cot_pred'], ensure_ascii=False)}`"]
    if extra_pred_key and extra_pred_key in h:
        out.append(f"- {extra_pred_key.replace('_pred', '')}:   `{json.dumps(h[extra_pred_key], ensure_ascii=False)}`")
    reasoning = (h.get("cot_reasoning") or "").strip()
    if reasoning:
        # 限制长度避免文档失控
        snippet = reasoning if len(reasoning) <= 1200 else reasoning[:1200] + "\n...[truncated]"
        out += ["\n**CoT 推理链摘录**", "```", snippet, "```"]
    if h.get("note"):
        out.append(f"\n**人工标注备注**: {h['note']}")
    return "\n".join(out)


def build_case_studies(tag: str, annotation_csv: Path | None = None) -> Path:
    std_preds = _load_preds("standard", tag)
    cot_preds = _load_preds("cot_full", tag)
    s1_preds  = _load_preds("ab_no_s1", tag)
    hist_idx  = _load_history_index()

    for label, d in [("standard", std_preds), ("cot_full", cot_preds), ("ab_no_s1", s1_preds)]:
        if not d:
            print(f"⚠️  缺失 {label} 预测 (tag={tag})；该类案例将跳过")

    class_a = find_class_A(std_preds, cot_preds) if std_preds and cot_preds else []
    class_b = find_class_B(std_preds, cot_preds) if std_preds and cot_preds else []
    class_d = find_class_D(s1_preds, cot_preds)  if s1_preds  and cot_preds else []
    class_c = find_class_C(annotation_csv, cot_preds) if annotation_csv else []

    lines = [
        f"# §5.5 错误案例研究（tag = `{tag}`）\n",
        "本节展示 4 类典型对话片段，对应消融实验的定量结论。",
        "案例由 `src/select_case_studies.py` 从全量预测中自动筛选。\n",
        "---\n",
    ]

    blocks = [
        ("A 类 — CoT 抑制 standard 幻觉", class_a, "std_pred"),
        ("B 类 — CoT 修复 standard 漏判", class_b, "std_pred"),
        ("C 类 — 虚假推理侥幸正确",       class_c, None),
        ("D 类 — Step 1 阻止跨域溢出",    class_d, "s1_pred"),
    ]
    for header, hits, pred_key in blocks:
        lines.append(f"## {header}")
        if not hits:
            lines.append("\n_（无满足条件的案例 — 可能因当前实验尚未跑全或 T3.2 未标注）_\n")
            continue
        for i, h in enumerate(hits, start=1):
            lines.append(_render_case_block(
                f"Case {header[0]}.{i}", h, hist_idx,
                extra_pred_key=pred_key,
            ))
            lines.append("\n---\n")

    out = RESULTS_DIR / f"case_studies_{tag}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ 案例研究已写入: {out}")
    print(f"   A={len(class_a)}  B={len(class_b)}  C={len(class_c)}  D={len(class_d)}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="taxi_full")
    ap.add_argument("--annotation_csv", default=None,
                    help="T3.2 输出的 faithfulness_annotation.csv，启用 C 类")
    args = ap.parse_args()
    csv_path = Path(args.annotation_csv) if args.annotation_csv else None
    build_case_studies(args.tag, csv_path)
