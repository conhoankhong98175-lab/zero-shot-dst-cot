"""
evaluator.py  v4  —  最终版
归一化四层：
  L1. 字符级      : 小写/去撇号/去连字符/压缩空格
  L2a. dontcare   : 统一同义词
  L2b. 时间格式   : H:MM / HH:MM am/pm → HH:MM (24h)
  L2c. 数字文字   : three→3, "3 nights"→3
  L2d. yes/no     : free→yes (hotel-internet 等)
  L3.  领域别名   : center→centre, theater→theatre, university→college
"""

import json
import re
from collections import defaultdict
from pathlib import Path

# 五大领域常量（与 prompt_builder.SLOT_SCHEMA 键名保持一致）
_DOMAINS = ["hotel", "restaurant", "attraction", "taxi", "train"]

# ── L2a: dontcare 同义词 ─────────────────────────────────────
_DONTCARE = {
    "don't care", "dont care", "do not care",
    "doesn't matter", "doesnt matter", "does not matter",
    "any", "anything", "either", "no preference",
    "not specified", "flexible", "it doesn't matter",
    "i don't care", "i dont care", "no particular",
    "not important", "no matter", "dontcare",
}

# ── L2c: 数字词 ──────────────────────────────────────────────
_W2N = {
    "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}

# ── L3: 领域别名 ─────────────────────────────────────────────
_ALIAS = {
    "center":     "centre",
    "theater":    "theatre",   # attraction-type
    "university": "college",   # attraction-type (MultiWOZ 使用 college)
    "pool":       "swimming pool",
    "swimmingpool": "swimming pool",
}


def _normalize_time(v: str) -> str:
    """
    统一时间为 HH:MM (24h)。
    处理：
      8:00       → 08:00
      8:30 am    → 08:30
      08:30pm    → 20:30
      by 08:30   → 08:30   (去除 "by"/"after"/"before" 前缀)
      8:00 am    → 08:00
    """
    # 去掉 "by / before / after / around" 等前缀
    v = re.sub(r"^(by|before|after|around|at)\s+", "", v).strip()

    # HH:MM am/pm 或 H:MM am/pm
    m = re.fullmatch(r"(\d{1,2}):(\d{2})\s*(am|pm)", v)
    if m:
        h, mn, period = int(m.group(1)), m.group(2), m.group(3)
        if period == "pm" and h != 12:
            h += 12
        if period == "am" and h == 12:
            h = 0
        return f"{h:02d}:{mn}"

    # HH:MM (24h, 无 am/pm)
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", v)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"

    # Ham / Hpm（无冒号，如 "8am"）
    m = re.fullmatch(r"(\d{1,2})(am|pm)", v)
    if m:
        h, period = int(m.group(1)), m.group(2)
        if period == "pm" and h != 12:
            h += 12
        if period == "am" and h == 12:
            h = 0
        return f"{h:02d}:00"

    return v   # 无法解析，原样返回


def normalize_value(v: str) -> str:
    if not isinstance(v, str):
        return ""

    # ── L1: 字符级 ──────────────────────────────────────────
    v = v.lower().strip()
    v = v.replace("'s", "s")
    v = v.replace("'", "")
    v = v.replace("-", " ")
    v = re.sub(r"\s+", " ", v).strip()

    # ── L2a: dontcare ───────────────────────────────────────
    if v in _DONTCARE:
        return "dontcare"

    # ── L2b: 时间格式 ────────────────────────────────────────
    v = _normalize_time(v)

    # ── L2c: 数字词 & 后缀 ───────────────────────────────────
    if v in _W2N:
        v = _W2N[v]
    v = re.sub(r"^(\d+)\s+(night|nights|people|person|persons|star|stars)$",
               r"\1", v)

    # ── L2d: yes/no 别名 ─────────────────────────────────────
    # hotel-internet/parking: LLM 有时输出 "free" 而非 "yes"
    if v in ("free", "included", "available"):
        v = "yes"

    # ── L3: 领域别名 ─────────────────────────────────────────
    v = _ALIAS.get(v, v)

    return v


def normalize_belief(belief: dict) -> dict:
    skip = {"", "none", "null", "not mentioned", "n/a", "unknown"}
    result = {}
    for k, v in belief.items():
        norm_v = normalize_value(v)
        if norm_v and norm_v not in skip:
            result[k.lower().strip()] = norm_v
    return result


def filter_by_domain(belief: dict, target_domain: str | None) -> dict:
    """限定为目标域的槽位（零样本跨域评测用）。target_domain=None 时不过滤。"""
    if not target_domain:
        return belief
    prefix = f"{target_domain}-"
    return {k: v for k, v in belief.items() if k.startswith(prefix)}


# ── 单条评测 ─────────────────────────────────────────────────
def eval_single(gold: dict, pred: dict, target_domain: str | None = None) -> dict:
    gold_n = filter_by_domain(normalize_belief(gold), target_domain)
    pred_n = filter_by_domain(normalize_belief(pred), target_domain)
    gold_set = set(gold_n.items())
    pred_set = set(pred_n.items())
    tp  = len(gold_set & pred_set)
    fp  = len(pred_set - gold_set)
    fn  = len(gold_set - pred_set)
    jga = (gold_n == pred_n)
    errors = []
    for slot, val in sorted(gold_set - pred_set):
        pred_val = pred_n.get(slot, "<missing>")
        errors.append({
            "type":      "wrong_value" if slot in pred_n else "missing_slot",
            "slot":      slot, "gold": val, "predicted": pred_val,
        })
    for slot, val in sorted(pred_set - gold_set):
        if slot not in gold_n:
            errors.append({
                "type": "hallucinated_slot",
                "slot": slot, "gold": "<none>", "predicted": val,
            })
    # return normalized dicts to avoid re-normalizing in the caller
    return {"jga": jga, "tp": tp, "fp": fp, "fn": fn, "errors": errors,
            "_gold_n": gold_n, "_pred_n": pred_n}


# ── 全量评测 ─────────────────────────────────────────────────
def evaluate(predictions_path: Path,
             target_domain: str | None = None) -> tuple[dict, list]:
    """
    target_domain: 若非 None，仅在该域的槽位上计算 JGA / Slot F1（零样本跨域口径）。
                   pred_belief 中的非目标域条目会被丢弃（不计为 hallucination），
                   gold_belief 同理（再次过滤是冗余防御，对 test_taxi.json 无害）。
    """
    with open(predictions_path, encoding="utf-8") as f:
        preds = json.load(f)

    total = len(preds)
    jga_correct = 0
    total_tp = total_fp = total_fn = 0
    domain_stats       = defaultdict(lambda: {"tp": 0, "total_gold": 0})
    error_type_counts  = defaultdict(int)
    wrong_value_detail = defaultdict(int)
    # 按域×错误类型的细分计数（消融分析用：Step1 主要影响跨域 hallucination）
    domain_error_counts = defaultdict(lambda: defaultdict(int))
    detailed = []

    for item in preds:
        result = eval_single(item["gold_belief"], item["pred_belief"],
                             target_domain=target_domain)
        if result["jga"]:
            jga_correct += 1
        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]

        gold_n = result.pop("_gold_n")
        pred_n = result.pop("_pred_n")
        for slot, val in gold_n.items():
            domain = slot.split("-")[0]
            domain_stats[domain]["total_gold"] += 1
            if pred_n.get(slot) == val:
                domain_stats[domain]["tp"] += 1

        for err in result["errors"]:
            error_type_counts[err["type"]] += 1
            if err["type"] == "wrong_value":
                wrong_value_detail[err["slot"]] += 1
            slot_name = err.get("slot", "")
            if "-" in slot_name:
                d = slot_name.split("-", 1)[0]
                domain_error_counts[d][err["type"]] += 1

        detailed.append({**item, "eval": result})

    jga       = jga_correct / total if total > 0 else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0)

    summary = {
        "total_samples":        total,
        "target_domain":        target_domain or "all",
        "JGA":                  round(jga * 100, 2),
        "Slot_F1":              round(f1 * 100, 2),
        "Slot_Precision":       round(precision * 100, 2),
        "Slot_Recall":          round(recall * 100, 2),
        "error_breakdown":      dict(error_type_counts),
        "wrong_value_by_slot":  dict(
            sorted(wrong_value_detail.items(), key=lambda x: -x[1])[:10]
        ),
        "domain_slot_accuracy": {
            d: round(s["tp"] / s["total_gold"] * 100, 2)
            for d, s in domain_stats.items() if s["total_gold"] > 0
        },
        # 按域 × 错误类型细分：supports S1 ablation narrative
        "domain_error_breakdown": {
            d: {
                "hallucinated_slot": domain_error_counts[d].get("hallucinated_slot", 0),
                "missing_slot":      domain_error_counts[d].get("missing_slot", 0),
                "wrong_value":       domain_error_counts[d].get("wrong_value", 0),
            }
            for d in _DOMAINS
        },
    }
    return summary, detailed


# ── 报告打印 ─────────────────────────────────────────────────
def print_report(summary: dict):
    w = 62
    print("\n" + "=" * w)
    print(f"{'📊 DST 评测报告':^{w}}")
    print("=" * w)
    print(f"  总样本数          : {summary['total_samples']}")
    print(f"  JGA               : {summary['JGA']:.2f}%   ← 论文主指标")
    print(f"  Slot F1           : {summary['Slot_F1']:.2f}%")
    print(f"  Slot Precision    : {summary['Slot_Precision']:.2f}%")
    print(f"  Slot Recall       : {summary['Slot_Recall']:.2f}%")
    print("-" * w)
    print("  错误类型分解:")
    for etype, cnt in sorted(summary["error_breakdown"].items()):
        print(f"    {etype:<30}: {cnt}")
    print("-" * w)
    if summary.get("wrong_value_by_slot"):
        print("  wrong_value 高频槽位 (Top 10):")
        for slot, cnt in summary["wrong_value_by_slot"].items():
            print(f"    {slot:<30}: {cnt}")
        print("-" * w)
    print("  各领域槽位准确率 (slot-level):")
    for domain, acc in sorted(summary["domain_slot_accuracy"].items()):
        bar = "█" * int(acc / 5)
        print(f"    {domain:<16}: {acc:5.1f}%  {bar}")
    deb = summary.get("domain_error_breakdown")
    if deb:
        print("-" * w)
        print("  按域幻觉分解 (hallucinated_slot by domain):")
        for domain in _DOMAINS:
            halluc = deb.get(domain, {}).get("hallucinated_slot", 0)
            if halluc:
                print(f"    {domain:<16}: {halluc}")
    print("=" * w)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="results/pilot_50.json")
    ap.add_argument("--output", default="results/eval_report.json")
    ap.add_argument("--target_domain", default=None,
                    help="若设置（如 taxi），只在该域槽位上算 JGA / Slot F1")
    args = ap.parse_args()
    summary, detailed = evaluate(Path(args.input),
                                 target_domain=args.target_domain)
    print_report(summary)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": detailed}, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 详细报告已保存至: {out}")
