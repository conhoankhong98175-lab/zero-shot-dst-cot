"""
auto_faithfulness_check.py
基于规则的推理忠实性自动初标，供 T3.2 节省人工。

## 设计原则

人工标注的核心动作是「在 reasoning 中找到每一次 turn 引用 → 核对该 turn 是否真的
包含被引用的内容」。本脚本用确定性规则模拟此过程，输出与人工标注同 schema 的 CSV。

判定流程（对每个样本）：
  1. 解析 reasoning, 抽取所有形如 `[turn N]` / `turn N` 的引用位置
  2. 对每个引用, 截取其右侧 ~120 字符内的"声明文本"
  3. 从声明文本提取实质性 token (英文名词 / 数字 / 时间 / 价格/位置词等)
  4. 从对话 history 取出该 turn 的 user 文本, 做同样 token 抽取 + 同义词归一化
  5. 计算声明 token 在 turn token 中的覆盖率 (overlap_ratio)
  6. 若所有引用 overlap >= THRESHOLD → faithful = True
     存在引用 overlap < THRESHOLD → faithful = False
  7. reasoning 为空 / 无 turn 引用 / token 抽取后 < 2 token → 标 inconclusive

## 误差方向（必须在论文中坦诚说明）

- **False Negative**（语义对但词面不重叠）: 模型说"depart at nine"而 turn 是"leave at 9 am",
  词面会 mismatch. 通过同义词表 + 数字词归一化部分缓解, 但不能根治.
- **False Positive**（语义错但词面巧合）: 模型 fabricate"user wants 7pm taxi"恰好对话里有"7"
  (但实际指 7 个人), 规则会判 faithful. 这种情况靠 token 多元 (动词+名词组合) 减少.

人类 review 时, 重点核对脚本判 True 的样本是否真的逐字匹配; 判 False 的样本可
直接信任或加 note 说明 "词面失配但语义对".

## 输出
  results/faithfulness_auto.csv     初标结果 (与 faithfulness_template.csv 同 schema + auto 字段)
  results/faithfulness_auto_summary.json  统计汇总

人工 review: 把 faithfulness_auto.csv 另存为 faithfulness_annotation.csv,
逐行核对 reasoning_faithful 与 note, 改错的几条, 重新保存即可.

## 自测
  python src/auto_faithfulness_check.py --self_test
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from utils import RESULTS_DIR, TEST_TAXI_PATH, prediction_path
from evaluator import eval_single


# ── 配置 ────────────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.40        # token 覆盖率阈值
DEFAULT_WINDOW    = 120         # 引用右侧截取字符数
MIN_TOKEN_COUNT   = 2           # 声明 token 少于此数视为 inconclusive
MIN_REASONING_LEN = 50          # reasoning 短于此长度视为 inconclusive

# 数字词 → 数字（与 evaluator 同步）
W2N = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

# 简单同义词归一化（仅本任务高频词）
SYN = {
    "depart":  "leave", "departs": "leave", "departing": "leave",
    "leaving": "leave", "leaves": "leave",
    "arrive":  "arrive", "arriving": "arrive", "arrives": "arrive",
    "centre":  "centre", "center": "centre",
    "theatre": "theatre", "theater": "theatre",
    "destination": "destination",
    "wifi": "internet", "wi-fi": "internet",
    "free":  "yes",   # parking/internet
}

# 极常见 stopword + 句法填充词（不计入 token）
STOPWORDS = {
    "the","a","an","of","to","at","in","on","by","for","and","or","but",
    "is","are","was","were","be","been","being","this","that","these","those",
    "i","you","he","she","it","we","they","my","your","his","her","its","our","their",
    "user","said","says","mention","mentioned","mentions","stated","states","note","notes",
    "explicit","implicit","evidence","step","reasoning","turn","slot","value","none",
    "domain","check","activation","search","inference","decision","verification",
    "no","yes","ok","also","because","since","when","where","what","which","who","how",
    "so","then","thus","therefore","still","just","not","do","does","did","have","has","had",
    "can","could","would","should","will","may","might","must",
    "with","from","into","onto","about","over","under","through",
    "as","than","then","there","here","now",
    "would","like","want","wants","wanted","need","needs","needed","looking","look",
}


# ── 文本规整 ─────────────────────────────────────────────────
def _normalize_token(t: str) -> str:
    t = t.lower().strip(",.!?\"'`()[]{}<>:;")
    t = W2N.get(t, t)
    t = SYN.get(t, t)
    return t


def _tokenize(text: str) -> list[str]:
    """轻量 tokenize: 单词 + 时间 (9:15) + 数字。"""
    if not text:
        return []
    # 抽时间 / 数字
    raw = re.findall(r"\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm)|\d+|[A-Za-z][A-Za-z'-]+",
                     text.lower())
    return [_normalize_token(t) for t in raw if t]


def _content_tokens(text: str) -> set[str]:
    """剔除 stopwords / 短词的实质 token 集合。"""
    return {t for t in _tokenize(text) if t and t not in STOPWORDS and len(t) >= 2}


# ── turn 文本回填 ────────────────────────────────────────────
def _user_turn_text(history: list, turn_n: int) -> str | None:
    """取 history 中第 turn_n 个 user 发言（1-indexed）。"""
    t = 0
    for role, text in history:
        if role == "user":
            t += 1
            if t == turn_n:
                return text
    return None


# ── reasoning 解析 ──────────────────────────────────────────
TURN_REF_RE = re.compile(r"\[?\bturn\s+(\d+)\]?", re.IGNORECASE)


def _find_turn_references(reasoning: str) -> list[tuple[int, int]]:
    """返回 [(turn_n, end_position_in_reasoning), ...]"""
    return [(int(m.group(1)), m.end()) for m in TURN_REF_RE.finditer(reasoning)]


# ── 单条样本判定 ─────────────────────────────────────────────
def check_one(reasoning: str, history: list, *,
              threshold: float = DEFAULT_THRESHOLD,
              window: int = DEFAULT_WINDOW) -> dict:
    """对单个样本判定 reasoning 忠实性。返回 dict 含 verdict / details。"""
    if not reasoning or len(reasoning) < MIN_REASONING_LEN:
        return {"verdict": "inconclusive",
                "reason":  "reasoning_too_short",
                "n_refs": 0, "n_match": 0, "n_mismatch": 0, "details": []}

    refs = _find_turn_references(reasoning)
    if not refs:
        return {"verdict": "inconclusive",
                "reason":  "no_turn_references",
                "n_refs": 0, "n_match": 0, "n_mismatch": 0, "details": []}

    details = []
    n_match = n_mismatch = 0

    for turn_n, end_pos in refs:
        claim_text = reasoning[end_pos: end_pos + window]
        # 截到下一个 turn 引用或换行段落边界，避免跨声明污染
        cut = re.search(r"\n\n|\[turn\s+\d+\]", claim_text, flags=re.IGNORECASE)
        if cut:
            claim_text = claim_text[: cut.start()]

        turn_text = _user_turn_text(history, turn_n)
        if turn_text is None:
            n_mismatch += 1
            details.append({
                "turn": turn_n, "verdict": "missing_turn",
                "claim_snippet": claim_text[:60], "turn_text": None,
                "claim_tokens": [], "overlap": []
            })
            continue

        claim_tokens = _content_tokens(claim_text)
        turn_tokens  = _content_tokens(turn_text)

        if len(claim_tokens) < MIN_TOKEN_COUNT:
            # 引用后紧跟着没有实质内容（可能只是"as shown in [turn 3]"风格）
            # 不计入 mismatch 也不计入 match — 弱判定
            details.append({
                "turn": turn_n, "verdict": "weak_claim",
                "claim_snippet": claim_text[:60], "turn_text": turn_text[:60],
                "claim_tokens": sorted(claim_tokens), "overlap": []
            })
            continue

        overlap = claim_tokens & turn_tokens
        ratio = len(overlap) / max(len(claim_tokens), 1)
        ok = ratio >= threshold

        if ok:
            n_match += 1
        else:
            n_mismatch += 1
        details.append({
            "turn": turn_n,
            "verdict": "match" if ok else "mismatch",
            "ratio": round(ratio, 3),
            "claim_snippet": claim_text[:80].strip(),
            "turn_text":     turn_text[:80].strip(),
            "claim_tokens":  sorted(claim_tokens),
            "overlap":       sorted(overlap),
        })

    if n_mismatch == 0 and n_match > 0:
        verdict = "TRUE"
    elif n_mismatch > 0:
        verdict = "FALSE"
    else:
        verdict = "inconclusive"

    return {
        "verdict":     verdict,
        "reason":      "all_match" if verdict == "TRUE" else
                       "has_mismatch" if verdict == "FALSE" else "all_weak",
        "n_refs":      len(refs),
        "n_match":     n_match,
        "n_mismatch":  n_mismatch,
        "details":     details,
    }


# ── 批量处理：从 cot_full 预测中抽样并标注 ──────────────────
def _format_history(history: list) -> str:
    out, t = [], 0
    for role, text in history:
        if role == "user":
            t += 1
            out.append(f"[turn {t}] User: {text}")
        else:
            out.append(f"           System: {text}")
    return "\n".join(out)


def _summarize_details(d: dict) -> str:
    """把 details 折叠为一行 note，便于人工 review。"""
    pieces = []
    for det in d["details"]:
        v = det["verdict"]
        if v == "match":
            pieces.append(f"turn{det['turn']}✓r={det['ratio']}")
        elif v == "mismatch":
            tail = (" claim≈" + " ".join(det['claim_tokens'][:4])) if det.get('claim_tokens') else ""
            pieces.append(f"turn{det['turn']}✗r={det['ratio']}{tail}")
        elif v == "weak_claim":
            pieces.append(f"turn{det['turn']}?weak")
        elif v == "missing_turn":
            pieces.append(f"turn{det['turn']}!nohist")
    return f"[{d['reason']}] " + " | ".join(pieces)


def run_auto_check(tag: str = "taxi_full",
                   n_samples: int = 50,
                   seed: int = 42,
                   target_domain: str = "taxi",
                   threshold: float = DEFAULT_THRESHOLD) -> Path:
    """读 cot_full 预测，抽 n 条 jga=true 样本，自动标注，写 CSV。"""
    pred_path = prediction_path("cot_full", tag)
    if not pred_path.exists():
        raise FileNotFoundError(f"找不到 {pred_path}；先跑 Phase 2B")

    with open(pred_path, encoding="utf-8") as f:
        preds = json.load(f)
    with open(TEST_TAXI_PATH, encoding="utf-8") as f:
        hist_idx = {(d["dial_id"], d["turn_id"]): d for d in json.load(f)}

    correct = []
    for item in preds:
        res = eval_single(item["gold_belief"], item["pred_belief"],
                          target_domain=target_domain)
        if res["jga"]:
            correct.append(item)

    print(f"总样本: {len(preds)}  |  jga=True ({target_domain}): {len(correct)}")
    if len(correct) < n_samples:
        raise RuntimeError(f"需要 {n_samples} 条 jga=True 但只有 {len(correct)}")

    rng = random.Random(seed)
    sampled = rng.sample(correct, n_samples)

    rows = []
    counts = {"TRUE": 0, "FALSE": 0, "inconclusive": 0}
    for i, item in enumerate(sampled, start=1):
        key = (item["dial_id"], item["turn_id"])
        sample = hist_idx.get(key, {})
        history = sample.get("history", [])
        reasoning = item.get("reasoning", "") or ""

        result = check_one(reasoning, history, threshold=threshold)
        counts[result["verdict"]] += 1

        rows.append({
            "sample_idx":          i,
            "dial_id":             item["dial_id"],
            "turn_id":             item["turn_id"],
            "history":             _format_history(history),
            "reasoning":           reasoning,
            "gold_belief":         json.dumps(item.get("gold_belief", {}), ensure_ascii=False),
            "pred_belief":         json.dumps(item.get("pred_belief", {}), ensure_ascii=False),
            "reasoning_faithful":  result["verdict"],
            "note":                _summarize_details(result),
            "auto_method":         f"rule-overlap@{threshold}",
            "n_refs":              result["n_refs"],
            "n_match":             result["n_match"],
            "n_mismatch":          result["n_mismatch"],
        })

    out = RESULTS_DIR / "faithfulness_auto.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")

    summary = {
        "tag":         tag,
        "n_samples":   n_samples,
        "threshold":   threshold,
        "seed":        seed,
        "verdict_counts": counts,
        "verdict_pct":    {k: round(v / n_samples * 100, 1) for k, v in counts.items()},
    }
    summary_path = RESULTS_DIR / "faithfulness_auto_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ 初标已写入: {out}")
    print(f"   分布: {counts}")
    print(f"📋 摘要: {summary_path}")
    print(f"\n下一步 (人工 review):")
    print(f"  1. 用 Excel 打开 {out.name}, 重点核查 reasoning_faithful 列")
    print(f"  2. 改正错判的几行（note 列已给出诊断细节）")
    print(f"  3. 另存为 faithfulness_annotation.csv 即可被 select_case_studies / notebook 读取")
    return out


# ── 自测 ────────────────────────────────────────────────────
def _self_test() -> None:
    """5 条 hand-crafted 案例验证规则行为符合预期。"""
    cases = [
        # 1. 完全忠实: reasoning 引用 turn 1 的"cheap east hotel"，turn 1 文本含这些词
        ("Step 2 | Explicit Search: User in [turn 1] mentioned a cheap hotel in the east.",
         [("user", "I need a cheap hotel in the east part of town.")],
         "TRUE"),
        # 2. 虚假引用: reasoning 说 turn 2 提到 7 个人，但 turn 2 文本不含此信息
        ("Step 2 | Explicit Search: [turn 2] user said 7 people for the booking.",
         [("user", "I want a hotel in the east."),
          ("system", "Sure"),
          ("user", "Also free wifi.")],
         "FALSE"),
        # 3. 弱声明: 仅"as shown in [turn 1]"无实质内容
        ("Step 1 | Domain Activation: User has engaged with the hotel domain, as in [turn 1].",
         [("user", "I need a cheap hotel.")],
         "inconclusive"),
        # 4. 空 reasoning
        ("",
         [("user", "anything")],
         "inconclusive"),
        # 5. 数字归一化: "nine" vs "9"
        ("Step 2 | Explicit Search: [turn 1] user wants to leave at 9 am.",
         [("user", "We need a taxi leaving at nine in the morning.")],
         "TRUE"),
    ]
    print("=" * 60)
    print(" auto_faithfulness_check self-test")
    print("=" * 60)
    n_pass = 0
    for i, (reasoning, history, expected) in enumerate(cases, start=1):
        result = check_one(reasoning, history)
        ok = result["verdict"] == expected
        n_pass += int(ok)
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] case {i}: expected={expected}, got={result['verdict']} "
              f"(refs={result['n_refs']}, m={result['n_match']}, x={result['n_mismatch']})")
        if not ok:
            for d in result["details"]:
                print(f"        detail: {d}")
    print("-" * 60)
    print(f" {n_pass}/{len(cases)} cases passed")
    print("=" * 60)
    if n_pass != len(cases):
        sys.exit(1)


# ── CLI ─────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="taxi_full")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_domain", default="taxi")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--self_test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
    else:
        run_auto_check(
            tag=args.tag, n_samples=args.n, seed=args.seed,
            target_domain=args.target_domain, threshold=args.threshold,
        )
