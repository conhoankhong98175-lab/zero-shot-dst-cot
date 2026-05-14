"""
build_faithfulness_template.py
从 cot_full 全量预测中抽 50 条 jga=True 样本，生成人工标注 CSV 模板。

输入:
    results/preds_cot_full_{tag}.json    (默认 tag = taxi_full)
    data/processed/test_taxi.json         (用于回填 history)

输出:
    results/faithfulness_template.csv

标注者随后用 Excel / VS Code 打开 CSV，在 reasoning_faithful 列填 TRUE/FALSE,
然后另存为 results/faithfulness_annotation.csv 供后续分析使用 (见 notebook).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from utils import RESULTS_DIR, TEST_TAXI_PATH, prediction_path
from evaluator import eval_single


def _format_history(history: list) -> str:
    out, t = [], 0
    for role, text in history:
        if role == "user":
            t += 1
            out.append(f"[turn {t}] User: {text}")
        else:
            out.append(f"           System: {text}")
    return "\n".join(out)


def build(tag: str, n: int = 50, seed: int = 42,
          target_domain: str = "taxi") -> Path:
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
    if len(correct) < n:
        raise RuntimeError(
            f"需要 {n} 条 jga=True 样本，实际只有 {len(correct)}。"
        )

    rng = random.Random(seed)
    sampled = rng.sample(correct, n)

    rows = []
    for i, item in enumerate(sampled, start=1):
        sample = hist_idx.get((item["dial_id"], item["turn_id"]), {})
        rows.append({
            "sample_idx":          i,
            "dial_id":             item["dial_id"],
            "turn_id":             item["turn_id"],
            "history":             _format_history(sample.get("history", [])),
            "reasoning":           item.get("reasoning", ""),
            "gold_belief":         json.dumps(item.get("gold_belief", {}), ensure_ascii=False),
            "pred_belief":         json.dumps(item.get("pred_belief", {}), ensure_ascii=False),
            "reasoning_faithful":  "",   # ← 标注者填 TRUE / FALSE
            "note":                "",
        })

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "faithfulness_template.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"✅ 模板已写入: {out}")
    print(f"   {n} 条样本 | seed={seed} | 标注者: 填 reasoning_faithful 列后另存为 faithfulness_annotation.csv")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="taxi_full")
    ap.add_argument("--n",   type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target_domain", default="taxi")
    args = ap.parse_args()
    build(args.tag, args.n, args.seed, args.target_domain)
