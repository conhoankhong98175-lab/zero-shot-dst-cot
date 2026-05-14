"""
parser.py
从 LLM 原始输出中稳健地抽取 slot-value JSON。

解析策略（优先级从高到低）：
  1. 提取 ```json ... ``` 代码块（标准输出格式）
  2. 提取最后一个完整 {...} 块（fallback）
  3. 返回空 dict（彻底失败时）

[v2 新增] 使用 prompt_builder.ALL_VALID_SLOTS 过滤幻觉槽位，
  防止模型输出不存在的键（如 "hotel-book people"）污染评测结果。
"""

import json
import re
from prompt_builder import ALL_VALID_SLOTS


def extract_belief(raw_output: str) -> dict:
    """
    从模型原始输出中提取 belief state dict。

    Returns:
        dict of {slot: value}，解析失败返回 {}
    """
    json_str = _find_json_block(raw_output)
    if not json_str:
        return {}

    raw_dict = _safe_json_parse(json_str)
    if raw_dict is None:
        return {}

    return _clean_belief(raw_dict)


def _find_json_block(text: str) -> str | None:
    """在文本中定位 JSON 字符串。"""
    # 策略 1：标准 ```json ... ``` 代码块
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if m:
        return m.group(1).strip()

    # 策略 2：寻找最后一个完整 {...} 块
    # 用栈式扫描处理嵌套括号，比正则更可靠
    start = None
    depth = 0
    last_block = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                last_block = text[start: i + 1]
    return last_block


def _safe_json_parse(json_str: str) -> dict | None:
    """带自动修复的 JSON 解析。"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # 修复 1：末尾多余逗号（LLM 常见错误）
    fixed = re.sub(r",\s*([}\]])", r"\1", json_str)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    return None


def _clean_belief(raw_dict: dict) -> dict:
    """
    清洗 belief dict：
      1. key/value 均 lowercase + strip
      2. 过滤空值和 "none"/"null" 等无效值
      3. 过滤不在合法 schema 中的槽位（防幻觉污染评测）
    """
    cleaned = {}
    invalid_values = {"none", "null", "", "not mentioned", "n/a", "unknown"}

    for k, v in raw_dict.items():
        if isinstance(v, bool):
            v = "yes" if v else "no"
        elif isinstance(v, (int, float)):
            v = str(int(v)) if isinstance(v, float) and v == int(v) else str(v)
        elif not isinstance(v, str):
            continue
        key = k.lower().strip()
        val = v.lower().strip()

        if val in invalid_values:
            continue

        # 过滤非法槽位名（不在 SLOT_SCHEMA 中的键视为幻觉）
        if key not in ALL_VALID_SLOTS:
            continue

        cleaned[key] = val

    return cleaned


def extract_reasoning(raw_output: str) -> str:
    """提取 <reasoning>...</reasoning> 块内容，供论文分析使用。"""
    m = re.search(r"<reasoning>([\s\S]*?)</reasoning>", raw_output)
    return m.group(1).strip() if m else ""


# ── 自测 ─────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        # 正常输出
        ("""
<reasoning>
Step 1 | Intent Analysis: user wants cheap hotel in east.
Step 2 | Multi-step Inference: hotel-pricerange: cheap (turn 1). hotel-area: east (turn 1).
Step 3 | None Verification: hotel-name: not mentioned.
</reasoning>

```json
{
  "hotel-pricerange": "cheap",
  "hotel-area": "east",
  "hotel-parking": "yes",
  "hotel-book people": "3"
}
```
""", "正常输出（含非法键 hotel-book people，应被过滤）"),

        # 缺少代码块的 fallback 情况
        ('The state is: {"restaurant-food": "chinese", "restaurant-area": "centre"}',
         "无代码块 fallback"),

        # 末尾多余逗号
        ('```json\n{"hotel-area": "east",\n"hotel-pricerange": "cheap",\n}\n```',
         "末尾逗号修复"),
    ]

    for output, desc in test_cases:
        print(f"\n[{desc}]")
        belief    = extract_belief(output)
        reasoning = extract_reasoning(output)
        print("  belief   :", belief)
        if reasoning:
            print("  reasoning:", reasoning[:80], "...")
