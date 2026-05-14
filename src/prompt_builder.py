"""
prompt_builder.py  v3 (ablation-ready)

为零样本跨域 DST 构建多种 prompt 变体，覆盖主实验（实验一）与消融实验（实验二）。

变体矩阵（见 VARIANTS）：
    standard    —  无 reasoning，直出 JSON（消融基线）
    cot_basic   —  "think step by step"，无结构约束
    cot_full    —  四步完整推理链 S1+S2+S3+S4（主方法）
    ab_no_s1    —  缺 Step 1 Domain Activation Check
    ab_no_s2    —  缺 Step 2 Explicit Information Search
    ab_no_s3    —  缺 Step 3 Implicit Intent Inference
    ab_no_s4    —  缺 Step 4 None Decision & Verification（重点：幻觉率变化）

对外接口保持向后兼容：
    build_prompt(sample)                   → 默认 cot_full（四步完整版）
    build_prompt(sample, variant="...")    → 指定变体

SLOT_SCHEMA / ALL_VALID_SLOTS 与原版完全一致，parser.py 无需改动。
"""

from __future__ import annotations

# ── MultiWOZ 2.1 槽位 Schema ────────────────────────────────
# 键名必须与 data/processed/ 中 gold belief 的键名完全一致。
# 预处理脚本将 book-category 槽扁平化为 {domain}-{slot}。
SLOT_SCHEMA = {
    "hotel": {
        "name":       "the name of the hotel",
        "area":       "the area or location of the hotel (centre/east/west/south/north)",
        "pricerange": "the price range of the hotel (cheap/moderate/expensive)",
        "type":       "the type of the accommodation (hotel/guesthouse)",
        "parking":    "whether the hotel offers free parking (yes/no)",
        "internet":   "whether the hotel offers free internet/wifi (yes/no)",
        "stars":      "the star rating of the hotel (0/1/2/3/4/5)",
        "stay":       "the number of nights to stay (from hotel booking)",
        "day":        "the day of the week to check in (from hotel booking)",
        "people":     "the number of people for the hotel booking",
    },
    "restaurant": {
        "name":       "the name of the restaurant",
        "area":       "the area or location of the restaurant (centre/east/west/south/north)",
        "pricerange": "the price range of the restaurant (cheap/moderate/expensive)",
        "food":       "the type of food or cuisine served",
        "time":       "the time of the restaurant reservation (from booking)",
        "day":        "the day of the restaurant reservation (from booking)",
        "people":     "the number of people for the restaurant reservation (from booking)",
    },
    "attraction": {
        "name":       "the name of the attraction",
        "area":       "the area or location of the attraction (centre/east/west/south/north)",
        "type":       "the type of the attraction (museum/park/theatre/cinema/college/...)",
    },
    "taxi": {
        "departure":  "the departure location for the taxi",
        "destination":"the destination location for the taxi",
        "arrive by":  "the time by which the taxi must arrive at the destination",
        "leave at":   "the time at which the taxi should depart",
    },
    "train": {
        "departure":  "the departure station for the train",
        "destination":"the destination station for the train",
        "arrive by":  "the time by which the train must arrive",
        "leave at":   "the time at which the train should depart",
        "day":        "the day of the train journey",
        "people":     "the number of train tickets to book",
    },
}

ALL_VALID_SLOTS: set[str] = {
    f"{domain}-{slot}"
    for domain, slots in SLOT_SCHEMA.items()
    for slot in slots
}

ALL_DOMAINS: list[str] = list(SLOT_SCHEMA.keys())


# ── fn_style 槽位类型注释 ────────────────────────────────────
# 仅 fn_style 变体使用：为每个槽位附加 type / allowed_values，
# 模仿函数调用 schema 的结构化参数描述（参照 FnCTOD 2402.10466）。
# 不含推理指令；仅"信息呈现得更结构化"。
_DAYS = ["monday", "tuesday", "wednesday", "thursday",
         "friday", "saturday", "sunday"]
_AREAS = ["centre", "east", "west", "south", "north"]
_PRICE = ["cheap", "moderate", "expensive"]

SLOT_TYPES: dict[str, dict] = {
    # hotel
    "hotel-name":       {"type": "string"},
    "hotel-area":       {"type": "enum",    "values": _AREAS},
    "hotel-pricerange": {"type": "enum",    "values": _PRICE},
    "hotel-type":       {"type": "enum",    "values": ["hotel", "guesthouse"]},
    "hotel-parking":    {"type": "enum",    "values": ["yes", "no"]},
    "hotel-internet":   {"type": "enum",    "values": ["yes", "no"]},
    "hotel-stars":      {"type": "enum",    "values": ["0", "1", "2", "3", "4", "5"]},
    "hotel-stay":       {"type": "integer"},
    "hotel-day":        {"type": "enum",    "values": _DAYS},
    "hotel-people":     {"type": "integer"},
    # restaurant
    "restaurant-name":       {"type": "string"},
    "restaurant-area":       {"type": "enum",    "values": _AREAS},
    "restaurant-pricerange": {"type": "enum",    "values": _PRICE},
    "restaurant-food":       {"type": "string"},
    "restaurant-time":       {"type": "time",    "format": "HH:MM (24h)"},
    "restaurant-day":        {"type": "enum",    "values": _DAYS},
    "restaurant-people":     {"type": "integer"},
    # attraction
    "attraction-name": {"type": "string"},
    "attraction-area": {"type": "enum", "values": _AREAS},
    "attraction-type": {"type": "string"},
    # taxi
    "taxi-departure":   {"type": "string"},
    "taxi-destination": {"type": "string"},
    "taxi-arrive by":   {"type": "time", "format": "HH:MM (24h)"},
    "taxi-leave at":    {"type": "time", "format": "HH:MM (24h)"},
    # train
    "train-departure":   {"type": "string"},
    "train-destination": {"type": "string"},
    "train-arrive by":   {"type": "time", "format": "HH:MM (24h)"},
    "train-leave at":    {"type": "time", "format": "HH:MM (24h)"},
    "train-day":         {"type": "enum",    "values": _DAYS},
    "train-people":      {"type": "integer"},
}
assert set(SLOT_TYPES.keys()) == ALL_VALID_SLOTS, (
    "SLOT_TYPES must cover exactly the same slots as SLOT_SCHEMA"
)


# ── 基础组件 ─────────────────────────────────────────────────
_HEADER = """You are an expert dialogue state tracker for task-oriented dialogue systems.

Your task is to extract the CUMULATIVE dialogue state (belief state) from a conversation.
The dialogue state is a set of slot-value pairs representing everything the user has requested so far."""


_OUTPUT_RULES = """## Critical Output Rules
1. Use ONLY the slot names listed above (format: domain-slot). Do NOT invent new slot names.
2. Only include slots that have been mentioned or can be DIRECTLY inferred from the conversation.
3. If a slot has NOT been mentioned, do NOT include it — omit it entirely from the JSON.
4. Slot values must be exact strings as spoken by the user (lowercase).
5. The state is CUMULATIVE: include all slots mentioned across ALL turns, not just the latest."""


# ── 四步推理模板 ──────────────────────────────────────────────
# 每一步的文本必须自包含：使用"概念"（活跃域 / 显式证据 / 隐含证据）而非
# 硬编码的"Step N"引用，确保任意子集组合（消融配置）下指令仍自洽。
_STEP_TEMPLATES: dict[str, str] = {
    "s1": """Domain Activation Check:
  List each of the 5 domains (hotel / restaurant / attraction / taxi / train) and
  decide whether the user has engaged with it in this conversation. A domain is
  "activated" only if the user explicitly requested information or booking for it.
  Do NOT fill any slot belonging to a non-activated domain; this blocks cross-domain
  hallucinations (e.g. filling taxi slots when taxi was never discussed).""",

    "s2": """Explicit Information Search:
  For each domain the user has engaged with, scan all USER turns for values that
  are stated in plain words (e.g. "cheap", "east", "5 stars", "9:15"). For each
  candidate value, cite the exact turn index that supports it. If a value is
  updated in a later turn, use the most recent one. Mark these as EXPLICIT
  evidence.""",

    "s3": """Implicit Intent Inference:
  For slots that lack direct, explicit mention in the user's words, check whether
  the value is implied by context (e.g. "the same price range as the hotel" →
  restaurant-pricerange = hotel-pricerange). Fill a slot here ONLY when the
  textual evidence makes the mapping unambiguous. Do NOT invent values based on
  world knowledge or plausible guesses. Mark these as IMPLICIT evidence.""",

    "s4": """None Decision & Verification:
  For EACH slot in the schema, apply the gating rule in order:
    (a) If EXPLICIT evidence supports a value → fill it.
    (b) Else if defensible IMPLICIT evidence supports a value → fill it.
    (c) Otherwise → OMIT the slot (equivalent to none).
  When earlier reasoning offered no support for a slot, treat that as case (c).
  This is the last line of defence against hallucination and must be executed
  for every slot, even those not mentioned earlier.""",
}


# 每个步骤的"短标题"，用于 <reasoning> 块的 placeholder。
_STEP_TITLES: dict[str, str] = {
    "s1": "Domain Activation",
    "s2": "Explicit Search",
    "s3": "Implicit Inference",
    "s4": "None Verification",
}


def _format_step_instructions(step_ids: list[str]) -> str:
    """按保留顺序为步骤动态编号 Step 1..k，避免硬编号在消融下断裂。"""
    blocks = []
    for i, sid in enumerate(step_ids, start=1):
        blocks.append(f"Step {i} — {_STEP_TEMPLATES[sid]}")
    return "\n\n".join(blocks)


def _make_reasoning_block_header(step_ids: list[str]) -> str:
    """构造 <reasoning> 块的 placeholder 模板，编号与上方 step instructions 一致。"""
    lines = [
        f"Step {i} | {_STEP_TITLES[sid]}:\n[your analysis here]"
        for i, sid in enumerate(step_ids, start=1)
    ]
    return "\n\n".join(lines)


# ── Variant 定义 ─────────────────────────────────────────────
# style:
#   "none"       —  不要求 reasoning，直接输出 JSON
#   "free"       —  要求 "think step by step"，自由推理
#   "structured" —  要求 <reasoning> 块，按指定 step 顺序输出
VARIANTS: dict[str, dict] = {
    "standard": {
        "steps": [],
        "style": "none",
        "description": "Standard Prompting (no CoT)",
    },
    "cot_basic": {
        "steps": [],
        "style": "free",
        "description": "CoT-Basic (free-form 'think step by step')",
    },
    "cot_full": {
        "steps": ["s1", "s2", "s3", "s4"],
        "style": "structured",
        "description": "CoT-Full (all 4 steps)",
    },
    "ab_no_s1": {
        "steps": ["s2", "s3", "s4"],
        "style": "structured",
        "description": "Ablation: drop Step 1 (Domain Activation)",
    },
    "ab_no_s2": {
        "steps": ["s1", "s3", "s4"],
        "style": "structured",
        "description": "Ablation: drop Step 2 (Explicit Search)",
    },
    "ab_no_s3": {
        "steps": ["s1", "s2", "s4"],
        "style": "structured",
        "description": "Ablation: drop Step 3 (Implicit Inference)",
    },
    "ab_no_s4": {
        "steps": ["s1", "s2", "s3"],
        "style": "structured",
        "description": "Ablation: drop Step 4 (None Decision) — expect ↑ hallucination",
    },
    "fn_style": {
        "steps": [],
        "style": "fn",
        "description": "Function-Calling Style (rich schema, no reasoning)",
    },
}


# ── 辅助函数 ─────────────────────────────────────────────────
def format_history(history: list) -> str:
    """将 [(role, text), ...] 对话历史转为可读字符串，标注轮号以便 CoT 引用。"""
    lines = []
    turn_idx = 0
    for role, text in history:
        if role == "user":
            turn_idx += 1
            lines.append(f"[turn {turn_idx}] User: {text}")
        else:
            lines.append(f"           System: {text}")
    return "\n".join(lines)


def build_slot_list() -> str:
    """生成槽位说明清单，供 system prompt 引用。"""
    lines = []
    for domain, slots in SLOT_SCHEMA.items():
        for slot, desc in slots.items():
            lines.append(f"  - {domain}-{slot}: {desc}")
    return "\n".join(lines)


def build_slot_list_fn_style() -> str:
    """fn_style 槽位清单：附 type / optional / description / allowed_values。

    格式参照 FnCTOD（2402.10466）的函数参数风格，但用纯文本渲染、
    不真正使用 OpenAI function calling API。每行形如：
        - domain-slot: (enum, optional) — description [allowed: a | b | c]
    """
    lines = []
    for domain, slots in SLOT_SCHEMA.items():
        for slot, desc in slots.items():
            key = f"{domain}-{slot}"
            t = SLOT_TYPES[key]
            type_label = t["type"]
            if type_label == "time":
                type_label = f"time, format={t.get('format', 'HH:MM')}"
            tail = ""
            if t["type"] == "enum":
                tail = f" [allowed: {' | '.join(t['values'])}]"
            lines.append(f"  - {key}: ({type_label}, optional) — {desc}{tail}")
    return "\n".join(lines)


def list_variants() -> list[str]:
    """对外暴露的合法 variant 列表，供 CLI 参数校验。"""
    return list(VARIANTS.keys())


# ── Prompt 构建 ──────────────────────────────────────────────
def build_system_prompt(variant: str = "cot_full") -> str:
    """根据 variant 拼装完整的 system prompt。"""
    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown variant: {variant!r}. "
            f"Valid options: {list_variants()}"
        )

    cfg = VARIANTS[variant]
    style = cfg["style"]
    steps = cfg["steps"]

    # fn_style 用结构化 schema 描述代替平铺槽位列表；其它变体保持原样
    if style == "fn":
        slot_block = build_slot_list_fn_style()
        slot_header = "## Slot Schema (function-call style)"
    else:
        slot_block = build_slot_list()
        slot_header = "## Available Slots"

    parts = [
        _HEADER,
        "",
        slot_header,
        slot_block,
        "",
        _OUTPUT_RULES,
        "",
    ]

    # --- 推理部分 ---
    if style in ("none", "fn"):
        parts += [
            "## Output Format (follow exactly)",
            "Output ONLY a fenced JSON block. Do NOT include any explanation or reasoning.",
            "",
            "```json",
            '{"domain-slot": "value"}',
            "```",
        ]

    elif style == "free":
        parts += [
            "## Reasoning Protocol",
            "Let's think step by step. First, briefly reason about what the user has "
            "asked for in each turn. Then output the final belief state as JSON.",
            "",
            "## Output Format (follow exactly)",
            "<reasoning>",
            "[your step-by-step reasoning here]",
            "</reasoning>",
            "",
            "```json",
            '{"domain-slot": "value"}',
            "```",
        ]

    elif style == "structured":
        # 结构化 CoT：按保留的步骤动态编号（消融下仍保持 Step 1..k 连续）
        step_instructions = _format_step_instructions(steps)
        reasoning_skeleton = _make_reasoning_block_header(steps)
        parts += [
            "## Reasoning Protocol (Chain-of-Thought)",
            "Before the final JSON, you MUST complete every step below inside a "
            "<reasoning> block, in the exact order given:",
            "",
            step_instructions,
            "",
            "## Output Format (follow exactly)",
            "<reasoning>",
            reasoning_skeleton,
            "</reasoning>",
            "",
            "```json",
            '{"domain-slot": "value"}',
            "```",
        ]

    else:
        raise RuntimeError(f"Invalid style {style!r} for variant {variant!r}")

    return "\n".join(parts)


def build_prompt(sample: dict, variant: str = "cot_full") -> tuple[str, str]:
    """
    将单条 DST 样本转换为 (system_prompt, user_message) 对。

    Args:
        sample: processed/test.json 中的单条记录
                （dial_id, turn_id, history, belief）
        variant: VARIANTS 中的任一键，默认 "cot_full"

    Returns:
        (system_prompt_str, user_message_str)
    """
    system = build_system_prompt(variant)
    history_str = format_history(sample["history"])

    if VARIANTS[variant]["style"] in ("none", "fn"):
        instruction = (
            "Extract the complete cumulative dialogue state at this point. "
            "Output ONLY the JSON block."
        )
    else:
        instruction = (
            "Extract the complete cumulative dialogue state at this point in the "
            "conversation.\nFollow the reasoning protocol above, then output the JSON."
        )

    user_message = f"## Conversation History\n{history_str}\n\n{instruction}"
    return system, user_message


# ── 快速自测 ────────────────────────────────────────────────
if __name__ == "__main__":
    fake_sample = {
        "dial_id": "test_001",
        "turn_id": 2,
        "history": [
            ("user",   "I need a cheap hotel in the east part of town."),
            ("system", "I found several options. Any other requirements?"),
            ("user",   "Yes, it must have free parking and internet. I need it for 3 nights."),
        ],
        "belief": {
            "hotel-pricerange": "cheap",
            "hotel-area":       "east",
            "hotel-parking":    "yes",
            "hotel-internet":   "yes",
            "hotel-stay":       "3",
        },
    }

    for variant in list_variants():
        sys_p, user_p = build_prompt(fake_sample, variant=variant)
        print("=" * 72)
        print(f"VARIANT: {variant}  —  {VARIANTS[variant]['description']}")
        print("-" * 72)
        print(f"[SYSTEM PROMPT LEN] {len(sys_p)} chars")
        print(f"[STEPS INCLUDED]    {VARIANTS[variant]['steps'] or '—'}")
    print("=" * 72)
    print(f"Total variants: {len(list_variants())}")
    print(f"Valid slots:    {len(ALL_VALID_SLOTS)}")
