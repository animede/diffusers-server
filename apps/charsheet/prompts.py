# -*- coding: utf-8 -*-
"""
8方向の視点定義(旧 charsheet/prompts.py 移植)。
各エントリ: key, label_ja, label_en, prompt

Phase 3 での方針転換(ユーザー目視比較の結果に基づく): 当初実装(プロンプトのみ、
Multiple-angles LoRA 廃止)で新旧比較したところ「コスチューム等の一貫性が旧LoRA版
よりかなり劣る」と判定されたため、LoRA 併用(families/qwen_image/edit_angles.py の
fp8-lightning-angles 変種)を復活させた。LoRA は特定のプロンプト文言(旧
~/charsheet/prompts.py のプロンプト、"keep items" 補強なし)でトリガー
学習されているため、LoRA 使用時はそのプロンプト形式をそのまま復元して使う
(補強を追加すると LoRA が学習していない文言になり、効果が薄れる/ノイズになる懸念)。

このモジュールは両方の形式を保持する:
  - VIEWS_LORA        : 旧 charsheet/prompts.py と同一の文言(LoRAトリガー文、
                        "keep items" 補強なし)
  - VIEWS_PROMPT_ONLY : Phase 3 当初実装のプロンプト(全方向末尾に "keep items" 補強
                        文を追加。LoRAなしのプロンプトのみ多アングル生成向け)

VIEWS / VIEW_BY_KEY は apps/charsheet/generate.py の use_angles_lora() の値に応じて
どちらかを指すモジュールレベル変数(呼び出し側 apps/charsheet/jobs.py が
`from apps.charsheet.prompts import VIEW_BY_KEY, VIEWS` する際に、プロセス起動時の
環境変数 DS_CHARSHEET_USE_ANGLES_LORA の値で選択される)。
"""
from apps.charsheet.generate import use_angles_lora

NEGATIVE_PROMPT = "低分辨率，低画质，肢体畸形，手指畸形"

_KEEP_ITEMS_SUFFIX = (
    " Keep all held items, weapons, and accessories exactly as shown in the "
    "reference image; do not omit or change any props, bags, weapons or "
    "decorations the character is holding or wearing."
)

# --- LoRA使用時(既定): 旧 charsheet/prompts.py と同一文言(LoRAトリガー文) ---
VIEWS_LORA = [
    {
        "key": "front",
        "label_ja": "前",
        "label_en": "Front",
        "prompt": (
            "Show the character in a full body front view, facing directly "
            "toward the camera, neutral standing A-pose with arms slightly "
            "away from the body, full figure visible from head to toe"
        ),
    },
    {
        "key": "back",
        "label_ja": "後ろ",
        "label_en": "Back",
        "prompt": (
            "Show the character from behind in a full body back view, rear "
            "facing the camera, neutral standing pose, showing all back "
            "details, hair, costume and accessories from behind, full "
            "figure head to toe"
        ),
    },
    {
        "key": "left",
        "label_ja": "左",
        "label_en": "Left",
        "prompt": (
            "Show the character in a full body left side profile view, "
            "character facing to the left, neutral standing pose, showing "
            "the complete left side silhouette from head to toe"
        ),
    },
    {
        "key": "right",
        "label_ja": "右",
        "label_en": "Right",
        "prompt": (
            "Show the character in a full body right side profile view, "
            "character facing to the right, neutral standing pose, showing "
            "the complete right side silhouette from head to toe"
        ),
    },
    {
        "key": "front_left_45",
        "label_ja": "左前45度",
        "label_en": "Front-Left 45°",
        "prompt": (
            "Show the character from a 3/4 front-left angle, neutral "
            "standing pose, showing both front and left side details, full "
            "body visible from head to toe"
        ),
    },
    {
        "key": "front_right_45",
        "label_ja": "右前45度",
        "label_en": "Front-Right 45°",
        "prompt": (
            "Show the character from a 3/4 front-right angle, neutral "
            "standing pose, showing both front and right side details, "
            "full body visible from head to toe"
        ),
    },
    {
        "key": "back_left_45",
        "label_ja": "左後ろ45度",
        "label_en": "Back-Left 45°",
        "prompt": (
            "Show the character from a 3/4 back-left angle, neutral "
            "standing pose, showing back and left side details, full body "
            "visible from head to toe"
        ),
    },
    {
        "key": "back_right_45",
        "label_ja": "右後ろ45度",
        "label_en": "Back-Right 45°",
        "prompt": (
            "Show the character from a 3/4 back-right angle, neutral "
            "standing pose, showing back and right side details, full body "
            "visible from head to toe"
        ),
    },
]

# --- LoRAなし(プロンプトのみ)時: Phase 3 当初実装。小物保持補強つき ---
def _with_keep_items_suffix(prompt: str) -> str:
    base = prompt if prompt.endswith(".") else prompt + "."
    return base + _KEEP_ITEMS_SUFFIX


VIEWS_PROMPT_ONLY = [{**v, "prompt": _with_keep_items_suffix(v["prompt"])} for v in VIEWS_LORA]

VIEWS = VIEWS_PROMPT_ONLY if not use_angles_lora() else VIEWS_LORA
VIEW_BY_KEY = {v["key"]: v for v in VIEWS}

# キャラクターシート合成時の並び順(旧仕様維持)
# 1行目: 前系, 2行目: 後ろ系
SHEET_LAYOUT = [
    ["front", "front_left_45", "front_right_45", "left"],
    ["right", "back_left_45", "back_right_45", "back"],
]
