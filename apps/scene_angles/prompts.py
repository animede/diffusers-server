# -*- coding: utf-8 -*-
"""
シーンアングル8種の定義(カメラ指示プロンプト)。

プロンプト文言は ComfyUI ワークフロー
templates-1_click_multiple_scene_angles-v1.0_api.json のノード66〜73
(PrimitiveStringMultiline)の値をそのまま採用する。これらは
Qwen-Edit-2509-Multiple-angles LoRA のトリガー文として学習された定型文であり、
文言を変えると LoRA の効きが劣化する懸念があるため一字一句変更しない
(charsheet の VIEWS_LORA と同じ方針。apps/charsheet/prompts.py 参照)。

ネガティブプロンプトも charsheet と同一(同じLoRA・同じベースモデルのため)。
"""

NEGATIVE_PROMPT = "低分辨率，低画质，肢体畸形，手指畸形"

# 各エントリ: key, label_ja, label_en, prompt(LoRAトリガー文そのまま)
ANGLES = [
    {
        "key": "close_up",
        "label_ja": "クローズアップ",
        "label_en": "Close-up",
        "prompt": "Turn the camera to a close-up.",
    },
    {
        "key": "wide_angle",
        "label_ja": "広角",
        "label_en": "Wide-angle",
        "prompt": "Turn the camera to a wide-angle lens.",
    },
    {
        "key": "right_90",
        "label_ja": "右90度",
        "label_en": "Rotate right 90°",
        "prompt": "Rotate the camera 90 degrees to the right.",
    },
    {
        "key": "right_45",
        "label_ja": "右45度",
        "label_en": "Rotate right 45°",
        "prompt": "Rotate the camera 45 degrees to the right.",
    },
    {
        "key": "aerial",
        "label_ja": "俯瞰(空撮)",
        "label_en": "Aerial view",
        "prompt": "Turn the camera to an aerial view.",
    },
    {
        "key": "low_angle",
        "label_ja": "ローアングル",
        "label_en": "Low-angle view",
        "prompt": "Turn the camera to a low-angle view.",
    },
    {
        "key": "left_90",
        "label_ja": "左90度",
        "label_en": "Rotate left 90°",
        "prompt": "Rotate the camera 90 degrees to the left.",
    },
    {
        "key": "left_45",
        "label_ja": "左45度",
        "label_en": "Rotate left 45°",
        "prompt": "Rotate the camera 45 degrees to the left.",
    },
]

ANGLE_BY_KEY = {a["key"]: a for a in ANGLES}
