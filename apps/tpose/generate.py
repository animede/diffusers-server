# -*- coding: utf-8 -*-
"""Tポーズ4ビュー生成の呼び出し層。

apps/charsheet/generate.py とは違い **Multiple-angles LoRA を使わない**
(apps/tpose/prompts.py の冒頭コメント参照。Tポーズでは通常 Edit の方が
同一性・速度とも優位という実機検証結果に基づく)。したがって
families/qwen_image の通常 Edit(mode="edit"、DS_QUANT 既定 fp8-lightning)を
registry 経由でロードして使う(CLAUDE.md 1番: registry.load() を経由しないと
排他グループの自動 unload が効かない)。

解像度は charsheet と同じ既定1024²(DS_TPOSE_SIZE で上書き可)。TE CPU退避
(DS_EDIT_TE_OFFLOAD、既定auto)が families/qwen_image 側で効くため、48GB専有でも
1024²で完走する(CLAUDE.md 23番)。
"""
import os
from typing import List, Optional

from PIL import Image

from core import gpu
from core.registry import registry

import families.qwen_image as qwen_image

_DEFAULT_SIZE = 1024
_MODE = "edit"


def edit_size() -> int:
    try:
        return int(os.environ.get("DS_TPOSE_SIZE", str(_DEFAULT_SIZE)))
    except ValueError:
        return _DEFAULT_SIZE


def preprocess_image(image: Image.Image) -> Image.Image:
    """ComfyUI の ImageScaleToTotalPixels(~1MP)相当の前処理(charsheetと同一実装)。"""
    return qwen_image.preprocess_image(image)


def ensure_edit_loaded() -> None:
    """通常 Edit グループをロードする(ジョブ開始前に1回だけ呼ぶ)。"""
    registry.load("qwen_image", _MODE)


def generate_view(
    images: List[Image.Image],
    prompt: str,
    seed: int = 0,
    negative_prompt: str = "",
    size: Optional[int] = None,
    progress_extra: Optional[dict] = None,
) -> dict:
    """1ビュー分を生成する。images は参照画像1〜3枚(2枚目以降はしっぽ参照等)。

    戻り値は families/qwen_image の run_edit() のメタデータ dict
    (image_url / elapsed_s / peak_vram_gb 等)。
    """
    px = size or edit_size()
    req = {
        "mode": _MODE,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": 4,
        "cfg": 1.0,
        "seed": seed if seed and seed > 0 else -1,
        "lightning": True,
        "shift": None,
        "width": px,
        "height": px,
        "_images": images,
        "_progress_extra": progress_extra,
    }
    family = registry.get("qwen_image")
    return family.generate(req)


def _name_color(rgb) -> str:
    """RGB(0-255)を粗い色名へ写す(プロンプトへ埋め込むための語彙)。"""
    import colorsys

    r, g, b = [float(v) / 255.0 for v in rgb]
    hue, sat, val = colorsys.rgb_to_hsv(r, g, b)
    hue *= 360.0
    if sat < 0.12:
        if val > 0.85:
            return "white"
        if val > 0.6:
            return "light grey"
        if val > 0.3:
            return "grey"
        return "black"
    if val > 0.78 and sat < 0.35 and 20.0 <= hue <= 65.0:
        return "cream white"
    if hue < 20.0 or hue >= 330.0:
        return "pink" if (val > 0.7 and sat < 0.5) else "reddish brown"
    if hue < 45.0:
        return "light brown" if val > 0.6 else "brown"
    if hue < 70.0:
        return "cream yellow" if val > 0.8 else "yellow"
    if hue < 170.0:
        return "green"
    if hue < 260.0:
        return "blue"
    return "purple"


def sample_fur_color(image: Image.Image) -> str:
    """入力画像から被写体の毛色を推定して色名を返す(失敗時は "")。

    「爪を消すと後足の指も消える」問題への対処に使う(apps/tpose/prompts.py の
    `_claws_none_with_color()` 参照。実測で「同じ毛色で」のような相対表現では効かず、
    **具体的な色名**が必要だった)。

    手順: rembg(背景除去に既に使っているCPU/ONNXモデル)で被写体マスクを取り、
    彩度の低い画素(=毛。衣装は彩度が高いことが多い)の中央値を色名へ写す。
    彩度の低い画素が被写体の15%未満しか無い場合(全身が濃色の衣装で覆われている等)は
    被写体全体の中央値を使う。rembg が使えない環境では "" を返し、呼び出し側は
    色名なしのフォールバック文言を使う。
    """
    try:
        import colorsys

        import numpy as np

        from apps.charsheet.bg import remove_background

        rgba = np.array(remove_background(image.convert("RGB")).convert("RGBA"))
    except Exception as exc:  # noqa: BLE001
        print(f"[apps.tpose] warning: 毛色の推定に失敗しました({exc})。色名なしで続行します。")
        return ""

    alpha = rgba[..., 3]
    rgb = rgba[..., :3].astype(float)
    subject = alpha > 200
    if subject.sum() < 500:
        return ""
    px = rgb[subject] / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*p) for p in px[:: max(1, len(px) // 20000)]])
    low_sat = hsv[:, 1] < 0.30
    sel = hsv[low_sat] if low_sat.mean() >= 0.15 else hsv
    h, s, v = np.median(sel, axis=0)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return _name_color((r * 255, g * 255, b * 255))


def acquire_generation_lock(blocking: bool = True) -> bool:
    return gpu.generation_lock.acquire(blocking=blocking)


def release_generation_lock() -> None:
    gpu.generation_lock.release()


def get_load_info() -> dict:
    from families.qwen_image import state

    group = state.edit_group
    return {
        "loaded": group.get("loaded", False),
        "quant": group.get("quant"),
        "fallback": group.get("fallback", False),
        "lightning_merged": group.get("lightning_merged"),
        "load_time_s": group.get("load_time_s"),
        "angles_lora": False,
        "method": "edit",
    }
