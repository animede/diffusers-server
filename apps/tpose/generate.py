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
