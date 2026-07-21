# -*- coding: utf-8 -*-
"""
Z-Image 生成本体(T2I / I2I / Inpaint)。

抽出元: ~/Z-image-diffusers/api_server.py の api_t2i() / api_i2i() /
api_edit()(base64 JSON API)。この統合サーバでは他ファミリーと同じ規約
(画像は前処理済みで渡される・outputs/保存とメタデータ組み立てまでこのモジュールが担当)に
合わせ、base64エンコードは行わない。

Z-Image-Turbo は蒸留モデル(既定 steps=8, guidance_scale=0.0 = CFGなし)。
guidance_scale>1 を指定すると negative_prompt が効くが、denoiseが2倍遅くなる
(旧実装ドキュメント、pipeline_manager.py 冒頭docstring参照)。

"Edit" 相当の機能は公式 Z-Image-Edit チェックポイントが存在しない(2026-07時点)ため、
ZImageInpaintPipeline によるマスクインペイントとして実装する(旧実装と同じ設計、
白マスク画素 = 再生成する領域)。
"""
import os
import time
import uuid
from datetime import datetime
from typing import Optional

import torch
from PIL import Image

from core import progress as progress_mod
from families.z_image import pipeline as pipeline_mod
from families.z_image.pipeline import DEFAULT_GUIDANCE_SCALE, DEFAULT_STEPS

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _new_output_path(mode: str, ext: str = "png"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{mode}_{ts}_{uuid.uuid4().hex[:8]}.{ext}"
    path = os.path.join(OUTPUTS_DIR, name)
    return path, name


def make_generator(seed: Optional[int]):
    """抽出元: api_server.py _make_generator()。CPU generator を使う(旧実装・
    families/flux2/generate.py と同じ流儀)。"""
    if seed is None or int(seed) < 0:
        return None, None
    seed = int(seed)
    return torch.Generator(device="cpu").manual_seed(seed), seed


def _reset_vram_stats():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_vram_gb():
    return torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else None


def _progress_callback(mode: str, total_steps: int):
    """diffusers の callback_on_step_end に渡すクロージャ。ZImage系パイプラインは
    callback_on_step_end(self, step_index, timestep, callback_kwargs) で
    step_index を 0-origin で渡す(diffusers共通シグネチャ、他ファミリーと同じ)。
    """
    progress_mod.start_generating(mode, total_steps)

    def _cb(pipe, step_index, timestep, callback_kwargs):
        progress_mod.update_step(step_index + 1, total_steps)
        return callback_kwargs

    return _cb


def _pipeline_meta() -> dict:
    from families.z_image import state

    bps = state.base_pipeline_state
    return {
        "model": "z-image-turbo",
        "quant": bps.get("precision"),
        "offload_mode": bps.get("offload_mode"),
    }


def run_t2i(req: dict) -> dict:
    pipe = pipeline_mod.get_base_pipeline()
    generator, used_seed = make_generator(req.get("seed", -1))

    width = int(req.get("width") or 1024)
    height = int(req.get("height") or 1024)
    steps = int(req.get("steps") or DEFAULT_STEPS)
    guidance_scale = float(req.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
    negative_prompt = req.get("negative_prompt") or None

    _reset_vram_stats()
    t0 = time.time()
    result = pipe(
        prompt=req["prompt"],
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
        callback_on_step_end=_progress_callback("zimage_t2i", steps),
    )
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    image = result.images[0]
    out_path, out_name = _new_output_path("zimage_t2i")
    image.save(out_path)

    meta = {
        "mode": "zimage_t2i",
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", ""),
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": guidance_scale,
        "guidance_scale": guidance_scale,
        "seed": used_seed,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "image_url": f"/outputs/{out_name}",
    }
    meta.update(_pipeline_meta())
    return meta


def run_i2i(req: dict, src_image: Image.Image) -> dict:
    pipe = pipeline_mod.get_i2i_pipeline()
    generator, used_seed = make_generator(req.get("seed", -1))

    width = int(req.get("width") or 1024)
    height = int(req.get("height") or 1024)
    steps = int(req.get("steps") or DEFAULT_STEPS)
    guidance_scale = float(req.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
    strength = float(req.get("strength", 0.6))
    negative_prompt = req.get("negative_prompt") or None

    _reset_vram_stats()
    t0 = time.time()
    result = pipe(
        prompt=req["prompt"],
        negative_prompt=negative_prompt,
        image=src_image,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
        callback_on_step_end=_progress_callback("zimage_i2i", steps),
    )
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    image = result.images[0]
    out_path, out_name = _new_output_path("zimage_i2i")
    image.save(out_path)

    meta = {
        "mode": "zimage_i2i",
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", ""),
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": guidance_scale,
        "guidance_scale": guidance_scale,
        "strength": strength,
        "seed": used_seed,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "image_url": f"/outputs/{out_name}",
    }
    meta.update(_pipeline_meta())
    return meta


def run_inpaint(req: dict, src_image: Image.Image, mask_image: Image.Image) -> dict:
    """マスクインペイント(白 = 再生成する領域)。公式 Z-Image-Edit が無いための代替実装
    (families/z_image/generate.py モジュールdocstring参照)。
    """
    pipe = pipeline_mod.get_inpaint_pipeline()
    generator, used_seed = make_generator(req.get("seed", -1))

    width = int(req.get("width") or 1024)
    height = int(req.get("height") or 1024)
    steps = int(req.get("steps") or DEFAULT_STEPS)
    guidance_scale = float(req.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
    strength = float(req.get("strength", 1.0))
    negative_prompt = req.get("negative_prompt") or None

    _reset_vram_stats()
    t0 = time.time()
    result = pipe(
        prompt=req["prompt"],
        negative_prompt=negative_prompt,
        image=src_image,
        mask_image=mask_image,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
        callback_on_step_end=_progress_callback("zimage_inpaint", steps),
    )
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    image = result.images[0]
    out_path, out_name = _new_output_path("zimage_inpaint")
    image.save(out_path)

    meta = {
        "mode": "zimage_inpaint",
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", ""),
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": guidance_scale,
        "guidance_scale": guidance_scale,
        "strength": strength,
        "seed": used_seed,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "image_url": f"/outputs/{out_name}",
    }
    meta.update(_pipeline_meta())
    return meta
