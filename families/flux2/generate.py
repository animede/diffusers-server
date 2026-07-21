# -*- coding: utf-8 -*-
"""
FLUX.2 生成本体(T2I / I2I)。

抽出元: flux2_diffusers/app.py の generate_t2i() / generate_i2i()(Gradio版、行41-111)。
Gradio固有のUI引数・戻り値は持ち込まず、Qwen系(families/qwen_image/generate.py)と同じ
「画像は前処理済みで渡される・outputs/保存とメタデータ組み立てまでこのモジュールが担当」
という規約に合わせる。

T2I/I2Iは image=None / image=[...] で同一の Flux2Pipeline インスタンスを呼び分ける
(flux2_diffusers の設計をそのまま踏襲)。参照画像は最大10枚(MAX_REF_IMAGES)。
"""
import os
import time
import uuid
from datetime import datetime
from typing import Optional

import torch
from PIL import Image

from core import progress as progress_mod
from families.flux2 import state
from families.flux2.pipeline import get_pipeline

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

MAX_REF_IMAGES = 10


def _new_output_path(mode: str, ext: str = "png"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{mode}_{ts}_{uuid.uuid4().hex[:8]}.{ext}"
    path = os.path.join(OUTPUTS_DIR, name)
    return path, name


def make_generator(seed: Optional[int]):
    """抽出元: flux2_diffusers/app.py _make_generator()。CPU generator を使う点も無変更
    (Flux2Pipeline は内部でCPU上のgeneratorを期待する構成のため)。
    """
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
    """diffusers の callback_on_step_end に渡すクロージャ(FLUX.2 系。families/qwen_image/
    generate.py の _progress_callback() と同じ規約。callback_on_step_end(self, step_index,
    timestep, callback_kwargs) で step_index は 0-origin(pipeline_flux2*.py で実装確認済み)。
    """
    progress_mod.start_generating(mode, total_steps)

    def _cb(pipe, step_index, timestep, callback_kwargs):
        progress_mod.update_step(step_index + 1, total_steps)
        return callback_kwargs

    return _cb


def _pipeline_meta() -> dict:
    """2026-07-19: ecocoro廃止に伴い常に "flux2-dev" のメタ情報を返す。"""
    return {
        "model": "flux2-dev",
        "quant": state.pipeline_state.get("precision"),
        "offload_mode": state.pipeline_state.get("offload_mode"),
    }


def run_t2i(req: dict) -> dict:
    """抽出元: flux2_diffusers/app.py generate_t2i()。

    req["model"](任意): "dev"(既定)| None。get_pipeline() はバリデーション済みの
    値のみ受け付ける(2026-07-19、ecocoro廃止)。
    """
    pipe = get_pipeline(req.get("model"))
    generator, used_seed = make_generator(req.get("seed", -1))

    width = int(req["width"])
    height = int(req["height"])

    steps = int(req["steps"])
    _reset_vram_stats()
    t0 = time.time()
    result = pipe(
        prompt=req["prompt"],
        image=None,
        num_inference_steps=steps,
        guidance_scale=float(req["guidance_scale"]),
        width=width,
        height=height,
        generator=generator,
        callback_on_step_end=_progress_callback("flux2_t2i", steps),
    )
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    image = result.images[0]
    out_path, out_name = _new_output_path("flux2_t2i")
    image.save(out_path)

    meta = {
        "mode": "flux2_t2i",
        "prompt": req["prompt"],
        "width": width,
        "height": height,
        "steps": int(req["steps"]),
        "cfg": float(req["guidance_scale"]),
        "guidance_scale": float(req["guidance_scale"]),
        "seed": used_seed,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "image_url": f"/outputs/{out_name}",
    }
    meta.update(_pipeline_meta())
    return meta


def run_i2i(req: dict, ref_images: "list[Image.Image]") -> dict:
    """抽出元: flux2_diffusers/app.py generate_i2i()。ref_images は呼び出し側で
    convert("RGB") 済み・1〜MAX_REF_IMAGES枚のリストを渡すこと。

    req["model"](任意): "dev"(既定)| None(2026-07-19、ecocoro廃止)。
    """
    if not ref_images:
        raise ValueError("参照画像を1〜10枚アップロードしてください。")
    if len(ref_images) > MAX_REF_IMAGES:
        raise ValueError(f"参照画像は最大{MAX_REF_IMAGES}枚までです。")

    pipe = get_pipeline(req.get("model"))
    generator, used_seed = make_generator(req.get("seed", -1))

    width = int(req["width"])
    height = int(req["height"])

    steps = int(req["steps"])
    _reset_vram_stats()
    t0 = time.time()
    result = pipe(
        prompt=req["prompt"],
        image=ref_images,
        num_inference_steps=steps,
        guidance_scale=float(req["guidance_scale"]),
        width=width,
        height=height,
        generator=generator,
        callback_on_step_end=_progress_callback("flux2_i2i", steps),
    )
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    image = result.images[0]
    out_path, out_name = _new_output_path("flux2_i2i")
    image.save(out_path)

    meta = {
        "mode": "flux2_i2i",
        "prompt": req["prompt"],
        "width": width,
        "height": height,
        "steps": int(req["steps"]),
        "cfg": float(req["guidance_scale"]),
        "guidance_scale": float(req["guidance_scale"]),
        "seed": used_seed,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "image_url": f"/outputs/{out_name}",
        "num_reference_images": len(ref_images),
    }
    meta.update(_pipeline_meta())
    return meta
