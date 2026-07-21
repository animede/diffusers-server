# -*- coding: utf-8 -*-
"""
JoyAI-Image-Edit-Plus 生成本体(複数参照画像編集)。

抽出元: tools/joyai_regress.py の run_single_ref_edit() / run_multi_ref_compose()。
`pipe(images=[...], prompt=..., ...)` 規約(CLAUDE.md 2番の Qwen Edit と同じく、
単体画像でも必ずリストで渡す)。

text_encoder CPU退避(48GB対応): enable_text_encoder_cpu_offload(pipe) を一度だけ
有効化しておき、生成の都度 pipe(...) を呼ぶ直前に prepare_for_encode(pipe, te_offload)
を呼ぶ(transformer をCPUへ退避・text_encoder をGPUへ載せる)。JoyImageEditPlusPipeline の
`_execution_device` は「アルファベット順で最初に見つかる nn.Module」(text_encoder)の
現在地を返すため、text_encoder の GPU 復帰は encode_prompt_multiple_images 呼び出しの
中では手遅れ(families/joyai/pipeline.py の enable_text_encoder_cpu_offload() docstring
参照)。
"""
import os
import time
import uuid
from datetime import datetime
from typing import List, Optional

import torch
from PIL import Image

from core import progress as progress_mod
from families.joyai import pipeline as pipeline_mod
from families.joyai.pipeline import DEFAULT_GUIDANCE_SCALE, DEFAULT_STEPS, MAX_REF_IMAGES
from families.joyai.runtime import should_offload_te

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def _new_output_path(mode: str, ext: str = "png"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{mode}_{ts}_{uuid.uuid4().hex[:8]}.{ext}"
    path = os.path.join(OUTPUTS_DIR, name)
    return path, name


def make_generator(seed: Optional[int]):
    if seed is None or int(seed) < 0:
        return None, None
    seed = int(seed)
    return torch.Generator(device="cuda").manual_seed(seed), seed


def _reset_vram_stats():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_vram_gb():
    return torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else None


def _progress_callback(mode: str, total_steps: int):
    progress_mod.start_generating(mode, total_steps)

    def _cb(pipe, step_index, timestep, callback_kwargs):
        progress_mod.update_step(step_index + 1, total_steps)
        return callback_kwargs

    return _cb


def _pipeline_meta() -> dict:
    from families.joyai import state

    ps = state.pipeline_state
    return {
        "model": "joyai-edit-plus",
        "te_offload": ps.get("te_offload"),
    }


def run_edit(req: dict, src_images: List[Image.Image]) -> dict:
    """単一/複数参照画像編集(Edit)。src_images は1〜3枚(app.py 側で制約)。"""
    pipe = pipeline_mod.get_pipeline()
    generator, used_seed = make_generator(req.get("seed", -1))

    steps = int(req.get("steps") or DEFAULT_STEPS)
    guidance_scale = float(req.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
    negative_prompt = req.get("negative_prompt") or None
    width = req.get("width")
    height = req.get("height")
    width = int(width) if width else None
    height = int(height) if height else None

    te_offload = should_offload_te()
    if te_offload:
        pipeline_mod.enable_text_encoder_cpu_offload(pipe)
    else:
        pipeline_mod.disable_text_encoder_cpu_offload(pipe)

    _reset_vram_stats()
    # 重要(実機デバッグで特定、families/joyai/pipeline.py の
    # enable_text_encoder_cpu_offload() docstring参照): JoyImageEditPlusPipeline は
    # `_execution_device` を `__call__` の**冒頭**(encode_prompt_multiple_images を
    # 呼ぶより前)で解決し、これは「アルファベット順で最初に見つかる nn.Module」
    # (text_encoder)の現在地を返す。そのため text_encoder の GPU 復帰(と
    # transformer の CPU 退避)は pipe(...) を呼ぶ**直前**に行う必要がある
    # (encode_prompt_multiple_images 呼び出しの中では手遅れ)。
    pipeline_mod.prepare_for_encode(pipe, te_offload)

    t0 = time.time()
    result = pipe(
        images=src_images,
        prompt=req["prompt"],
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        callback_on_step_end=_progress_callback("joyai_edit", steps),
    )
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    image = result.images[0]
    out_path, out_name = _new_output_path("joyai_edit")
    image.save(out_path)

    meta = {
        "mode": "joyai_edit",
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", ""),
        "num_ref_images": len(src_images),
        "width": image.width,
        "height": image.height,
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
