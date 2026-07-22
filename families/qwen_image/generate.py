# -*- coding: utf-8 -*-
"""
生成本体(T2I / I2I / Edit / ControlNet / Inpaint / Canny / Layered)。

抽出元: Ｑwenimage-edit-diffusers/app.py の各エンドポイント関数の中身
(pipeline取得後〜画像保存前まで)。画像の保存・ファイル名生成・前処理・レスポンス
組み立ては元実装のロジックをそのまま踏襲するが、HTTP層(FastAPI固有の型・例外)は
families 側に持ち込まず、app.py 側に残す(REBUILD_PLAN: "app.py は薄く保つ")。

この関数群は「画像(PIL.Image 等)は既に読み込み・前処理済み」という前提で呼ばれる。
outputs/ への保存・メタデータ dict の組み立てまでをこのモジュールが担当し、
app.py はアップロードファイルの読み込みと HTTPException 変換のみを行う。

生成の排他制御(GPU 同時1件)は呼び出し側(core.registry.FamilyRegistry.generate()、
gpu.generation_scope() 経由)が担うため、ここでは torch.cuda.empty_cache() /
reset_peak_memory_stats() の「生成直前」呼び出しのみ行う(元実装 app.py の各エンドポイントに
あった処理をそのまま踏襲)。
"""
import os
import time
import uuid
from datetime import datetime
from typing import Optional

import torch
from PIL import Image

from core import progress as progress_mod
from core.optimize import (
    disable_text_encoder_cpu_offload,
    enable_text_encoder_cpu_offload,
    ensure_text_encoder_on_gpu,
    should_offload_edit_text_encoder,
)

from families.qwen_image import controlnet as controlnet_mod
from families.qwen_image import edit as edit_mod
from families.qwen_image import edit_angles as edit_angles_mod
from families.qwen_image import edit_angles_bf16 as edit_angles_bf16_mod
from families.qwen_image import edit_angles_bf16group as edit_angles_bf16group_mod
from families.qwen_image import layered as layered_mod
from families.qwen_image import state, t2i as t2i_mod
from families.qwen_image.paths import (
    LAYERED_DEFAULT_CFG,
    LAYERED_DEFAULT_LAYERS,
    LAYERED_DEFAULT_RESOLUTION,
    LAYERED_DEFAULT_STEPS,
    LAYERED_LIGHTNING_CFG,
    LAYERED_LIGHTNING_SHIFT,
    LAYERED_LIGHTNING_STEPS,
    LAYERED_VALID_RESOLUTIONS,
    LIGHTNING_SHIFT,
    LIGHTNING_TRUE_CFG_SCALE,
    NEGATIVE_PROMPT_DEFAULT,
    is_2512_model,
)

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

MAX_EDIT_IMAGES = 3


def _round16(x: int) -> int:
    return max(16, round(x / 16) * 16)


def _new_output_path(mode: str, ext: str = "png"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{mode}_{ts}_{uuid.uuid4().hex[:8]}.{ext}"
    path = os.path.join(OUTPUTS_DIR, name)
    return path, name


def preprocess_image(image: Image.Image, target_pixels: int = 1024 * 1024) -> Image.Image:
    """ComfyUI の ImageScaleToTotalPixels 相当。アスペクト比維持で総画素数を揃え、16の倍数に丸める。

    抽出元: app.py _preprocess_image()。
    """
    image = image.convert("RGB")
    w, h = image.size
    if w <= 0 or h <= 0:
        raise ValueError("invalid image size")
    scale = (target_pixels / (w * h)) ** 0.5
    new_w = _round16(w * scale)
    new_h = _round16(h * scale)
    if (new_w, new_h) != (w, h):
        image = image.resize((new_w, new_h), Image.LANCZOS)
    return image


def make_generator(seed: Optional[int]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if seed is None or seed < 0:
        return None, None
    return torch.Generator(device=device).manual_seed(seed), seed


def t2i_model_meta() -> dict:
    """現在アクティブな T2I モデル(DS_T2I_MODEL)に応じたメタデータ断片を返す。

    2026-07-18変更: 2512 も無印と同じ t2i_group(fp8-lightning fuse)になったため、
    quant は常に t2i_group["quant"] を返す(旧NF4の "nf4" 特別扱いは廃止)。

    抽出元: app.py _t2i_model_meta()。
    """
    model = "2512" if is_2512_model(state.get_runtime_config().t2i_model) else "qwen-image"
    return {"model": model, "quant": state.t2i_group["quant"]}


def _reset_vram_stats():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_vram_gb():
    return torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else None


def _progress_callback(mode: str, total_steps: int, extra: "dict | None" = None):
    """diffusers の callback_on_step_end に渡すクロージャを作る。

    全 Qwen-Image 系パイプライン(pipeline_qwenimage*.py)は
    callback_on_step_end(self, step_index, timestep, callback_kwargs) の
    シグネチャで step_index を 0-origin で渡してくる(実装確認済み)。
    dict を返す必要がある(callback_kwargs をそのまま素通しする)。
    """
    progress_mod.start_generating(mode, total_steps, extra)

    def _cb(pipe, step_index, timestep, callback_kwargs):
        progress_mod.update_step(step_index + 1, total_steps)
        return callback_kwargs

    return _cb


def run_t2i(req: dict) -> dict:
    """抽出元: app.py generate_t2i()。

    req["model"](タスク1、任意): "2512"(既定) | "qwen-image"(旧値 "2512-4bit" は
    後方互換で "2512" 扱い)。family.load() で既に
    切替済みのはずだが、get_t2i_pipeline() は model=None なら現在のランタイム設定を
    そのまま使う冪等な実装のため、ここで再度渡しても二重切替にはならない。
    """
    pipe = t2i_mod.get_t2i_pipeline(req.get("model"))

    # 2026-07-18追加(I2IのOOM修正に伴う整合性対応): T2Iは I2I/Edit系と
    # state.shared["text_encoder"] を共有している(2512/無印とも同じ共有構造)。
    # run_i2i() が TE を CPU 退避したまま戻ってきた直後に run_t2i() が呼ばれると、
    # QwenImagePipeline.__call__() も encode_prompt より前に self._execution_device を
    # 確定させる実装のため、text_encoder が CPU に残っていると device mismatch になりうる
    # (run_edit() と同じ罠、core.optimize.ensure_text_encoder_on_gpu のdocstring参照)。
    ensure_text_encoder_on_gpu(pipe)

    steps = req["steps"]
    cfg = req["cfg"]
    shift = req.get("shift")
    lightning_active = False
    lightning_unavailable = False
    if req.get("lightning"):
        req_steps = steps if steps in (4, 8) else 4
        lightning_active = t2i_mod.set_t2i_lightning(pipe, req_steps)
        if lightning_active:
            steps = req_steps
            cfg = LIGHTNING_TRUE_CFG_SCALE
            shift = shift if shift is not None else LIGHTNING_SHIFT
        else:
            lightning_unavailable = True
            if steps <= 8:
                steps = 30
            if cfg <= 1.5:
                cfg = 4.0
    else:
        t2i_mod.set_t2i_lightning(pipe, -1)

    edit_mod.set_scheduler_shift(pipe.scheduler, shift)

    generator, used_seed = make_generator(req.get("seed", -1))
    width = _round16(req["width"])
    height = _round16(req["height"])

    # TE(text_encoder) CPU退避(2026-07-18追加、run_i2i()/run_edit() と同じ方式・同じ判定関数)。
    # fp8-lightning は既定解像度1328²でピーク約43GBに達し、48GB専有でも他プロセス
    # (例: llama-server がGPU 0に~1.5GB確保)と同居すると VAE decode の一時ピークで
    # OOM する(実機で発生)。2512 も無印と同じ fp8-lightning 構造のため同様に適用する。
    if should_offload_edit_text_encoder(width, height):
        enable_text_encoder_cpu_offload(pipe)
        ensure_text_encoder_on_gpu(pipe)
    else:
        disable_text_encoder_cpu_offload(pipe)
        ensure_text_encoder_on_gpu(pipe)

    _reset_vram_stats()
    t0 = time.time()
    result = pipe(
        prompt=req["prompt"],
        negative_prompt=req.get("negative_prompt") or None,
        width=width,
        height=height,
        num_inference_steps=steps,
        true_cfg_scale=cfg,
        generator=generator,
        callback_on_step_end=_progress_callback("t2i", steps),
    )
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    image = result.images[0]
    out_path, out_name = _new_output_path("t2i")
    image.save(out_path)

    meta = {
        "mode": "t2i",
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", NEGATIVE_PROMPT_DEFAULT),
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": cfg,
        "shift": shift,
        "seed": used_seed,
        "lightning": lightning_active,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "image_url": f"/outputs/{out_name}",
        "lightning_requested": req.get("lightning", False),
        "lightning_unavailable_reason": (
            "量子化中はLoRA非対応のためフォールバックしました(steps/cfgを通常品質側に自動調整)"
            if lightning_unavailable else None
        ),
    }
    meta.update(t2i_model_meta())
    return meta


def run_i2i(req: dict, src_image: Image.Image) -> dict:
    """抽出元: app.py generate_i2i()。src_image は preprocess_image() 済みを渡すこと。

    req["model"](2026-07-18更新、任意): "2512"(既定) | "qwen-image"。2512 が無印と
    同じ transformer 共有構造になったため、I2I もモデル切替に対応した(旧NF4は
    Img2Img クラスを持たない自己完結パイプラインのため I2I 非対応で 400 を返していた)。

    TE(text_encoder) CPU退避(2026-07-18、既存バグ修正: CLAUDE.md 26番参照)。
    既定構成(DS_QUANT=fp8-lightning)のクリーンな状態から `/api/i2i` を単独で呼ぶと、
    denoiseまでは成功するが `pipe.vae.decode()` 内の conv3d で CUDA OOM することを
    実機再現・特定した(トレースバック確認: allocated 38.01GB / free 1.84GB で
    "Tried to allocate 5.06 GiB"。run_edit() が持つ text_encoder CPU退避が
    run_i2i() には未適用だったことが原因)。I2Iの入力画像は preprocess_image() で
    ほぼ1024²相当に揃えられており run_edit() と同条件のため、同じ
    should_offload_edit_text_encoder()(DS_EDIT_TE_OFFLOAD、既定auto)をそのまま
    再利用する(I2I専用の環境変数は増やさない)。GPU復帰は run_edit() と同じ理由
    (QwenImageImg2ImgPipeline.__call__() も encode_prompt より前に
    self._execution_device を確定させる、core.optimize.ensure_text_encoder_on_gpu の
    docstring参照)で pipe(...) 呼び出し**直前**に行う。
    """
    pipe = t2i_mod.get_i2i_pipeline(req.get("model"))

    steps_eff = req["steps"]
    cfg_eff = req["cfg"]
    shift_eff = req.get("shift")
    lightning_active = False
    lightning_unavailable = False
    if req.get("lightning"):
        req_steps = steps_eff if steps_eff in (4, 8) else 4
        lightning_active = t2i_mod.set_t2i_lightning(pipe, req_steps)
        if lightning_active:
            steps_eff = req_steps
            cfg_eff = LIGHTNING_TRUE_CFG_SCALE
            shift_eff = shift_eff if shift_eff is not None else LIGHTNING_SHIFT
        else:
            lightning_unavailable = True
            if steps_eff <= 8:
                steps_eff = 30
            if cfg_eff <= 1.5:
                cfg_eff = 4.0
    else:
        t2i_mod.set_t2i_lightning(pipe, -1)

    edit_mod.set_scheduler_shift(pipe.scheduler, shift_eff)

    generator, used_seed = make_generator(req.get("seed", -1))

    # TE(text_encoder) CPU退避(run_edit() と同じ方式・同じ判定関数。上記docstring参照)。
    # src_image は preprocess_image() 済みのため実際に使われる解像度がそのまま分かる。
    src_w, src_h = src_image.size
    te_offload = should_offload_edit_text_encoder(src_w, src_h)
    if te_offload:
        enable_text_encoder_cpu_offload(pipe)
        ensure_text_encoder_on_gpu(pipe)
    else:
        disable_text_encoder_cpu_offload(pipe)
        # text_encoder は t2i_pipe / i2i_pipe / edit系パイプライン間で同一インスタンスを
        # 共有している(CLAUDE.md 8番)。disable_text_encoder_cpu_offload() は「この pipe
        # インスタンス自身が前回有効化していた場合のみ」GPUへ戻す実装のため、直前に
        # 別のパイプライン経由でCPU退避されたケースを取りこぼす可能性がある。保険として
        # 明示的に確認する。
        ensure_text_encoder_on_gpu(pipe)

    _reset_vram_stats()
    t0 = time.time()
    # 注意: strength<1.0 の場合、実際に走るdenoiseステップ数は
    # round(steps_eff * strength) 相当(img2imgの一般的な挙動)で steps_eff より
    # 少なくなりうる。callback の step_index はその実ステップ数基準で 0-origin なため、
    # total_steps に steps_eff をそのまま渡すとバーが100%未満で止まる場合があるが、
    # 完了時に finish() が呼ばれるため実害は軽微(進捗バー消滅で完了と分かる)。
    result = pipe(
        prompt=req["prompt"],
        negative_prompt=req.get("negative_prompt") or None,
        image=src_image,
        strength=req.get("strength", 0.6),
        num_inference_steps=steps_eff,
        true_cfg_scale=cfg_eff,
        generator=generator,
        callback_on_step_end=_progress_callback("i2i", steps_eff),
    )
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    out_image = result.images[0]
    out_path, out_name = _new_output_path("i2i")
    out_image.save(out_path)

    meta = {
        "mode": "i2i",
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", NEGATIVE_PROMPT_DEFAULT),
        "strength": req.get("strength", 0.6),
        "steps": steps_eff,
        "cfg": cfg_eff,
        "shift": shift_eff,
        "seed": used_seed,
        "lightning": lightning_active,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "image_url": f"/outputs/{out_name}",
        "input_size": list(src_image.size),
        "te_offload": te_offload,
        "quant": state.t2i_group["quant"],
        "lightning_requested": req.get("lightning", False),
        "lightning_unavailable_reason": (
            "GGUF量子化中はLoRA非対応のためフォールバックしました(steps/cfgを通常品質側に自動調整)"
            if lightning_unavailable else None
        ),
    }
    return meta


def run_edit(req: dict, src_images: list) -> dict:
    """抽出元: app.py generate_edit()。src_images は preprocess_image() 済みリスト。

    Phase 1 申し送り事項(REBUILD_PLAN §4 Phase 2 タスク4)への対処: 48GB専有では
    Edit の標準解像度(参照画像から自動推定される1024²相当)で VAE decode の一時ピークが
    ~5GBほど乗り、OOMすることを実機確認済み(height=640,width=640 明示で成功: 39GB)。
    そのため任意パラメータ height / width を追加した。未指定(None)時は旧挙動のまま
    (QwenImageEditPlusPipeline が参照画像から自動推定する解像度を使う)。
    """
    if not src_images:
        raise ValueError("参照画像を1〜3枚アップロードしてください。")
    if len(src_images) > MAX_EDIT_IMAGES:
        raise ValueError(f"参照画像は最大{MAX_EDIT_IMAGES}枚までです。")

    pipe = edit_mod.get_edit_pipeline()

    steps_eff = req["steps"]
    cfg_eff = req["cfg"]
    shift_eff = req.get("shift")
    lightning = req.get("lightning", True)
    lightning_active = edit_mod.set_edit_lightning(pipe, lightning)
    lightning_unavailable = lightning and not lightning_active
    if lightning and lightning_active:
        if steps_eff <= 0 or steps_eff > 12:
            steps_eff = 4
        cfg_eff = LIGHTNING_TRUE_CFG_SCALE
        shift_eff = shift_eff if shift_eff is not None else LIGHTNING_SHIFT
    elif lightning_unavailable:
        if steps_eff <= 8:
            steps_eff = 30
        if cfg_eff <= 1.5:
            cfg_eff = 4.0

    edit_mod.set_scheduler_shift(pipe.scheduler, shift_eff)

    generator, used_seed = make_generator(req.get("seed", -1))

    # 任意の解像度指定(OOM回避用)。未指定なら None のままパイプラインに渡し、
    # 参照画像からの自動推定(旧挙動)に任せる。
    width = _round16(req["width"]) if req.get("width") else None
    height = _round16(req["height"]) if req.get("height") else None

    # TE(text_encoder) CPU退避(DS_EDIT_TE_OFFLOAD、既定 auto)。プロンプトエンコード後に
    # text_encoder をCPUへ移し、denoise/VAE decodeの間VRAMを空けておくことで、48GB専有でも
    # 1024²相当(height/width未指定時の自動推定を含む)がOOMなく収まるようにする
    # (CLAUDE.md 3番/22番、core.optimize.enable_text_encoder_cpu_offload 参照)。
    # 重要: text_encoder のGPU復帰は pipe(...) を呼ぶ直前に明示的に行うこと
    # (ensure_text_encoder_on_gpu())。QwenImageEditPlusPipeline.__call__() は
    # encode_prompt を呼ぶより前に self._execution_device(= text_encoder 等の現在地から
    # 解決される)を確定させるため、encode_prompt 側で GPU へ戻すのでは手遅れ
    # (input_ids は cpu ・ embed_tokens.weight は cuda という食い違いでdevice mismatch
    # エラーになることを実機デバッグで特定した。core.optimize.ensure_text_encoder_on_gpu
    # のdocstring参照)。CPUへ戻すタイミングは encode_prompt 完了直後のまま(ラップ内)。
    te_offload = should_offload_edit_text_encoder(width, height)
    if te_offload:
        enable_text_encoder_cpu_offload(pipe)
        ensure_text_encoder_on_gpu(pipe)
    else:
        disable_text_encoder_cpu_offload(pipe)
        # text_encoder は t2i_pipe / i2i_pipe / edit系パイプライン間で同一インスタンスを
        # 共有している(CLAUDE.md 8番)。disable_text_encoder_cpu_offload() は「この pipe
        # インスタンス自身が前回有効化していた場合のみ」GPUへ戻す実装のため、直前に
        # 別のパイプライン(例: run_i2i())経由でCPU退避されたケースを取りこぼす
        # 可能性がある(2026-07-18、I2IのOOM修正に伴う整合性対応)。保険として明示的に
        # 確認する。
        ensure_text_encoder_on_gpu(pipe)

    _reset_vram_stats()
    t0 = time.time()
    # 重要: QwenImageEditPlusPipeline には必ずリスト(image=[img, ...])で渡すこと。
    # 単体画像で渡すと Plus 条件付けが効かずキャラクター同一性が崩れる。
    result = pipe(
        image=src_images,
        prompt=req["prompt"],
        negative_prompt=req.get("negative_prompt") or None,
        num_inference_steps=steps_eff,
        true_cfg_scale=cfg_eff,
        width=width,
        height=height,
        generator=generator,
        callback_on_step_end=_progress_callback(
            "charsheet" if req.get("_progress_extra") else "edit",
            steps_eff,
            req.get("_progress_extra"),
        ),
    )
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    out_image = result.images[0]
    out_path, out_name = _new_output_path("edit")
    out_image.save(out_path)

    meta = {
        "mode": "edit",
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", NEGATIVE_PROMPT_DEFAULT),
        "steps": steps_eff,
        "cfg": cfg_eff,
        "shift": shift_eff,
        "width": width,
        "height": height,
        "seed": used_seed,
        "lightning": lightning_active,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "image_url": f"/outputs/{out_name}",
        "num_reference_images": len(src_images),
        "edit_transformer_fallback": state.edit_group["fallback"],
        "quant": state.edit_group["quant"],
        "lightning_requested": lightning,
        "lightning_unavailable_reason": (
            "GGUF量子化中はLoRA非対応のためフォールバックしました(steps/cfgを通常品質側に自動調整)"
            if lightning_unavailable else None
        ),
        "te_offload": te_offload,
    }
    return meta


def run_edit_angles(req: dict, src_images: list) -> dict:
    """charsheet 専用: Edit(fp8-lightning-angles、Multiple-angles LoRA fuse済み)での生成。

    run_edit() とのちがい: このグループは常にLightning + Multiple-anglesが重みに
    fuse済みのため、set_edit_lightning() のようなトグルは存在しない(常にLightning
    パラメータ(steps=4, cfg=1.0, shift=LIGHTNING_SHIFT)を使う)。height/width は
    呼び出し側(apps/charsheet)が640x640等を明示する前提(未指定なら自動推定、
    通常のEditと同じOOMリスクがあるため charsheet 側では必ず指定すること)。
    """
    if not src_images:
        raise ValueError("参照画像を1〜3枚アップロードしてください。")
    if len(src_images) > MAX_EDIT_IMAGES:
        raise ValueError(f"参照画像は最大{MAX_EDIT_IMAGES}枚までです。")

    pipe = edit_angles_mod.get_edit_angles_pipeline()

    steps_eff = req["steps"] if req.get("steps") and 0 < req["steps"] <= 12 else 4
    cfg_eff = LIGHTNING_TRUE_CFG_SCALE
    shift_eff = req.get("shift") if req.get("shift") is not None else LIGHTNING_SHIFT

    edit_angles_mod.set_scheduler_shift(pipe.scheduler, shift_eff)

    generator, used_seed = make_generator(req.get("seed", -1))

    width = _round16(req["width"]) if req.get("width") else None
    height = _round16(req["height"]) if req.get("height") else None

    # TE CPU退避(run_edit() と同じ方式・同じ判定関数。charsheet が CHARSHEET_EDIT_SIZE=1024
    # 既定で呼ぶ場合、640*640閾値を超えるため auto で実質 on になる)。GPU復帰を
    # pipe(...) 呼び出し直前に明示するのが重要な理由は run_edit() のコメント
    # (core.optimize.ensure_text_encoder_on_gpu のdocstring)参照。
    te_offload = should_offload_edit_text_encoder(width, height)
    if te_offload:
        enable_text_encoder_cpu_offload(pipe)
        ensure_text_encoder_on_gpu(pipe)
    else:
        disable_text_encoder_cpu_offload(pipe)
        # 保険(2026-07-18追加): text_encoder は t2i/i2i/edit系と共有インスタンスのため、
        # 別パイプライン経由でCPU退避されたケースを取りこぼさないよう明示確認する
        # (run_edit() の同種コメント参照)。
        ensure_text_encoder_on_gpu(pipe)

    _reset_vram_stats()
    t0 = time.time()
    # 重要: QwenImageEditPlusPipeline には必ずリスト(image=[img, ...])で渡すこと。
    # 単体画像で渡すと Plus 条件付けが効かずキャラクター同一性が崩れる。
    result = pipe(
        image=src_images,
        prompt=req["prompt"],
        negative_prompt=req.get("negative_prompt") or None,
        num_inference_steps=steps_eff,
        true_cfg_scale=cfg_eff,
        width=width,
        height=height,
        generator=generator,
        callback_on_step_end=_progress_callback(
            "charsheet", steps_eff, req.get("_progress_extra")
        ),
    )
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    out_image = result.images[0]
    out_path, out_name = _new_output_path("edit_angles")
    out_image.save(out_path)

    meta = {
        "mode": "edit_angles",
        "te_offload": te_offload,
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", NEGATIVE_PROMPT_DEFAULT),
        "steps": steps_eff,
        "cfg": cfg_eff,
        "shift": shift_eff,
        "width": width,
        "height": height,
        "seed": used_seed,
        "lightning": True,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "image_url": f"/outputs/{out_name}",
        "num_reference_images": len(src_images),
        "quant": state.edit_angles_group["quant"],
    }
    return meta


def run_edit_angles_bf16(req: dict, src_images: list) -> dict:
    """charsheet 専用: Edit(fp8-base-adapters、Lightning + Multiple-angles を adapter として
    weight=1.0 で同時適用)での生成(旧 /home/animede/charsheet 方式)。

    run_edit_angles() とのちがい: このグループは transformer をfuseせず、ベースのみ
    fp8ストレージ化した上に LoRA adapter を2本(lightning, angles)重ねて適用する
    (families/qwen_image/edit_angles_bf16.py 参照)。fp8化によりtransformerは
    「小さいtransformer」として扱えるため、TEオフロードは run_edit()/run_edit_angles()
    と全く同じ通常パターン(text_encoderのみのCPU退避)で足りる。**transformer自体を
    CPUへ退避する実装は行わない**(過去にホストRAMスワップ暴走でシステムフリーズを
    起こした事故があるため、CLAUDE.md 17番・このモジュール上部の事故の教訓参照)。
    生成パラメータ(steps=4固定、cfg=LIGHTNING_TRUE_CFG_SCALE、shift=LIGHTNING_SHIFT)は
    run_edit_angles() と同一(fp8-lightning-angles版と同じLightning前提のため)。
    """
    if not src_images:
        raise ValueError("参照画像を1〜3枚アップロードしてください。")
    if len(src_images) > MAX_EDIT_IMAGES:
        raise ValueError(f"参照画像は最大{MAX_EDIT_IMAGES}枚までです。")

    pipe = edit_angles_bf16_mod.get_edit_angles_bf16_pipeline()

    steps_eff = req["steps"] if req.get("steps") and 0 < req["steps"] <= 12 else 4
    cfg_eff = LIGHTNING_TRUE_CFG_SCALE
    shift_eff = req.get("shift") if req.get("shift") is not None else LIGHTNING_SHIFT

    edit_angles_bf16_mod.set_scheduler_shift(pipe.scheduler, shift_eff)

    generator, used_seed = make_generator(req.get("seed", -1))

    width = _round16(req["width"]) if req.get("width") else None
    height = _round16(req["height"]) if req.get("height") else None

    # TE CPU退避(run_edit()/run_edit_angles() と全く同じ方式・同じ判定関数)。
    # fp8-base-adapters は transformer をfp8化済み(~19-20GB)のため「小さいtransformer」
    # として扱え、text_encoderのみのCPU退避で十分収まる(transformer自体のCPU退避は
    # 一切行わない。過去にホストRAMスワップ暴走でシステムフリーズを起こした事故が
    # あるため、このモジュール上部のdocstring参照)。
    te_offload = should_offload_edit_text_encoder(width, height)
    if te_offload:
        enable_text_encoder_cpu_offload(pipe)
        ensure_text_encoder_on_gpu(pipe)
    else:
        disable_text_encoder_cpu_offload(pipe)
        # 保険(run_edit()/run_edit_angles() と同じ理由): text_encoder は t2i/i2i/edit系と
        # 共有インスタンスのため、別パイプライン経由でCPU退避されたケースを取りこぼさない
        # よう明示確認する。
        ensure_text_encoder_on_gpu(pipe)

    _reset_vram_stats()
    t0 = time.time()
    # 重要: QwenImageEditPlusPipeline には必ずリスト(image=[img, ...])で渡すこと。
    # 単体画像で渡すと Plus 条件付けが効かずキャラクター同一性が崩れる。
    result = pipe(
        image=src_images,
        prompt=req["prompt"],
        negative_prompt=req.get("negative_prompt") or None,
        num_inference_steps=steps_eff,
        true_cfg_scale=cfg_eff,
        width=width,
        height=height,
        generator=generator,
        callback_on_step_end=_progress_callback(
            "charsheet", steps_eff, req.get("_progress_extra")
        ),
    )
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    out_image = result.images[0]
    out_path, out_name = _new_output_path("edit_angles_bf16")
    out_image.save(out_path)

    meta = {
        "mode": "edit_angles_bf16",
        "te_offload": te_offload,
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", NEGATIVE_PROMPT_DEFAULT),
        "steps": steps_eff,
        "cfg": cfg_eff,
        "shift": shift_eff,
        "width": width,
        "height": height,
        "seed": used_seed,
        "lightning": True,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "image_url": f"/outputs/{out_name}",
        "num_reference_images": len(src_images),
        "quant": state.edit_angles_bf16_group["quant"],
    }
    return meta


def run_edit_angles_bf16group(req: dict, src_images: list) -> dict:
    """charsheet 専用: Edit(bf16-group、旧 /home/animede/charsheet 完全再現、CLAUDE.md 33番、
    新既定)での生成。

    run_edit_angles_bf16() とのちがい: このグループは transformer をfuseもfp8化も
    行わず、bf16のまま block-level group offloading(families/qwen_image/
    edit_angles_bf16group.py)で GPU/CPU を入れ替える。旧 /home/animede/charsheet/
    pipeline.py の "group" モードは text_encoder を常時GPU常駐のままにする設計
    (offload_all_components=False、_apply_group_offload_to_transformer() 行323-330)
    だったため、本方式では DS_EDIT_TE_OFFLOAD による text_encoder CPU退避スワップは
    **使わない**(旧実装との整合を優先。group offload で transformer 側のVRAMは既に
    賄われているため、text_encoderまで退避する必要が無い、旧実装の設計判断をそのまま
    踏襲する)。生成パラメータ(steps=4固定、cfg=LIGHTNING_TRUE_CFG_SCALE、
    shift=LIGHTNING_SHIFT)は run_edit_angles()/run_edit_angles_bf16() と同一。
    """
    if not src_images:
        raise ValueError("参照画像を1〜3枚アップロードしてください。")
    if len(src_images) > MAX_EDIT_IMAGES:
        raise ValueError(f"参照画像は最大{MAX_EDIT_IMAGES}枚までです。")

    pipe = edit_angles_bf16group_mod.get_edit_angles_bf16group_pipeline()

    steps_eff = req["steps"] if req.get("steps") and 0 < req["steps"] <= 12 else 4
    cfg_eff = LIGHTNING_TRUE_CFG_SCALE
    shift_eff = req.get("shift") if req.get("shift") is not None else LIGHTNING_SHIFT

    edit_angles_bf16group_mod.set_scheduler_shift(pipe.scheduler, shift_eff)

    generator, used_seed = make_generator(req.get("seed", -1))

    width = _round16(req["width"]) if req.get("width") else None
    height = _round16(req["height"]) if req.get("height") else None

    # text_encoder は常時GPU常駐のまま(旧 pipeline.py "group" モードと同じ、上記docstring
    # 参照)。DS_EDIT_TE_OFFLOAD による CPU退避スワップは使わない
    # (block-level group offload が transformer 側のVRAMを既に賄っているため不要)。
    ensure_text_encoder_on_gpu(pipe)

    _reset_vram_stats()
    t0 = time.time()
    # 重要: QwenImageEditPlusPipeline には必ずリスト(image=[img, ...])で渡すこと。
    # 単体画像で渡すと Plus 条件付けが効かずキャラクター同一性が崩れる。
    result = pipe(
        image=src_images,
        prompt=req["prompt"],
        negative_prompt=req.get("negative_prompt") or None,
        num_inference_steps=steps_eff,
        true_cfg_scale=cfg_eff,
        width=width,
        height=height,
        generator=generator,
        callback_on_step_end=_progress_callback(
            "charsheet", steps_eff, req.get("_progress_extra")
        ),
    )
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    out_image = result.images[0]
    out_path, out_name = _new_output_path("edit_angles_bf16group")
    out_image.save(out_path)

    meta = {
        "mode": "edit_angles_bf16group",
        "te_offload": False,
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", NEGATIVE_PROMPT_DEFAULT),
        "steps": steps_eff,
        "cfg": cfg_eff,
        "shift": shift_eff,
        "width": width,
        "height": height,
        "seed": used_seed,
        "lightning": True,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "image_url": f"/outputs/{out_name}",
        "num_reference_images": len(src_images),
        "quant": state.edit_angles_bf16group_group["quant"],
        "group_offload": state.edit_angles_bf16group_group["group_offload_config"],
    }
    return meta


def run_controlnet(req: dict, ctrl_img: Image.Image) -> dict:
    """抽出元: app.py generate_controlnet()。ctrl_img は未リサイズ(このモジュール内でリサイズする)。

    TE(text_encoder) device mismatch 対策(2026-07-21、実機報告に基づき修正: CLAUDE.md
    参照)。ControlNetパイプライン(controlnet.py の get_controlnet_pipeline())は
    T2I/I2I/Editと同一の共有text_encoderインスタンス(state.shared["text_encoder"])を
    使う(CLAUDE.md 8番)。run_i2i()/run_edit() 等はTE CPU退避(should_offload_edit_
    text_encoder、既定auto)により、pipe(...) 完了後にこの共有インスタンスをCPUへ
    退避したまま返すことがある。run_controlnet() 自身はTE退避を行わないため、この
    保険呼び出しが無いと直前がi2i/edit等だった場合に text_encoder がCPUに残ったまま
    pipe(...) を呼んでしまい、`RuntimeError: Expected all tensors to be on the same
    device` になる(_execution_device 解決の罠は core.optimize.ensure_text_encoder_on_gpu
    のdocstring参照。run_t2i()/run_i2i()/run_edit() 等の全run_*関数が持つ「保険としての
    ensure_text_encoder_on_gpu() 呼び出し」パターンと同じもの)。
    """
    w = _round16(req["width"]) if req.get("width") else _round16(ctrl_img.width)
    h = _round16(req["height"]) if req.get("height") else _round16(ctrl_img.height)
    if (w, h) != ctrl_img.size:
        ctrl_img = ctrl_img.resize((w, h), Image.LANCZOS)

    pipe = controlnet_mod.get_controlnet_pipeline()
    ensure_text_encoder_on_gpu(pipe)
    generator, used_seed = make_generator(req.get("seed", -1))

    _reset_vram_stats()
    t0 = time.time()
    result = pipe(
        prompt=req["prompt"],
        negative_prompt=req.get("negative_prompt") or " ",
        control_image=ctrl_img,
        controlnet_conditioning_scale=req.get("controlnet_conditioning_scale", 1.0),
        width=w,
        height=h,
        num_inference_steps=req["steps"],
        true_cfg_scale=req["cfg"],
        generator=generator,
        callback_on_step_end=_progress_callback("controlnet", req["steps"]),
    )
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    image = result.images[0]
    out_path, out_name = _new_output_path("controlnet")
    image.save(out_path)

    meta = {
        "mode": "controlnet",
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", " "),
        "control_type": req.get("control_type", "canny"),
        "controlnet_conditioning_scale": req.get("controlnet_conditioning_scale", 1.0),
        "width": w,
        "height": h,
        "steps": req["steps"],
        "cfg": req["cfg"],
        "seed": used_seed,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "image_url": f"/outputs/{out_name}",
    }
    return meta


def run_inpaint(req: dict, src_image: Image.Image, mask_img: Image.Image) -> dict:
    """抽出元: app.py generate_inpaint()。

    TE device mismatch対策はrun_controlnet()と同じ理由(共有text_encoderを使う
    ControlNet系パイプラインのため、直前呼び出しのCPU退避を引き継がないよう保険で
    ensure_text_encoder_on_gpu()を呼ぶ。run_controlnet()のdocstring参照)。
    """
    w = _round16(src_image.width)
    h = _round16(src_image.height)
    if (w, h) != src_image.size:
        src_image = src_image.resize((w, h), Image.LANCZOS)
    if mask_img.size != (w, h):
        mask_img = mask_img.resize((w, h), Image.NEAREST)

    pipe = controlnet_mod.get_controlnet_inpaint_pipeline()
    ensure_text_encoder_on_gpu(pipe)
    generator, used_seed = make_generator(req.get("seed", -1))

    _reset_vram_stats()
    t0 = time.time()
    # 重要: control_image は元画像(マスクされていない全体)、control_mask がマスク
    # (白=編集領域)。パイプライン内部でマスク領域のみ再生成するよう合成される。
    result = pipe(
        prompt=req["prompt"],
        negative_prompt=req.get("negative_prompt") or " ",
        control_image=src_image,
        control_mask=mask_img,
        controlnet_conditioning_scale=req.get("controlnet_conditioning_scale", 1.0),
        width=w,
        height=h,
        num_inference_steps=req["steps"],
        true_cfg_scale=req["cfg"],
        generator=generator,
        callback_on_step_end=_progress_callback("inpaint", req["steps"]),
    )
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    out_image = result.images[0]
    out_path, out_name = _new_output_path("inpaint")
    out_image.save(out_path)

    meta = {
        "mode": "inpaint",
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", " "),
        "controlnet_conditioning_scale": req.get("controlnet_conditioning_scale", 1.0),
        "width": w,
        "height": h,
        "steps": req["steps"],
        "cfg": req["cfg"],
        "seed": used_seed,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "image_url": f"/outputs/{out_name}",
    }
    return meta


def run_canny(image: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> dict:
    """アップロード画像から cv2.Canny でエッジ検出した制御画像を作る(おまけ機能、GPU不使用)。

    抽出元: app.py make_canny()。
    """
    import cv2
    import numpy as np

    src = image.convert("RGB")
    arr = np.array(src)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    out_img = Image.fromarray(edges_rgb)

    out_path, out_name = _new_output_path("canny")
    out_img.save(out_path)
    return {"image_url": f"/outputs/{out_name}"}


def run_layered(req: dict, src_image: Image.Image) -> dict:
    """抽出元: app.py generate_layered()。src_image は .convert("RGBA") 済みを渡すこと。

    TE device mismatch対策はrun_controlnet()と同じ理由(Layeredのtext_encoder/
    tokenizerも共有インスタンスを再利用するため。run_controlnet()のdocstring参照)。
    """
    resolution = req.get("resolution", LAYERED_DEFAULT_RESOLUTION)
    layers = req.get("layers", LAYERED_DEFAULT_LAYERS)
    if resolution not in LAYERED_VALID_RESOLUTIONS:
        raise ValueError(f"resolution は {LAYERED_VALID_RESOLUTIONS} のいずれかである必要があります")
    if layers < 1 or layers > 16:
        raise ValueError("layers は 1〜16 の範囲で指定してください")

    pipe = layered_mod.get_layered_pipeline()
    ensure_text_encoder_on_gpu(pipe)
    lightning_merged = state.layered_group["lightning_merged"]

    steps = req.get("steps")
    cfg = req.get("cfg")
    shift = req.get("shift")
    if lightning_merged:
        steps_eff = steps if steps is not None else LAYERED_LIGHTNING_STEPS
        cfg_eff = cfg if cfg is not None else LAYERED_LIGHTNING_CFG
        shift_eff = shift if shift is not None else LAYERED_LIGHTNING_SHIFT
    else:
        steps_eff = steps if steps is not None else LAYERED_DEFAULT_STEPS
        cfg_eff = cfg if cfg is not None else LAYERED_DEFAULT_CFG
        shift_eff = shift

    edit_mod.set_scheduler_shift(pipe.scheduler, shift_eff)

    generator, used_seed = make_generator(req.get("seed", -1))
    prompt = req.get("prompt") or None
    negative_prompt = req.get("negative_prompt") or None
    cfg_normalize = req.get("cfg_normalize", False)
    use_en_prompt = req.get("use_en_prompt", True)

    _reset_vram_stats()
    t0 = time.time()
    result = pipe(
        image=src_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        true_cfg_scale=cfg_eff,
        layers=layers,
        num_inference_steps=steps_eff,
        resolution=resolution,
        cfg_normalize=cfg_normalize,
        use_en_prompt=use_en_prompt,
        generator=generator,
        callback_on_step_end=_progress_callback("layered", steps_eff),
    )
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    layer_images = result.images[0]  # list of `layers` RGBA PIL.Image
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    layer_urls = []
    for idx, layer_img in enumerate(layer_images):
        name = f"layered_{ts}_{uid}_layer{idx}.png"
        layer_img.save(os.path.join(OUTPUTS_DIR, name))
        layer_urls.append(f"/outputs/{name}")

    # 合成プレビュー: 各レイヤーを順にアルファ合成して元画像相当を再構成する
    # (レイヤーの並び順が背面->前面である想定。目視で崩れる場合は並びが逆の可能性がある)。
    composite = Image.new("RGBA", layer_images[0].size, (0, 0, 0, 0))
    for layer_img in layer_images:
        composite = Image.alpha_composite(composite, layer_img)
    composite_name = f"layered_{ts}_{uid}_composite.png"
    composite.save(os.path.join(OUTPUTS_DIR, composite_name))

    meta = {
        "mode": "layered",
        "prompt": req.get("prompt", ""),
        "negative_prompt": req.get("negative_prompt", ""),
        "layers": layers,
        "resolution": resolution,
        "steps": steps_eff,
        "cfg": cfg_eff,
        "shift": shift_eff,
        "cfg_normalize": cfg_normalize,
        "use_en_prompt": use_en_prompt,
        "seed": used_seed,
        "quant": state.layered_group["quant"],
        "lightning_merged": lightning_merged,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "layer_image_urls": layer_urls,
        "composite_image_url": f"/outputs/{composite_name}",
        "image_url": f"/outputs/{composite_name}",
    }
    return meta
