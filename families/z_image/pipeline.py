# -*- coding: utf-8 -*-
"""
ZImagePipeline のロード(bf16/bnb-4bit ビルダー + offload/attention/compile 適用)。
I2I / Inpaint パイプラインは base の components から遅延構築する。

抽出元: /home/animede/Z-image-diffusers/pipeline_manager.py
  - MODEL_ID / DEFAULT_STEPS / DEFAULT_GUIDANCE_SCALE(行53-57)
  - _build_bf16() / _build_bnb_4bit() / PRECISION_BUILDERS(行154-184)
  - _apply_vae_tiling()(行328-341)
  - load_pipeline() / get_i2i_pipeline() / get_inpaint_pipeline()(行343-420)

変更点:
  - offload 適用は core.optimize.apply_flux2_offload() を再利用する(ZImagePipeline も
    Flux2Pipeline と同じ "none"/"model_cpu_offload"/"group" の三択・同じ
    enable_model_cpu_offload()/enable_group_offload() API を持つため、専用実装を
    新設せず既存のパイプライン単位オフロード関数をそのまま流用する)。
  - attention backend / compile 適用は core.optimize.apply_attention_backend() /
    apply_compile() に委譲(Qwen系・flux2系と共通のロジック、transformer に対して
    同じ呼び出しをするだけなので無変更で流用できる)。
  - タスク指示33番(丸ごとCPUスワップ禁止): 48GB専有では bf16(~20GB)が余裕を持って
    収まるため、既定オフロードは "none"(全常駐)のまま(families/z_image/runtime.py
    DEFAULT_OFFLOAD 参照)。
  - 2026-07-19タスク(CLAUDE.md 35番): Hub配布の transformer は fp32のみ(3シャード
    24.6GB)。tools/convert_z_image_bf16.py でローカルに bf16単一ファイル(約11.5GB)へ
    変換し、以後は _load_transformer_bf16() がそれを streaming ロードする(見つからなければ
    従来どおり Hub から fp32 shards を取得して bf16 キャストする、後方互換フォールバック)。
    vae/text_encoder/tokenizer/scheduler は元々 bf16/軽量なので変換対象外
    (from_pretrained(..., transformer=<preloaded>) で transformer だけ差し替える、
    families/qwen_image/layered.py と同じ「component個別ロード→pipeline構築」パターン)。
"""
import os
import time

import torch
from diffusers import ZImageImg2ImgPipeline, ZImageInpaintPipeline, ZImagePipeline, ZImageTransformer2DModel

from core.loaders import load_transformer_from_config
from core.optimize import apply_attention_backend, apply_compile, apply_flux2_offload
from core.resolve import resolve_comfyui_path

from families.z_image import state
from families.z_image.runtime import ZImageRuntimeConfig

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"

# Turbo既定: 蒸留版のため少ステップ・CFGなしを前提に調整済み。
DEFAULT_STEPS = 8
DEFAULT_GUIDANCE_SCALE = 0.0

# ローカル変換済み bf16 transformer(tools/convert_z_image_bf16.py の出力先と一致させること)。
LOCAL_BF16_TRANSFORMER_PATH = resolve_comfyui_path("diffusion_models", "z_image_turbo_bf16.safetensors")


def _load_transformer_bf16(load_device: str = "cuda:0") -> ZImageTransformer2DModel:
    """ローカル bf16 変換済みファイルがあれば streaming ロード、無ければ Hub の fp32 shards
    から from_pretrained(torch_dtype=bf16) でロードする(旧来どおりの後方互換フォールバック)。
    """
    if os.path.exists(LOCAL_BF16_TRANSFORMER_PATH):
        print(
            f"[families.z_image] loading transformer from local bf16 file "
            f"{LOCAL_BF16_TRANSFORMER_PATH} (streaming)"
        )
        return load_transformer_from_config(
            ZImageTransformer2DModel, MODEL_ID, "transformer", LOCAL_BF16_TRANSFORMER_PATH, load_device
        )
    print(
        f"[families.z_image] {LOCAL_BF16_TRANSFORMER_PATH} が見つからないため "
        f"Hub の fp32 shards から transformer をロードします "
        f"(tools/convert_z_image_bf16.py でローカル変換すると以後は再DL不要になります)"
    )
    return ZImageTransformer2DModel.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16)


def _build_bf16(config: ZImageRuntimeConfig) -> ZImagePipeline:
    transformer = _load_transformer_bf16(load_device=config.device)
    return ZImagePipeline.from_pretrained(MODEL_ID, transformer=transformer, torch_dtype=torch.bfloat16)


def _build_bnb_4bit(config: ZImageRuntimeConfig) -> ZImagePipeline:
    from diffusers.quantizers import PipelineQuantizationConfig

    quant_config = PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs=dict(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        components_to_quantize=["transformer", "text_encoder"],
    )
    return ZImagePipeline.from_pretrained(
        MODEL_ID, quantization_config=quant_config, torch_dtype=torch.bfloat16
    )


PRECISION_BUILDERS = {
    "bf16": _build_bf16,
    "bnb-4bit": _build_bnb_4bit,
}


def _apply_vae_tiling(pipe: ZImagePipeline, config: ZImageRuntimeConfig) -> None:
    """抽出元: pipeline_manager.py _apply_vae_tiling()(行328-341)。"""
    if not config.vae_tiling:
        return
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        print("[families.z_image] VAE slicing + tiling enabled")
    except Exception as e:  # noqa: BLE001
        print(
            f"[families.z_image] warning: could not enable VAE tiling "
            f"({type(e).__name__}: {e}); continuing without it."
        )


def _load_base_pipeline_locked() -> None:
    """state.lock を呼び出し側が保持している前提でロードする(冪等)。"""
    ps = state.base_pipeline_state
    if ps["loaded"]:
        return

    config = state.get_runtime_config()
    if config.precision not in PRECISION_BUILDERS:
        raise ValueError(
            f"Unknown DS_ZIMAGE_PRECISION {config.precision!r}. Available: {list(PRECISION_BUILDERS)}"
        )

    print(f"[families.z_image] loading ZImagePipeline: {config}")
    t0 = time.time()

    builder = PRECISION_BUILDERS[config.precision]
    pipe = builder(config)

    apply_flux2_offload(pipe, config.offload, device=config.device, group_stream=config.group_stream)
    apply_attention_backend(pipe.transformer, config.attention_backend, config.device)
    apply_compile(pipe.transformer, config.compile)
    _apply_vae_tiling(pipe, config)

    ps["pipe"] = pipe
    ps["loaded"] = True
    ps["precision"] = config.precision
    ps["offload_mode"] = config.offload
    ps["load_time_s"] = time.time() - t0
    print(
        f"[families.z_image] loaded in {ps['load_time_s']:.1f}s "
        f"(precision={config.precision}, offload={config.offload})"
    )


def get_base_pipeline():
    """T2I用の ZImagePipeline を返す(冪等ロード)。"""
    if not state.base_pipeline_state["loaded"]:
        with state.lock:
            if not state.base_pipeline_state["loaded"]:
                _load_base_pipeline_locked()
    return state.base_pipeline_state["pipe"]


def get_i2i_pipeline():
    """ZImageImg2ImgPipeline を返す。base の components から遅延構築(参照共有、コピーで
    はないため VRAM 使用量は増えない。旧実装 get_i2i_pipeline() と同じ設計)。
    """
    base = get_base_pipeline()
    ps = state.i2i_pipeline_state
    if not ps["loaded"]:
        with state.lock:
            if not ps["loaded"]:
                ps["pipe"] = ZImageImg2ImgPipeline(**base.components)
                ps["loaded"] = True
                print("[families.z_image] ZImageImg2ImgPipeline derived from base pipeline components")
    return ps["pipe"]


def get_inpaint_pipeline():
    """ZImageInpaintPipeline を返す。base の components から遅延構築(旧実装
    get_inpaint_pipeline() と同じ設計)。
    """
    base = get_base_pipeline()
    ps = state.inpaint_pipeline_state
    if not ps["loaded"]:
        with state.lock:
            if not ps["loaded"]:
                ps["pipe"] = ZImageInpaintPipeline(**base.components)
                ps["loaded"] = True
                print("[families.z_image] ZImageInpaintPipeline derived from base pipeline components")
    return ps["pipe"]
