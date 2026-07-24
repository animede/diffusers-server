# -*- coding: utf-8 -*-
"""
Layered(QwenImageLayeredPipeline、画像 -> 複数RGBAレイヤー分解)のロード。

抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py
  - _load_layered_group_locked()(行1381-1494)
  - get_layered_pipeline()(行1497-1502)

ロジック・既定値・実測コメント・ロード順序は無変更。
"""
import time

import torch
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImageLayeredPipeline,
    QwenImageTransformer2DModel,
)
from huggingface_hub import snapshot_download
from transformers import Qwen2VLProcessor

from core.loaders import (
    fuse_lightning_lora_and_cast_to_fp8,
    load_transformer_from_pretrained_streaming,
    load_transformer_gguf,
)
from core.optimize import apply_attention_backend, apply_compile, configure_transformer_offload
from core import progress as progress_mod
from core.resolve import COMFYUI_MODELS_DIR, resolve_model_path

from families.qwen_image import state
from families.qwen_image.paths import (
    LAYERED_GGUF_CONFIG_REPO,
    LAYERED_GGUF_FILENAME_TEMPLATE,
    LAYERED_GGUF_HF_REPO,
    LAYERED_REPO,
    T2I_LORA_4STEP_HF_FILE,
    T2I_LORA_4STEP_HF_REPO,
    T2I_LORA_4STEP_PATH,
)
from families.qwen_image.runtime import is_fp8_lightning_quant, quant_suffix
from families.qwen_image.shared import load_shared_components_locked, warn_if_model_cpu_with_quant

import os


def _load_layered_group_locked() -> None:
    """Layered transformer/vae/processor をロードし、QwenImageLayeredPipeline を構築する。

    text_encoder/tokenizer は shared を再利用する(sha256一致を確認済み)。vae は専用
    (input_channels=4、RGBA画素空間)なので必ず個別ロードする。ロード順序は
    fp8-lightning/GGUF 時、Edit/T2Iと同じ理由(大きいtransformerを先にVRAMへ確保してから
    共有コンポーネントを読む)で制御する。

    抽出元: pipeline_manager.py _load_layered_group_locked()(行1381-1494)。
    """
    layered_group = state.layered_group
    if layered_group["loaded"]:
        return
    raw_quant = state.get_runtime_config().layered_quant
    quant = quant_suffix(raw_quant)
    fp8_lightning = is_fp8_lightning_quant(raw_quant)
    small_transformer = bool(quant) or fp8_lightning
    t0 = time.time()
    offload_mode = state.get_offload_mode(small_transformer_active=small_transformer)
    load_device = state.t2i_load_device(offload_mode)

    if quant:
        filename = LAYERED_GGUF_FILENAME_TEMPLATE.format(suffix=quant)
        local_path = os.path.join(COMFYUI_MODELS_DIR, "diffusion_models", filename)
        path = resolve_model_path(local_path, LAYERED_GGUF_HF_REPO, filename)
        print(f"[families.qwen_image] loading Layered transformer as GGUF({quant}) from {path}")
        transformer = load_transformer_gguf(QwenImageTransformer2DModel, path, LAYERED_GGUF_CONFIG_REPO)
        # GGUFは小さいためCPU RAM常駐を避け、常にフルGPU常駐にする(group系オフロードは使わない)。
        transformer.to("cuda")
        transformer_offload_mode = "none"
    elif fp8_lightning:
        # bf16 Layered transformer(実測約38.05GB、Edit-2511と同一アーキテクチャサイズ)を
        # HF Hubシャードから直接cuda:0へストリーミングロードし、その場でLightning LoRA
        # (T2I/I2Iと同一ファイル、ComfyUIワークフローのnode 93と同じ)をfuseしてから
        # enable_layerwise_casting でストレージをfp8化する。共有コンポーネント読み込み前に
        # VRAMの空きを最大限確保するため、offload_modeに関わらず常に直接 "cuda:0" へロードする。
        print(
            f"[families.qwen_image] loading Layered transformer for fp8-lightning fuse from "
            f"{LAYERED_REPO}/transformer (HF Hub shards, 直接cuda:0へストリーミング)"
        )
        transformer = load_transformer_from_pretrained_streaming(
            QwenImageTransformer2DModel, LAYERED_REPO, "transformer", "cuda:0"
        )
        print("[families.qwen_image] fusing Layered Lightning LoRA and casting to fp8_e4m3fn storage...")
        fuse_lightning_lora_and_cast_to_fp8(
            transformer, T2I_LORA_4STEP_PATH, T2I_LORA_4STEP_HF_REPO, T2I_LORA_4STEP_HF_FILE
        )
        if torch.cuda.is_available():
            print(
                f"[families.qwen_image] Layered fp8-lightning transformer ready, "
                f"VRAM={torch.cuda.memory_allocated() / 1024**3:.2f}GB"
            )
        transformer_offload_mode = "none"
    else:
        print(
            f"[families.qwen_image] loading Layered transformer (bf16, non-quantized) from "
            f"{LAYERED_REPO}/transformer (load_device={load_device})"
        )
        transformer = load_transformer_from_pretrained_streaming(
            QwenImageTransformer2DModel, LAYERED_REPO, "transformer", load_device
        )
        transformer_offload_mode = offload_mode

    if fp8_lightning:
        load_shared_components_locked(small_transformer_active=small_transformer)
        warn_if_model_cpu_with_quant("fp8-lightning", offload_mode)
    else:
        load_shared_components_locked(small_transformer_active=small_transformer)
        warn_if_model_cpu_with_quant(quant, offload_mode)

    print(f"[families.qwen_image] loading Layered vae (dedicated, RGBA) from {LAYERED_REPO}/vae")
    vae = AutoencoderKLQwenImage.from_pretrained(LAYERED_REPO, subfolder="vae", torch_dtype=torch.bfloat16)
    vae.to("cuda" if torch.cuda.is_available() else "cpu")  # 小さい(約250MB)ため常にフルGPU常駐
    # DS_QWEN_TILED_VAE(CLAUDE.md 56番)は対象外: このVAEは重み自体が約250MBと小さく
    # OOMの実害が無い(CLAUDE.md 8番、共有VAEと違いinput_channels=4のRGBA専用チェックポイント)。
    # タイル化はエンコード/デコードの空間分割によるメモリ削減が目的のため、重みサイズと
    # activation側のメモリ要求は無関係で理論上は同様に効くはずだが、RGBA画素空間のブレンド
    # 挙動(通常RGBのblend_v/blend_hと同一コードパスだが未検証)による画質影響の実機確認が
    # 無いため、実害の無いこのVAEには適用しない(触らない方が安全、という判断)。

    proc_dir = snapshot_download(repo_id=LAYERED_REPO, allow_patterns=["processor/*"])
    # AutoProcessor.from_pretrained()はこの環境だとQwen2VLProcessorへ正しく解決できないため、
    # 明示的に Qwen2VLProcessor を使う(Edit と同じ理由)。
    processor = Qwen2VLProcessor.from_pretrained(proc_dir, subfolder="processor")

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(LAYERED_REPO, subfolder="scheduler")

    if transformer_offload_mode == "model_cpu":
        pipe = QwenImageLayeredPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=state.shared["text_encoder"],
            tokenizer=state.shared["tokenizer"],
            processor=processor,
            transformer=transformer,
        )
        pipe.enable_model_cpu_offload()
    else:
        if not quant and not fp8_lightning:
            configure_transformer_offload(transformer, transformer_offload_mode)
        pipe = QwenImageLayeredPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=state.shared["text_encoder"],
            tokenizer=state.shared["tokenizer"],
            processor=processor,
            transformer=transformer,
        )

    apply_attention_backend(transformer, state.get_runtime_config().attention_backend, state.get_runtime_config().device)
    apply_compile(transformer, state.get_runtime_config().compile)

    quant_label = "fp8-lightning" if fp8_lightning else quant
    layered_group["transformer"] = transformer
    layered_group["vae"] = vae
    layered_group["processor"] = processor
    layered_group["pipe"] = pipe
    layered_group["scheduler"] = scheduler
    layered_group["loaded"] = True
    layered_group["load_time_s"] = time.time() - t0
    layered_group["quant"] = quant_label
    layered_group["lightning_merged"] = fp8_lightning
    print(
        f"[families.qwen_image] Layered group loaded in {layered_group['load_time_s']:.1f}s "
        f"(offload_mode={transformer_offload_mode}, quant={quant_label})"
    )


def get_layered_pipeline() -> QwenImageLayeredPipeline:
    """抽出元: pipeline_manager.py get_layered_pipeline()(行1497-1502)。"""
    if not state.layered_group["loaded"]:
        with state.lock:
            if not state.layered_group["loaded"]:
                _load_layered_group_locked()
    pipe = state.layered_group["pipe"]
    progress_mod.disable_diffusers_tqdm(pipe)
    return pipe
