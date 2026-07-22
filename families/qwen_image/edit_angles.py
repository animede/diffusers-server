# -*- coding: utf-8 -*-
"""
charsheet 専用: Edit の「fp8-lightning-angles」変種のロード・排他管理。

背景(タスク指示): charsheet(8方向キャラシート生成)は Phase 3 でプロンプトのみ方式に
移行したが、ユーザー目視比較で「コスチューム等の一貫性が旧 Multiple-angles LoRA 版より
劣る」と判定されたため、REBUILD_PLAN §4 Phase 3 の予定どおり LoRA 併用を復活させる。

構築順序(families/qwen_image/edit.py の fp8-lightning 経路を拡張):
  bf16 2511 transformer ロード(直接 cuda:0)
  -> Lightning LoRA fuse(core.loaders.fuse_lightning_and_extra_lora_and_cast_to_fp8 内で
     fuse_lora_into_transformer を1回目呼び出し)
  -> Multiple-angles LoRA fuse(同関数内で2回目、adapter_name="angles_fuse", weight=1.0。
     旧 /home/animede/charsheet/pipeline.py _apply_loras() の
     pipe.set_adapters(["lightning", "angles"], adapter_weights=[1.0, 1.0]) と
     同じ強度で重みに永続 fuse する)
  -> enable_layerwise_casting(fp8_e4m3fn) でストレージ圧縮

このグループ(state.edit_angles_group)は通常の Edit(state.edit_group、fp8-lightning、
Multiple-angles なし)とは完全に別物として管理し、同時常駐しない。排他は
families/qwen_image/family.py の load()/generate() 側で
「edit_angles をロードする前に edit_group がロード済みなら unload、逆も同様」という
形で実装する(通常の /api/edit の挙動・重みには一切影響を与えない)。

fuse後は Lightning・Multiple-angles とも通常のLoRAアダプタとして無効化できない
(重みに永続的に焼き込まれるため)。そのため lightning_merged は常に True。
"""
import time

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from huggingface_hub import snapshot_download
from transformers import Qwen2VLProcessor

from core.loaders import fuse_lightning_and_extra_lora_and_cast_to_fp8, load_transformer_from_config
from core.optimize import apply_attention_backend, apply_compile
from core.resolve import resolve_model_path

from families.qwen_image import state
from families.qwen_image.paths import (
    EDIT_LORA_ANGLES_HF_FILE,
    EDIT_LORA_ANGLES_HF_REPO,
    EDIT_LORA_ANGLES_PATH,
    EDIT_LORA_LIGHTNING_HF_FILE,
    EDIT_LORA_LIGHTNING_HF_REPO,
    EDIT_LORA_LIGHTNING_PATH,
    EDIT_PROCESSOR_REPO,
    EDIT_TRANSFORMER_HF_FILE,
    EDIT_TRANSFORMER_HF_REPO,
    EDIT_TRANSFORMER_PATH,
    BASE_REPO,
)
from families.qwen_image.shared import load_shared_components_locked, warn_if_model_cpu_with_quant


def _load_edit_angles_group_locked() -> None:
    """Edit(fp8-lightning-angles)transformer をロードし、
    QwenImageEditPlusPipeline を構築する(共有コンポーネントを再利用)。

    edit.py の _load_edit_group_locked() の fp8-lightning 分岐とほぼ同じロード順序
    (transformer を先に直接 cuda:0 へロードしてから共有コンポーネントをロードする。
    fuse時点で bf16 約38GB を一時的に必要とするため)。このグループは常に
    fp8-lightning-angles 固定(GGUF/量子化切替は持たない。charsheet専用のため)。
    """
    edit_angles_group = state.edit_angles_group
    if edit_angles_group["loaded"]:
        return
    t0 = time.time()
    # angles変種は常にfp8-lightning相当の「小さいtransformer」として扱う
    # (offload_modeの閾値判定・共有コンポーネントのGPU配置判断に使う)。
    offload_mode = state.get_offload_mode(small_transformer_active=True)

    path = resolve_model_path(EDIT_TRANSFORMER_PATH, EDIT_TRANSFORMER_HF_REPO, EDIT_TRANSFORMER_HF_FILE)
    print(
        f"[families.qwen_image] loading Edit transformer for fp8-lightning-angles fuse from {path} "
        "(直接cuda:0へ)"
    )
    transformer = load_transformer_from_config(
        QwenImageTransformer2DModel, EDIT_PROCESSOR_REPO, "transformer", path, "cuda:0"
    )
    print(
        "[families.qwen_image] fusing Lightning LoRA + Multiple-angles LoRA and casting to "
        "fp8_e4m3fn storage..."
    )
    fuse_lightning_and_extra_lora_and_cast_to_fp8(
        transformer,
        EDIT_LORA_LIGHTNING_PATH, EDIT_LORA_LIGHTNING_HF_REPO, EDIT_LORA_LIGHTNING_HF_FILE,
        EDIT_LORA_ANGLES_PATH, EDIT_LORA_ANGLES_HF_REPO, EDIT_LORA_ANGLES_HF_FILE,
        extra_adapter_name="angles_fuse", extra_weight=1.0,
    )
    if torch.cuda.is_available():
        print(
            "[families.qwen_image] fp8-lightning-angles transformer ready, "
            f"VRAM={torch.cuda.memory_allocated()/1024**3:.2f}GB"
        )

    # transformer をVRAMに確保した後で共有コンポーネント(text_encoder等)をロードする
    # (edit.py と同様の順序に関する注意。fuse中の一時ピークを優先的に確保するため)。
    load_shared_components_locked(small_transformer_active=True)
    warn_if_model_cpu_with_quant("fp8-lightning-angles", offload_mode)

    proc_dir = snapshot_download(repo_id=EDIT_PROCESSOR_REPO, allow_patterns=["processor/*"])
    processor = Qwen2VLProcessor.from_pretrained(proc_dir, subfolder="processor")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(BASE_REPO, subfolder="scheduler")

    edit_pipe = QwenImageEditPlusPipeline(
        scheduler=scheduler,
        vae=state.shared["vae"],
        text_encoder=state.shared["text_encoder"],
        tokenizer=state.shared["tokenizer"],
        processor=processor,
        transformer=transformer,
    )

    runtime_config = state.get_runtime_config()
    apply_attention_backend(transformer, runtime_config.attention_backend, runtime_config.device)
    apply_compile(transformer, runtime_config.compile)

    # Lightning・Multiple-anglesともすでに重みへfuse済み。追加のLoRAロードは不要で、
    # 常にLightning適用状態として扱う(無効化不可)。
    edit_angles_group["transformer"] = transformer
    edit_angles_group["edit_pipe"] = edit_pipe
    edit_angles_group["scheduler"] = scheduler
    edit_angles_group["processor"] = processor
    edit_angles_group["loaded"] = True
    edit_angles_group["load_time_s"] = time.time() - t0
    edit_angles_group["fallback"] = False
    edit_angles_group["quant"] = "fp8-lightning-angles"
    edit_angles_group["lightning_merged"] = True
    print(
        f"[families.qwen_image] Edit-angles group loaded in {edit_angles_group['load_time_s']:.1f}s "
        "(quant=fp8-lightning-angles)"
    )


def set_scheduler_shift(scheduler, shift) -> None:
    """edit.py の set_scheduler_shift() に委譲する(2026-07-18: 重複実装だと修正漏れの
    リスクがあるため一本化。use_dynamic_shifting時の base_shift/max_shift 上書き修正の
    詳細は edit.py の docstring 参照)。"""
    from families.qwen_image.edit import set_scheduler_shift as _set_scheduler_shift

    _set_scheduler_shift(scheduler, shift)


def get_edit_angles_pipeline() -> QwenImageEditPlusPipeline:
    if not state.edit_angles_group["loaded"]:
        with state.lock:
            if not state.edit_angles_group["loaded"]:
                _load_edit_angles_group_locked()
    return state.edit_angles_group["edit_pipe"]
