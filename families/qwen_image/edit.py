# -*- coding: utf-8 -*-
"""
Edit(QwenImageEditPlusPipeline、Qwen-Image-Edit-2511)のロードと Lightning LoRA 制御。

抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py
  - _load_edit_group_locked()(行1200-1321)
  - _apply_edit_loras()(行1324-1349)
  - set_edit_lightning()(行1352-1368)
  - set_scheduler_shift()(行1371-1375)
  - get_edit_pipeline()(行1532-1537)

ロジック・既定値・実測コメント・ロード順序(transformer 先読み)は無変更。
"""
import math
import time

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from huggingface_hub import snapshot_download
from transformers import Qwen2VLProcessor

from core.loaders import fuse_lightning_lora_and_cast_to_fp8, load_transformer_from_config
from core.optimize import (
    apply_attention_backend,
    apply_compile,
    configure_transformer_offload,
    suspended_whole_module_swap_offload,
)
from core import progress as progress_mod
from core.resolve import resolve_model_path

from families.qwen_image import state
from families.qwen_image.paths import (
    EDIT_GGUF_CONFIG_REPO,
    EDIT_GGUF_FILENAME_TEMPLATE,
    EDIT_GGUF_HF_REPO,
    EDIT_LORA_LIGHTNING_HF_FILE,
    EDIT_LORA_LIGHTNING_HF_REPO,
    EDIT_LORA_LIGHTNING_PATH,
    EDIT_PROCESSOR_REPO,
    EDIT_TRANSFORMER_FALLBACK_HF_FILE,
    EDIT_TRANSFORMER_FALLBACK_HF_REPO,
    EDIT_TRANSFORMER_FALLBACK_PATH,
    EDIT_TRANSFORMER_FALLBACK_PREFIX,
    EDIT_TRANSFORMER_HF_FILE,
    EDIT_TRANSFORMER_HF_REPO,
    EDIT_TRANSFORMER_PATH,
    BASE_REPO,
)
from families.qwen_image.runtime import is_fp8_lightning_quant, quant_suffix
from families.qwen_image.shared import load_shared_components_locked, warn_if_model_cpu_with_quant

import os

from core.resolve import COMFYUI_MODELS_DIR
from core.loaders import load_transformer_gguf


def _load_edit_group_locked() -> None:
    """Edit transformer をロードし、QwenImageEditPlusPipeline を構築する(共有コンポーネントを再利用)。

    ロード順序に注意: 大きい transformer(bf16で約40GB)は「まず自分だけでVRAMに載せる」
    ことを優先し、共有コンポーネント(text_encoder等)のロードはその後に回す(先に共有
    コンポーネントをロードしてしまうと、その分VRAMの空きが減り、40GB近い transformer の
    直接ロードが収まらなくなるリスクが高まるため)。fp8-lightning は特にこの順序が重要
    (fuse時点では圧縮前のbf16、約38GBを一時的に必要とする)。

    抽出元: pipeline_manager.py _load_edit_group_locked()(行1200-1321)。
    """
    edit_group = state.edit_group
    if edit_group["loaded"]:
        return
    raw_quant = state.get_runtime_config().quant
    quant = quant_suffix(raw_quant)
    fp8_lightning = is_fp8_lightning_quant(raw_quant)
    small_transformer = bool(quant) or fp8_lightning
    t0 = time.time()
    offload_mode = state.get_offload_mode(small_transformer_active=small_transformer)
    load_device = state.t2i_load_device(offload_mode)

    fallback_used = False
    if quant:
        filename = EDIT_GGUF_FILENAME_TEMPLATE.format(suffix=quant)
        local_path = os.path.join(COMFYUI_MODELS_DIR, "diffusion_models", filename)
        path = resolve_model_path(local_path, EDIT_GGUF_HF_REPO, filename)
        print(f"[families.qwen_image] loading Edit transformer as GGUF({quant}) from {path}")
        transformer = load_transformer_gguf(QwenImageTransformer2DModel, path, EDIT_GGUF_CONFIG_REPO)
        # GGUFは小さいためCPU RAM常駐を避け、常にフルGPU常駐にする(group系オフロードは使わない)。
        transformer.to("cuda")
        transformer_offload_mode = "none"
    elif fp8_lightning:
        # bf16 2511 transformerを直接GPUへロード(実測 約38GB)し、その場でLightning
        # LoRAをfuseしてから enable_layerwise_casting でストレージをfp8化する(実測19.05GB)。
        # 共有コンポーネント読み込み前にVRAMの空きを最大限確保するため、offload_modeに
        # 関わらず常に直接 "cuda:0" へロードする(group系オフロードは使わない)。
        path = resolve_model_path(EDIT_TRANSFORMER_PATH, EDIT_TRANSFORMER_HF_REPO, EDIT_TRANSFORMER_HF_FILE)
        print(f"[families.qwen_image] loading Edit transformer for fp8-lightning fuse from {path} (直接cuda:0へ)")
        transformer = load_transformer_from_config(
            QwenImageTransformer2DModel, EDIT_PROCESSOR_REPO, "transformer", path, "cuda:0"
        )
        print("[families.qwen_image] fusing Lightning LoRA and casting to fp8_e4m3fn storage...")
        fuse_lightning_lora_and_cast_to_fp8(
            transformer, EDIT_LORA_LIGHTNING_PATH, EDIT_LORA_LIGHTNING_HF_REPO, EDIT_LORA_LIGHTNING_HF_FILE
        )
        if torch.cuda.is_available():
            print(f"[families.qwen_image] fp8-lightning transformer ready, VRAM={torch.cuda.memory_allocated()/1024**3:.2f}GB")
        transformer_offload_mode = "none"
    else:
        try:
            path = resolve_model_path(EDIT_TRANSFORMER_PATH, EDIT_TRANSFORMER_HF_REPO, EDIT_TRANSFORMER_HF_FILE)
            print(f"[families.qwen_image] loading Edit transformer from {path} (load_device={load_device})")
            transformer = load_transformer_from_config(
                QwenImageTransformer2DModel, EDIT_PROCESSOR_REPO, "transformer", path, load_device
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[families.qwen_image] primary Edit transformer failed ({exc!r}); falling back to 2509 fp8")
            fallback_used = True
            path = resolve_model_path(
                EDIT_TRANSFORMER_FALLBACK_PATH, EDIT_TRANSFORMER_FALLBACK_HF_REPO, EDIT_TRANSFORMER_FALLBACK_HF_FILE
            )
            transformer = load_transformer_from_config(
                QwenImageTransformer2DModel,
                EDIT_PROCESSOR_REPO,
                "transformer",
                path,
                load_device,
                strip_prefix=EDIT_TRANSFORMER_FALLBACK_PREFIX,
            )
        transformer_offload_mode = offload_mode

    # transformer をVRAMに確保した後で共有コンポーネント(text_encoder等)をロードする
    # (上記の順序に関する注意を参照)。
    load_shared_components_locked(small_transformer_active=small_transformer)
    warn_if_model_cpu_with_quant(quant or ("fp8-lightning" if fp8_lightning else None), offload_mode)

    proc_dir = snapshot_download(repo_id=EDIT_PROCESSOR_REPO, allow_patterns=["processor/*"])
    # 注意: AutoProcessor.from_pretrained() はこの環境(transformers 5.x系)だと
    # preprocessor_config.json 内の "processor_class": "Qwen2VLProcessor" を正しく解決できず、
    # 画像処理を持たない素の Qwen2Tokenizer にフォールバックしてしまう(生成時に
    # "pixel_values" KeyError で落ちる。実機で確認済み)。Qwen2VLProcessor を明示指定する。
    processor = Qwen2VLProcessor.from_pretrained(proc_dir, subfolder="processor")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(BASE_REPO, subfolder="scheduler")

    if transformer_offload_mode == "model_cpu":
        edit_pipe = QwenImageEditPlusPipeline(
            scheduler=scheduler,
            vae=state.shared["vae"],
            text_encoder=state.shared["text_encoder"],
            tokenizer=state.shared["tokenizer"],
            processor=processor,
            transformer=transformer,
        )
        edit_pipe.enable_model_cpu_offload()
    else:
        if not quant and not fp8_lightning:  # quant/fp8_lightning時は上ですでに .to("cuda") 済み
            configure_transformer_offload(transformer, transformer_offload_mode)
        edit_pipe = QwenImageEditPlusPipeline(
            scheduler=scheduler,
            vae=state.shared["vae"],
            text_encoder=state.shared["text_encoder"],
            tokenizer=state.shared["tokenizer"],
            processor=processor,
            transformer=transformer,
        )

    apply_attention_backend(transformer, state.get_runtime_config().attention_backend, state.get_runtime_config().device)
    apply_compile(transformer, state.get_runtime_config().compile)

    quant_label = "fp8-lightning" if fp8_lightning else quant
    if fp8_lightning:
        # Lightning LoRAはすでに重みにfuse済み。追加のLoRAロードは不要で、常にLightning
        # 適用状態として扱う(無効化不可)。
        edit_group["lora_available"] = True
        edit_group["lightning_merged"] = True
    else:
        _apply_edit_loras(edit_pipe, gguf_quantized=bool(quant))

    edit_group["transformer"] = transformer
    edit_group["edit_pipe"] = edit_pipe
    edit_group["scheduler"] = scheduler
    edit_group["processor"] = processor
    edit_group["loaded"] = True
    edit_group["load_time_s"] = time.time() - t0
    edit_group["fallback"] = fallback_used
    edit_group["quant"] = quant_label
    print(
        f"[families.qwen_image] Edit group loaded in {edit_group['load_time_s']:.1f}s "
        f"(offload_mode={transformer_offload_mode}, fallback={fallback_used}, quant={quant_label})"
    )


def _apply_edit_loras(pipe, gguf_quantized: bool = False) -> None:
    """Edit 用 Lightning 4steps LoRA をロードし、既定は無効化しておく。

    GGUF量子化 transformer(GGUFLinear層)には現行の diffusers/peft では LoRAを適用できない
    (実機検証済み: PEFTのターゲットモジュール検出がGGUFLinearを認識せず
    "Target modules {...} not found in the base model" で失敗する)。失敗時はエラーにせず
    警告ログを出し、lora_available=False としてフォールバックする(呼び出し側
    set_edit_lightning() がこれを見て無効化する。app.py 側で steps/cfg を通常品質側に
    フォールバックさせる)。

    抽出元: pipeline_manager.py _apply_edit_loras()(行1324-1349)。
    """
    edit_group = state.edit_group
    try:
        path = resolve_model_path(EDIT_LORA_LIGHTNING_PATH, EDIT_LORA_LIGHTNING_HF_REPO, EDIT_LORA_LIGHTNING_HF_FILE)
        # group_lowvram の text_encoder 手動スワップを一時解除してからロードする
        # (t2i.py の _apply_t2i_loras() と同じ理由。詳細は
        # core.optimize.suspended_whole_module_swap_offload() のdocstring参照)。
        with suspended_whole_module_swap_offload(pipe.text_encoder):
            pipe.load_lora_weights(path, adapter_name="lightning")
        pipe.disable_lora()  # 既定は無効
        edit_group["lora_available"] = True
    except Exception as e:  # noqa: BLE001
        edit_group["lora_available"] = False
        edit_group["lora_unavailable_reason"] = f"{type(e).__name__}: {e}"[:300]
        if gguf_quantized:
            print(
                f"[families.qwen_image] warning: GGUF量子化transformerへのEdit Lightning LoRA適用に失敗しました"
                f"(既知の制約: PEFTがGGUFLinearを未対応)。LoRAなしにフォールバックします: "
                f"{type(e).__name__}: {e}"
            )
        else:
            print(f"[families.qwen_image] warning: Edit Lightning LoRA のロードに失敗しました: {e}")


def set_edit_lightning(pipe, enabled: bool) -> bool:
    """LoRA未ロード/非対応(GGUF量子化等)の場合は常に False を返す(要求されても無効のまま)。
    fp8-lightning(重みにfuse済み)の場合は常に True を返す(無効化不可)。

    抽出元: pipeline_manager.py set_edit_lightning()(行1352-1368)。
    """
    edit_group = state.edit_group
    if edit_group.get("lightning_merged"):
        return True
    if not edit_group.get("lora_available"):
        return False
    try:
        if enabled:
            pipe.set_adapters(["lightning"], adapter_weights=[1.0])
        else:
            pipe.disable_lora()
        return enabled
    except Exception as e:  # noqa: BLE001
        print(f"[families.qwen_image] warning: Edit Lightning LoRA 切替に失敗しました: {e}")
        return False


def set_scheduler_shift(scheduler, shift) -> None:
    """scheduler の shift(dynamic shifting用)を上書きする。None なら何もしない。

    抽出元: pipeline_manager.py set_scheduler_shift()(行1371-1375)。

    2026-07-18バグ修正(T2I fp8-lightning 4steps の霧がかかったような品質崩壊の根本原因):
    Qwen/Qwen-Image のscheduler_config.jsonは use_dynamic_shifting=true。diffusersの
    FlowMatchEulerDiscreteScheduler.set_timesteps() は use_dynamic_shifting=True の場合、
    `self.shift`(=scheduler.config["shift"])を一切参照せず、代わりに呼び出し側
    パイプライン(pipeline_qwenimage.py等)が解像度依存で計算した mu を使って
    `sigmas = exp(mu) / (exp(mu) + (1/t - 1))` で時間シフトする(scheduling_flow_match_
    euler_discrete.py の set_timesteps/_time_shift_exponential 参照)。この mu は
    scheduler.config の base_shift/max_shift/base_image_seq_len/max_image_seq_len から
    calculate_shift() で算出され、"shift" キーは完全に無視される。そのため旧実装
    (scheduler.config["shift"] = float(shift) のみ)は fp8-lightning T2I 時に
    LIGHTNING_SHIFT=3.0 を指定していたつもりが実際には何の効果もなく、実際の有効shift
    (exp(mu))は解像度依存の値(1024²で約2.0、624²で約1.76)になっていた
    ---Lightning LoRA(4steps distilled)は特定のshiftで学習されているため、この
    ズレが「霧がかかったような品質崩壊」の直接原因だった(実機再現・特定済み)。
    Edit(2511)は参照画像を含めてimage_seq_lenが大きくなりやすく、たまたま有効shiftが
    3.0に近い値になっていたため症状が目立たなかった(charsheet 2-3枚参照時は
    mu→exp(mu)≈3.0〜3.7で偶然近似していた)。

    lightx2v公式のQwen-Image-Lightningモデルカード(diffusers使用例)は
    base_shift=max_shift=math.log(3) と指定することで、use_dynamic_shifting=True の
    ままでも解像度に依存しない一定の有効shift(=exp(mu)=3.0)を実現している
    (base_shift==max_shiftだとcalculate_shiftの線形補間が定数になるため)。
    本修正はこれに倣い、use_dynamic_shifting が有効なら base_shift/max_shift を両方
    math.log(shift) に上書きする(mu = log(shift) で固定され、time_shift後の有効shiftが
    ちょうど shift になる)。use_dynamic_shifting が無効なスケジューラでは従来通り
    "shift" キーを直接上書きする(非dynamic分岐は self.shift を直接使うため)。
    """
    if shift is None:
        return
    shift = float(shift)
    if scheduler.config.get("use_dynamic_shifting", False):
        log_shift = math.log(shift)
        scheduler.config["base_shift"] = log_shift
        scheduler.config["max_shift"] = log_shift
    else:
        scheduler.config["shift"] = shift


def get_edit_pipeline() -> QwenImageEditPlusPipeline:
    """抽出元: pipeline_manager.py get_edit_pipeline()(行1532-1537)。"""
    if not state.edit_group["loaded"]:
        with state.lock:
            if not state.edit_group["loaded"]:
                _load_edit_group_locked()
    pipe = state.edit_group["edit_pipe"]
    progress_mod.disable_diffusers_tqdm(pipe)
    return pipe
