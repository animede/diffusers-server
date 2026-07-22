# -*- coding: utf-8 -*-
"""
charsheet 専用: Edit の「bf16-group」変種(旧 /home/animede/charsheet 完全再現)のロード・
排他管理(CLAUDE.md 33番)。

目的: /home/animede/charsheet/pipeline.py と数値的に完全同等の品質を得ること。
edit_angles_bf16.py(fp8-base-adapters、既定の "bf16-adapters" 方式)は transformer の
ベースを fp8_e4m3fn へストレージ圧縮してから adapter を乗せる設計だったが、ユーザー決定
により「fp8化を一切行わず bf16 のまま常駐させ、代わりに block-level group offloading で
VRAM を賄う」旧方式へ切り替える。

絶対に守ること(CLAUDE.md 17番の事故の教訓、edit_angles_bf16.py 冒頭のdocstring参照):
transformer を丸ごと `.to("cpu")`/`.to("cuda")` する自前スワップは書かない。
`enable_group_offload()`(block_level、num_blocks_per_group=1、CUDA stream プリフェッチ)
のみを使う。旧実装の丸ごとスワップ(このモジュールとは別物、旧
enable_vae_transformer_swap 系)がホストRAMスワップ暴走でシステムフリーズを2回起こした
教訓を踏まえ、block-level group offload はブロック単位で必要な分だけホストRAM<->VRAMを
移動するため、丸ごとスワップとは事故のリスク構造が異なる(旧 charsheet/pipeline.py の
"group" モードとして実運用されていた実績がある)。

手順(旧 pipeline.py の構築フロー(行373-424)を忠実に再現):
  1. bf16 2511 transformer を **CPU へ** streaming ロード(cuda:0 へ直接ロードすると
     enable_group_offload の hook が効く前に transformer 本体(~38GB)だけで空きVRAMを
     使い切って OOM しうるため。旧実装 _load_pipeline_locked() のコメント(行381-384)
     と同じ理由。offload_mode != "none" の場合は transformer_load_device="cpu" に
     している)。
  2. pipe を構築(vae/text_encoder/tokenizer は共有コンポーネントを再利用、scheduler は
     BASE_REPO から)。
  3. Lightning + Multiple-angles LoRA を `pipe.load_lora_weights()` ×2 でロードし
     `pipe.set_adapters(["lightning","angles"], adapter_weights=[1.0, 1.0])` で有効化
     (fuseしない。旧 _apply_loras()(行271-279)と同一)。
  4. `pipe.transformer.enable_group_offload(...)`(block-level、旧
     _apply_group_offload_to_transformer() のパラメータをそのまま
     core.optimize.apply_group_offload_to_transformer_full() へ移植済み)。
  5. vae は常時 GPU 常駐(小さいため)。text_encoder も 48GB 環境では常時 GPU 常駐
     (旧実装の "group" モード相当。"group_lowvram" の text_encoder スワップは行わない、
     タスク指示どおり)。

重要な適用順序(旧実装のコメント(行390-409)を踏襲): "group"/"group_lowvram" モードでは
LoRA適用 -> group offload 適用 の順(_apply_loras() を先に呼んでから
_configure_pipeline_runtime() を呼ぶ)。これは旧実装の分岐そのままの順序であり、本実装も
同じ順序(pipe構築 -> LoRAロード・有効化 -> group offload 適用)を踏襲する。

ロード前のホストRAMガード(タスク指示、事故防止): transformer bf16(~38GB)を一時的に
CPU RAM へ確保するため、ロード開始前に空きRAM(MemAvailable)を確認し、
DS_CHARSHEET_GROUP_OFFLOAD_MIN_RAM_GB(既定42.0GB)を下回る場合は明確な RuntimeError で
中止する(黙ってスワップに突入しない)。

このグループ(state.edit_angles_bf16group_group)は edit_group / edit_angles_group /
edit_angles_bf16_group とは別物として管理し、同時常駐しない。排他は
families/qwen_image/family.py の load()/generate() 側で「edit / edit_angles /
edit_angles_bf16 / edit_angles_bf16group / t2i」の相互排他として実装する。
"""
import time

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from huggingface_hub import snapshot_download
from transformers import Qwen2VLProcessor

from core import gpu
from core.config import DS_CHARSHEET_GROUP_OFFLOAD_MIN_RAM_GB
from core.loaders import load_transformer_from_config
from core.optimize import apply_attention_backend, apply_compile, apply_group_offload_to_transformer_full
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


def _check_ram_guard() -> None:
    """ロード前にホスト RAM の空きを確認する(タスク指示: 黙ってスワップに突入しない)。"""
    available_gb = gpu.available_ram_gb()
    if available_gb is None:
        print(
            "[families.qwen_image] warning: /proc/meminfo からホストRAM空きを取得できません"
            "でした(RAMガードをスキップします)。"
        )
        return
    if available_gb < DS_CHARSHEET_GROUP_OFFLOAD_MIN_RAM_GB:
        raise RuntimeError(
            f"bf16-group 方式のロードには約38GB+余裕のホストRAMが必要ですが、現在の空きRAM "
            f"(MemAvailable)は {available_gb:.1f}GB しかありません"
            f"(閾値 DS_CHARSHEET_GROUP_OFFLOAD_MIN_RAM_GB={DS_CHARSHEET_GROUP_OFFLOAD_MIN_RAM_GB:.1f}GB)。"
            "他のプロセスを終了するか、閾値を調整してください。"
            "transformerを丸ごとホストRAMへ退避する設計のため、空きRAM不足のまま実行すると"
            "スワップ暴走によるシステムフリーズのリスクがあります(過去に2回発生した既知の事故)。"
        )
    print(
        f"[families.qwen_image] bf16-group RAM guard OK: available={available_gb:.1f}GB "
        f"(threshold={DS_CHARSHEET_GROUP_OFFLOAD_MIN_RAM_GB:.1f}GB)"
    )


def _load_edit_angles_bf16group_group_locked() -> None:
    """Edit(bf16-group)transformer をロードし、QwenImageEditPlusPipeline を構築する
    (共有コンポーネントを再利用)。旧 /home/animede/charsheet/pipeline.py
    _load_pipeline_locked() の "group" モード分岐(行373-424)を忠実に再現する。

    ロード順序: transformer を **CPU へ** streaming ロード(bf16、fuseしない、fp8化もしない)
    -> pipe構築 -> LoRA(lightning, angles)を adapter としてロード・有効化(fuseなし)
    -> transformer.enable_group_offload()(block-level)-> vae/text_encoder は共有
    コンポーネントとして GPU 常駐のまま。
    """
    group = state.edit_angles_bf16group_group
    if group["loaded"]:
        return
    t0 = time.time()

    _check_ram_guard()

    # 旧実装同様、group offload 系では transformer を直接 cuda:0 へロードしない
    # (enable_group_offload の hook が効く前に transformer 本体(~38GB)だけで空きVRAMを
    # 使い切って OOM しうるため。pipeline.py 行381-384 のコメントと同じ理由)。
    path = resolve_model_path(EDIT_TRANSFORMER_PATH, EDIT_TRANSFORMER_HF_REPO, EDIT_TRANSFORMER_HF_FILE)
    print(
        f"[families.qwen_image] loading Edit transformer for bf16-group from {path} "
        "(CPUへstreaming、fuseなし・fp8化なし、旧charsheet完全再現)"
    )
    transformer = load_transformer_from_config(
        QwenImageTransformer2DModel, EDIT_PROCESSOR_REPO, "transformer", path, "cpu"
    )
    print(
        f"[families.qwen_image] bf16-group transformer loaded to CPU "
        f"(host RAM available={gpu.available_ram_gb():.1f}GB)"
        if gpu.available_ram_gb() is not None else
        "[families.qwen_image] bf16-group transformer loaded to CPU"
    )

    # 共有コンポーネント(vae/text_encoder/tokenizer)をロードする(48GB環境では
    # text_encoder も GPU 常駐、旧実装の "group" モード相当。"group_lowvram" の
    # text_encoder スワップは使わない、タスク指示どおり)。
    load_shared_components_locked(small_transformer_active=False)
    runtime_config = state.get_runtime_config()
    warn_if_model_cpu_with_quant("bf16-group", runtime_config.offload)

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

    # LoRA を adapter としてロード・有効化(fuseしない。旧 _apply_loras() (pipeline.py
    # 行271-279)と同一: load_lora_weights ×2 -> set_adapters(weights=[1.0, 1.0])。
    # 旧実装は "group"/"group_lowvram" モードでは LoRA適用 -> group offload 適用の順
    # (pipeline.py 行393-395)だったため、本実装も同じ順序を踏襲する。
    lightning_path = resolve_model_path(
        EDIT_LORA_LIGHTNING_PATH, EDIT_LORA_LIGHTNING_HF_REPO, EDIT_LORA_LIGHTNING_HF_FILE
    )
    angles_path = resolve_model_path(EDIT_LORA_ANGLES_PATH, EDIT_LORA_ANGLES_HF_REPO, EDIT_LORA_ANGLES_HF_FILE)
    edit_pipe.load_lora_weights(lightning_path, adapter_name="lightning")
    edit_pipe.load_lora_weights(angles_path, adapter_name="angles")
    edit_pipe.set_adapters(["lightning", "angles"], adapter_weights=[1.0, 1.0])
    print("[families.qwen_image] bf16-group: LoRA adapters loaded (lightning+angles, weights=[1.0, 1.0])")

    # transformer を block-level group offload(旧実装のパラメータを完全移植)。
    group_offload_config = apply_group_offload_to_transformer_full(transformer)

    # vae は常時 GPU 常駐(旧実装 _apply_group_offload_to_transformer() 行326-330と同じ、
    # offload_all_components=False 相当。text_encoder も 48GB 環境では常駐のままにする)。
    if torch.cuda.is_available():
        state.shared["vae"].to("cuda")
        state.shared["text_encoder"].to("cuda")

    apply_attention_backend(transformer, runtime_config.attention_backend, runtime_config.device)
    apply_compile(transformer, runtime_config.compile)

    if torch.cuda.is_available():
        print(
            "[families.qwen_image] bf16-group ready, "
            f"VRAM={torch.cuda.memory_allocated()/1024**3:.2f}GB"
        )

    group["transformer"] = transformer
    group["edit_pipe"] = edit_pipe
    group["scheduler"] = scheduler
    group["processor"] = processor
    group["loaded"] = True
    group["load_time_s"] = time.time() - t0
    group["fallback"] = False
    group["quant"] = "bf16-group"
    group["lightning_merged"] = False  # adapter方式(fuse済みではない)
    group["group_offload_config"] = group_offload_config
    print(
        f"[families.qwen_image] Edit-angles-bf16group group loaded in {group['load_time_s']:.1f}s "
        f"(quant=bf16-group, group_offload={group_offload_config})"
    )


def set_scheduler_shift(scheduler, shift) -> None:
    """edit.py の set_scheduler_shift() に委譲する(重複実装だと修正漏れのリスクがあるため
    一本化)。"""
    from families.qwen_image.edit import set_scheduler_shift as _set_scheduler_shift

    _set_scheduler_shift(scheduler, shift)


def get_edit_angles_bf16group_pipeline() -> QwenImageEditPlusPipeline:
    if not state.edit_angles_bf16group_group["loaded"]:
        with state.lock:
            if not state.edit_angles_bf16group_group["loaded"]:
                _load_edit_angles_bf16group_group_locked()
    return state.edit_angles_bf16group_group["edit_pipe"]
