# -*- coding: utf-8 -*-
"""
charsheet 専用: Edit の「fp8-base-adapters」変種のロード・排他管理。

経緯・事故の教訓(重要、必ず読むこと): 当初この方式は「bf16 2511 transformer を
fuse・fp8化せず丸ごと常駐(~38GB)させ、Lightning + Multiple-angles を adapter として
適用する」設計で実装された。この bf16 常駐(~38GB)は 48GB VRAM に収まらず、text_encoder
(~16GB)や VAE decode の一時ピークと共存できなかったため、対処として「VAE encode/decode
中や text_encoder エンコード中に transformer(38GB)を CPU へ退避するスワップ機構」を
追加していた。この実装がホスト RAM(62GB)を圧迫し、**transformer 38GB + text_encoder
16GB + その他が同時に CPU 上に乗る瞬間にスワップ暴走 -> システムフリーズ -> OOM
キラーによる開発環境ごとの強制終了が2回発生**した。**transformer を CPU へ退避する
実装はこのマシンでは禁忌**(core/optimize.py の enable_vae_transformer_swap /
enable_text_encoder_cpu_offload_with_transformer_swap 系は完全に削除した。再実装しないこと)。

修正後の設計(本ファイル): 「fuseはしないが、ベースをfp8ストレージ化してから adapter を
乗せる」方式に変更した。手順(core.loaders.load_lora_adapters_and_cast_to_fp8() 参照):
  1. bf16 2511 transformer をロード(streaming、直接 cuda:0)
  2. Lightning + Multiple-angles を transformer に adapter としてロード・有効化
     (`transformer.load_lora_adapter()` ×2 -> `transformer.set_adapters(["lightning",
     "angles"], weights=[1.0, 1.0])`。旧 ~/charsheet/pipeline.py の
     _apply_loras()(行271-279)と同じ LoRA 計算。fuse はしない)
  3. その後 `enable_layerwise_casting(storage_dtype=fp8_e4m3fn, compute_dtype=bf16)`
     でベースのみ fp8 ストレージ化(実測 ~19-20GB)

**このロード順序(LoRA先→fp8キャスト後)は必須**(実機で特定したバグの回避):
先に fp8 キャストしてから LoRA をロードすると、PEFT が新規作成する lora_A/lora_B の
nn.Linear がベース層の「その時点の dtype」(=fp8)に合わせて作られてしまい、forward時に
`NotImplementedError: "addmm_cuda" not implemented for 'Float8_e4m3fn'` になることを
実機で再現・特定した(diffusers の layerwise-casting hook は「先に登録されていた
モジュール」の forward だけを差し替える実装のため、後から追加される lora_A/lora_B には
hook が適用されない)。詳細は core/loaders.py の
load_lora_adapters_and_cast_to_fp8() のdocstring参照。

品質面の根拠(CLAUDE.md 29番、コミット 9dbb0ab の調査時の対照実験): 「adapterモード
(fuseせずPEFT経由で加算、ベースのみfp8ストレージ)」はシャープな品質になることを実機
確認済み。LoRA計算はbf16フル精度で走る(PEFT の adapter forward は base の fp8
ストレージを compute_dtype=bf16 に都度キャストしてから LoRA 差分を加算する)ため、
fuse+fp8化で起きる「LoRA差分がfp8の量子化グリッドに丸め込まれて消える」問題が構造的に
起きない(fuseしていないのでLoRA差分はベースの重みに一切混ざらない)。

VRAM収支(fp8ベース化により純bf16常駐(~38GB)より大幅に軽くなる): ロード過程の一時
ピークは bf16 transformer(~38GB、LoRA adapter分を含めても~38.2GB程度)で fuse系と
ほぼ同じ。fp8キャスト後はベース ~19-20GB + adapter(bf16、2本、小さい) + VAE(~1GB)。
text_encoder(共有、~16GB)を足しても ~37GB で 48GB専有に収まる見込み。1024² の
VAE decode ピーク(+5GB)分は、既存の text_encoder CPU退避(DS_EDIT_TE_OFFLOAD、
通常のTEのみのswap、core.optimize.enable_text_encoder_cpu_offload/
ensure_text_encoder_on_gpu)で対応する。**transformer の CPU 退避は一切行わない**
(fp8化後は「小さいtransformer」として扱えるため、共有コンポーネントの GPU 配置は
edit_angles.py(fp8-lightning-angles)と同様の通常経路でよく、text_encoder を
特別扱いでCPUに留める必要もない)。

構築順序(fuse系の edit_angles.py と同じく、共有コンポーネントより先に transformer を
確保する。ロード過程の一時ピーク ~38GB を優先的に確保するため):
  bf16 2511 transformer ロード(直接 cuda:0)
  -> Lightning + Multiple-angles を adapter としてロード・有効化(fuseなし)
  -> enable_layerwise_casting(fp8_e4m3fn)(ベースのみ圧縮。adapterはbf16のまま)
  -> 共有コンポーネント(vae/text_encoder/tokenizer を再利用、通常配置)
  -> pipe 構築

このグループ(state.edit_angles_bf16_group)は edit_group・edit_angles_group とは
別物として管理し、同時常駐しない。排他は families/qwen_image/family.py の
load()/generate() 側で「edit / edit_angles / edit_angles_bf16 / t2i」の相互排他として
実装する(通常の /api/edit の挙動・重みには一切影響を与えない)。
"""
import time

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from huggingface_hub import snapshot_download
from transformers import Qwen2VLProcessor

from core.loaders import load_lora_adapters_and_cast_to_fp8, load_transformer_from_config
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


def _load_edit_angles_bf16_group_locked() -> None:
    """Edit(fp8-base-adapters)transformer をロードし、QwenImageEditPlusPipeline を構築する
    (共有コンポーネントを再利用)。

    ロード順序: transformer を先に直接 cuda:0 へロード(bf16、~38GB一時ピーク)
    -> LoRA adapter(lightning, angles)をロード・有効化(fuseなし、bf16のまま)
    -> enable_layerwise_casting(fp8_e4m3fn) でベースをストレージ圧縮(~19-20GBまで
    縮小、adapterはbf16のまま)-> 共有コンポーネントをロード(fp8化後は「小さい
    transformer」扱いでよいため、edit_angles.py の fp8-lightning-angles と同じ通常配置)。
    """
    edit_angles_bf16_group = state.edit_angles_bf16_group
    if edit_angles_bf16_group["loaded"]:
        return
    t0 = time.time()
    # fp8ベース化後は「小さいtransformer」として扱う(offload_modeの閾値判定・
    # 共有コンポーネントのGPU配置判断に使う。edit_angles.py の fp8-lightning-angles と同じ)。
    offload_mode = state.get_offload_mode(small_transformer_active=True)

    path = resolve_model_path(EDIT_TRANSFORMER_PATH, EDIT_TRANSFORMER_HF_REPO, EDIT_TRANSFORMER_HF_FILE)
    print(
        f"[families.qwen_image] loading Edit transformer for fp8-base-adapters from {path} "
        "(直接cuda:0へ、fuseなし)"
    )
    transformer = load_transformer_from_config(
        QwenImageTransformer2DModel, EDIT_PROCESSOR_REPO, "transformer", path, "cuda:0"
    )
    if torch.cuda.is_available():
        print(
            "[families.qwen_image] fp8-base-adapters transformer loaded (bf16, pre-cast), "
            f"VRAM={torch.cuda.memory_allocated()/1024**3:.2f}GB"
        )

    # LoRA adapter(lightning + angles、fuseなし)をロードしてから fp8 キャストする
    # (この順序は必須。core.loaders.load_lora_adapters_and_cast_to_fp8() のdocstring、
    # このモジュール上部のdocstring参照: 先にfp8化するとPEFTが新規作成するlora_A/lora_Bが
    # fp8のnn.Linearとして作られてしまい、forward時にNotImplementedErrorになる)。
    try:
        load_lora_adapters_and_cast_to_fp8(
            transformer,
            EDIT_LORA_LIGHTNING_PATH, EDIT_LORA_LIGHTNING_HF_REPO, EDIT_LORA_LIGHTNING_HF_FILE,
            EDIT_LORA_ANGLES_PATH, EDIT_LORA_ANGLES_HF_REPO, EDIT_LORA_ANGLES_HF_FILE,
            extra_adapter_name="angles", extra_weight=1.0,
        )
        edit_angles_bf16_group["lora_available"] = True
    except Exception as e:  # noqa: BLE001
        edit_angles_bf16_group["lora_available"] = False
        edit_angles_bf16_group["lora_unavailable_reason"] = f"{type(e).__name__}: {e}"[:300]
        print(
            f"[families.qwen_image] warning: Edit-angles-bf16(fp8-base-adapters) の adapter "
            f"LoRA(Lightning+angles)ロードに失敗しました: {e}。ベースのみfp8化してフォールバックします。"
        )
        transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if torch.cuda.is_available():
        print(
            "[families.qwen_image] fp8-base-adapters transformer ready, "
            f"VRAM={torch.cuda.memory_allocated()/1024**3:.2f}GB"
        )

    # fp8化により transformer は「小さい」ため、共有コンポーネント(text_encoder等)は
    # 通常どおりGPUへ配置してよい(edit_angles.py と同じ経路。特別なCPU留め置きは不要)。
    load_shared_components_locked(small_transformer_active=True)
    warn_if_model_cpu_with_quant("fp8-base-adapters", offload_mode)

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

    edit_angles_bf16_group["transformer"] = transformer
    edit_angles_bf16_group["edit_pipe"] = edit_pipe
    edit_angles_bf16_group["scheduler"] = scheduler
    edit_angles_bf16_group["processor"] = processor
    edit_angles_bf16_group["loaded"] = True
    edit_angles_bf16_group["load_time_s"] = time.time() - t0
    edit_angles_bf16_group["fallback"] = False
    edit_angles_bf16_group["quant"] = "fp8-base-adapters"
    edit_angles_bf16_group["lightning_merged"] = False  # adapter方式(fuse済みではない)
    print(
        f"[families.qwen_image] Edit-angles-bf16 group loaded in {edit_angles_bf16_group['load_time_s']:.1f}s "
        "(quant=fp8-base-adapters)"
    )


def set_scheduler_shift(scheduler, shift) -> None:
    """edit.py の set_scheduler_shift() に委譲する(重複実装だと修正漏れのリスクがあるため
    一本化。use_dynamic_shifting時の base_shift/max_shift 上書き修正の詳細は edit.py の
    docstring 参照)。"""
    from families.qwen_image.edit import set_scheduler_shift as _set_scheduler_shift

    _set_scheduler_shift(scheduler, shift)


def get_edit_angles_bf16_pipeline() -> QwenImageEditPlusPipeline:
    if not state.edit_angles_bf16_group["loaded"]:
        with state.lock:
            if not state.edit_angles_bf16_group["loaded"]:
                _load_edit_angles_bf16_group_locked()
    return state.edit_angles_bf16_group["edit_pipe"]
