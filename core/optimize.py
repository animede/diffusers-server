# -*- coding: utf-8 -*-
"""
offload 自動判定 / attention backend 切替 / torch.compile 適用の共通関数群。

抽出元:
  - Ｑwenimage-edit-diffusers/pipeline_manager.py
    - _auto_offload_mode() (行361-368)
    - _OFFLOAD_ALIASES / _VALID_OFFLOAD_MODES (行371-387)
    - _resolve_offload_mode() (行390-404)
    - _smoke_test_attention_backend() (行522-529)
    - apply_attention_backend() (行532-561)
    - apply_compile() (行564-583)
    - _apply_group_offload_to_transformer() (行589-599)
    - _apply_whole_module_swap_offload() (行602-610)
    - configure_transformer_offload() (行613-623)
    - configure_shared_offload() (行646-657)
  - flux2_diffusers/pipeline_manager.py
    - KNOWN_GOOD_ATTN_BACKENDS / KNOWN_BAD_ATTN_BACKEND_PREFIXES (行54-70、
      Ｑwenimage-edit-diffusers 側と同一の知見。core.config に統合済み)
    - _apply_offload() (flux2 独自の "model"/"none"/"group" 三択、行138-181) — Flux2Pipeline は
      enable_model_cpu_offload() をパイプライン単位で持つため、Qwen系の
      transformer 単位オフロードとは別に flux2 固有の適用関数として残す
    - _apply_attention_backend() / _apply_compile() (行210-289、Qwen 側とほぼ同一実装)

方針:
  - offload 自動判定・attention backend・compile の「判定ロジック」は Qwen 側(より詳細な
    group/group_lowvram/model_cpu の三段階 + GGUF閾値緩和)をベースに共通化する。
    flux2 側は model_cpu_offload が既定でパイプライン単位のため、Phase 2 で
    families/flux2.py 側から個別に呼び出す薄いラッパー(apply_flux2_offload)を用意する。
  - sm_120 で sage 系 attention 不可の知見は core.config の KNOWN_GOOD_ATTN_BACKENDS /
    KNOWN_BAD_ATTN_BACKEND_PREFIXES に統合済み(両ワークスペースで完全一致していたため)。
  - torch の import はモジュールトップレベルで行うが、CUDA 初期化を伴う処理(smoke test 等)は
    関数内に閉じ込める。
"""
import traceback

import torch

from core.config import (
    DS_CHARSHEET_GROUP_OFFLOAD_BLOCKS,
    DS_CHARSHEET_GROUP_OFFLOAD_DISK_PATH,
    DS_CHARSHEET_GROUP_OFFLOAD_LOW_CPU_MEM,
    DS_CHARSHEET_GROUP_OFFLOAD_NON_BLOCKING,
    DS_CHARSHEET_GROUP_OFFLOAD_RECORD_STREAM,
    DS_CHARSHEET_GROUP_OFFLOAD_USE_STREAM,
    EDIT_TE_OFFLOAD_AREA_THRESHOLD,
    EDIT_TE_OFFLOAD_FREE_VRAM_GB_THRESHOLD,
    GGUF_VRAM_FREE_THRESHOLD_GB,
    KNOWN_BAD_ATTN_BACKEND_PREFIXES,
    KNOWN_GOOD_ATTN_BACKENDS,
    VRAM_FREE_THRESHOLD_GB,
    VRAM_LOW_THRESHOLD_GB,
    get_edit_te_offload_mode,
)
from core.gpu import free_vram_gb

__all__ = [
    "auto_offload_mode",
    "resolve_offload_mode",
    "apply_attention_backend",
    "apply_compile",
    "apply_group_offload_to_transformer",
    "apply_group_offload_to_transformer_full",
    "apply_whole_module_swap_offload",
    "configure_transformer_offload",
    "configure_shared_offload",
    "apply_flux2_offload",
    "ensure_text_encoder_on_gpu",
    "enable_text_encoder_cpu_offload",
    "disable_text_encoder_cpu_offload",
    "should_offload_edit_text_encoder",
]


# ============================================================================
# offload 自動判定(Qwen 系: none / group / group_lowvram の三段階)
# ============================================================================
def auto_offload_mode(
    free_gb: float,
    free_threshold: "float | None" = None,
    low_threshold: "float | None" = None,
) -> str:
    """空き VRAM(GB)からオフロードモードを自動判定する。

    抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py の _auto_offload_mode()(行361-368)。
    """
    free_threshold = VRAM_FREE_THRESHOLD_GB if free_threshold is None else free_threshold
    low_threshold = VRAM_LOW_THRESHOLD_GB if low_threshold is None else low_threshold
    if free_gb >= free_threshold:
        return "none"
    if free_gb >= low_threshold:
        return "group"
    return "group_lowvram"


_OFFLOAD_ALIASES = {
    "cpu": "model_cpu",
    "cpu_offload": "model_cpu",
    "model": "model_cpu",
    "model_cpu_offload": "model_cpu",
    "group_offload": "group",
    "group_low": "group_lowvram",
    "group_low_vram": "group_lowvram",
    "lowvram": "group_lowvram",
    "low_vram": "group_lowvram",
    "off": "none",
    "cuda": "none",
    "full_cuda": "none",
    "disable": "none",
    "disabled": "none",
}
_VALID_OFFLOAD_MODES = {"model_cpu", "group", "group_lowvram", "none"}


def resolve_offload_mode(requested: str, free_gb: float, small_transformer_active: bool = False) -> str:
    """明示指定・エイリアス・"auto" のいずれも解決して最終的なオフロードモードを返す。

    抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py の _resolve_offload_mode()
    (行390-404)。small_transformer_active=True の場合、GGUF 等の小さい transformer 向けに
    "none" 判定の閾値を GGUF_VRAM_FREE_THRESHOLD_GB まで緩和する(元実装のコメント:
    行264-269 参照)。
    """
    requested = requested.strip().lower().replace("-", "_")
    requested = _OFFLOAD_ALIASES.get(requested, requested)
    free_threshold = GGUF_VRAM_FREE_THRESHOLD_GB if small_transformer_active else None
    if requested == "auto":
        return auto_offload_mode(free_gb, free_threshold=free_threshold)
    if requested not in _VALID_OFFLOAD_MODES:
        print(f"[core.optimize] unknown offload mode {requested!r}; falling back to auto")
        return auto_offload_mode(free_gb, free_threshold=free_threshold)
    if requested in {"group", "group_lowvram", "none"} and not torch.cuda.is_available():
        print(f"[core.optimize] CUDA unavailable; {requested} mode falls back to CPU execution")
        return "none"
    return requested


# ============================================================================
# attention backend(Qwen / flux2 共通。sm_120 で sage 系不可の知見を統合済み)
# ============================================================================
def _smoke_test_attention_backend(backend: str, device: str) -> None:
    """カーネル起動時にのみ失敗するバックエンド(sm_120非対応の sage等)を検出する。

    抽出元: 両ワークスペースでほぼ同一実装
      - Ｑwenimage-edit-diffusers/pipeline_manager.py 行522-529
      - flux2_diffusers/pipeline_manager.py 行184-207(shape の根拠コメントも同じ趣旨)
    shape は「batch 1, seq 4608(~1024x1024 相当) , 24 heads, head_dim 128」で、
    sage の失敗パスが長seqでのみ顕在化する点を踏まえた生成時に近い形状。
    """
    from diffusers.models.attention_dispatch import AttentionBackendName, dispatch_attention_fn

    q = torch.randn(1, 4608, 24, 128, dtype=torch.bfloat16, device=device)
    out = dispatch_attention_fn(q, q, q, backend=AttentionBackendName(backend))
    torch.cuda.synchronize()
    del q, out


def apply_attention_backend(transformer, backend: str, device: str = "cuda") -> None:
    """transformer の attention backend を設定する。失敗時は警告を出して default のまま。

    抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py の apply_attention_backend()
    (行532-561)。flux2_diffusers 側の _apply_attention_backend()(行210-248)も同一ロジック。
    """
    backend = (backend or "default").lower()
    if backend in ("default", ""):
        return

    if backend.startswith(KNOWN_BAD_ATTN_BACKEND_PREFIXES):
        print(
            f"[core.optimize] warning: attention backend {backend!r} is known to be broken "
            f"on this GPU (sm_120 kernels missing); falling back to default SDPA."
        )
        return

    if backend not in KNOWN_GOOD_ATTN_BACKENDS:
        print(
            f"[core.optimize] note: attention backend {backend!r} is not in the "
            f"verified-working list {list(KNOWN_GOOD_ATTN_BACKENDS)}; trying anyway."
        )
    try:
        transformer.set_attention_backend(backend)
        _smoke_test_attention_backend(backend, device)
        print(f"[core.optimize] attention backend set to {backend!r}")
    except Exception as e:  # noqa: BLE001
        print(
            f"[core.optimize] warning: attention backend {backend!r} is not usable "
            f"({type(e).__name__}: {e}); falling back to default SDPA."
        )
        try:
            transformer.reset_attention_backend()
        except Exception:  # noqa: BLE001
            pass


def apply_compile(transformer, enabled: bool) -> None:
    """transformer を torch.compile する(リージョナルコンパイル優先、失敗時は通常コンパイル)。

    抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py の apply_compile()(行564-583)。
    flux2_diffusers 側の _apply_compile()(行251-289、bnb 4bit の data-dependent branch に
    関する注記あり)も同一ロジック。
    """
    if not enabled:
        return
    try:
        transformer.compile_repeated_blocks(fullgraph=False, dynamic=False)
        print("[core.optimize] transformer compiled (regional, compile_repeated_blocks)")
        return
    except Exception as e:  # noqa: BLE001
        print(
            f"[core.optimize] regional compilation failed ({type(e).__name__}: {e}); "
            f"falling back to plain torch.compile"
        )
        traceback.print_exc()

    try:
        transformer.compile()
        print("[core.optimize] transformer compiled (plain torch.compile)")
    except Exception as e:  # noqa: BLE001
        print(f"[core.optimize] warning: torch.compile failed ({type(e).__name__}: {e}); continuing uncompiled.")


# ============================================================================
# transformer / 共有コンポーネント単位のオフロード適用(Qwen 系)
# ============================================================================
def apply_group_offload_to_transformer(transformer, num_blocks: int = 1) -> None:
    """抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py の
    _apply_group_offload_to_transformer()(行589-599)。
    """
    print(f"[core.optimize] enabling transformer group offload (num_blocks_per_group={num_blocks})")
    transformer.enable_group_offload(
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=num_blocks,
        non_blocking=True,
        use_stream=True,
        low_cpu_mem_usage=True,
    )


def apply_group_offload_to_transformer_full(transformer) -> dict:
    """transformer を block-level group offloading で GPU/CPU 入れ替える(旧
    ~/charsheet/pipeline.py の _apply_group_offload_to_transformer()
    (行282-334)の完全移植、"bf16-group" 方式専用)。

    apply_group_offload_to_transformer()(このファイル上部の簡易版、num_blocks のみ
    引数化)との違い: 旧実装が持っていた全ノブ(use_stream / non_blocking /
    record_stream / low_cpu_mem_usage / offload_to_disk_path)を DS_* 環境変数
    (DS_CHARSHEET_GROUP_OFFLOAD_*、core.config 参照)からそのまま復元し、
    transformer.enable_group_offload() に一言一句同じ引数構成で渡す。既存の簡易版は
    他の呼び出し元(GGUF等の "group" オフロードモード)に影響するため変更しない。

    抽出元: ~/charsheet/pipeline.py の
    _apply_group_offload_to_transformer()(行282-334)。ロジック・既定値とも無変更
    (offload_all_components 分岐は "bf16-group" では常に False 相当 = vae/text_encoder
    は呼び出し側で別途 GPU 常駐させる。48GB環境では group_lowvram 相当の text_encoder
    スワップは使わない、タスク指示どおり)。
    """
    num_blocks = int(DS_CHARSHEET_GROUP_OFFLOAD_BLOCKS)
    use_stream = DS_CHARSHEET_GROUP_OFFLOAD_USE_STREAM
    non_blocking_raw = DS_CHARSHEET_GROUP_OFFLOAD_NON_BLOCKING
    non_blocking = (
        use_stream
        if non_blocking_raw == ""
        else non_blocking_raw.strip().lower() in {"1", "true", "yes", "on"}
    )
    record_stream = DS_CHARSHEET_GROUP_OFFLOAD_RECORD_STREAM
    low_cpu_mem_usage = DS_CHARSHEET_GROUP_OFFLOAD_LOW_CPU_MEM
    offload_to_disk_path = DS_CHARSHEET_GROUP_OFFLOAD_DISK_PATH or None
    if offload_to_disk_path:
        import os as _os

        _os.makedirs(offload_to_disk_path, exist_ok=True)

    config = {
        "component": "transformer",
        "offload_type": "block_level",
        "num_blocks_per_group": num_blocks,
        "use_stream": use_stream,
        "non_blocking": non_blocking,
        "record_stream": record_stream,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "offload_to_disk_path": offload_to_disk_path,
    }
    print(f"[core.optimize] enabling transformer group offload (bf16-group): {config}")
    transformer.enable_group_offload(
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=num_blocks,
        non_blocking=non_blocking,
        use_stream=use_stream,
        record_stream=record_stream,
        low_cpu_mem_usage=low_cpu_mem_usage,
        offload_to_disk_path=offload_to_disk_path,
    )
    return config


def apply_whole_module_swap_offload(module) -> None:
    """module(vae や text_encoder)を使用直前にGPUへ、使用直後にCPUへ丸ごと入れ替える。

    抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py の
    _apply_whole_module_swap_offload()(行602-610)。
    """
    from accelerate import cpu_offload_with_hook

    _, hook = cpu_offload_with_hook(module, execution_device=torch.device("cuda"))

    def _offload_after_forward(_module, _args, output):
        hook.offload()
        return output

    module.register_forward_hook(_offload_after_forward)


def configure_transformer_offload(transformer, offload_mode: str) -> None:
    """transformer 単体へのオフロード適用(none / group / group_lowvram で共通処理)。
    model_cpu は呼び出し側でパイプライン単位に enable_model_cpu_offload() を使う。

    抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py の configure_transformer_offload()
    (行613-623)。
    """
    if offload_mode == "none":
        transformer.to("cuda")
    elif offload_mode == "group":
        apply_group_offload_to_transformer(transformer)
    elif offload_mode == "group_lowvram":
        apply_group_offload_to_transformer(transformer)
    # "model_cpu" はここでは何もしない(呼び出し側でパイプライン単位に処理)


def configure_shared_offload(vae, text_encoder, offload_mode: str) -> None:
    """共有コンポーネント(vae / text_encoder)へのオフロード適用。プロセス内で一度だけ呼ぶ。

    抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py の configure_shared_offload()
    (行646-657)。
    """
    if offload_mode in ("none", "group"):
        # vae は小さいので常時 GPU 常駐。text_encoder も group では常駐のままにする
        # (transformer のみブロック単位でオフロードする方が効果的なため)。
        vae.to("cuda")
        text_encoder.to("cuda")
    elif offload_mode == "group_lowvram":
        # 24GB級カード向け: text_encoder(bf16で~16GB)も使用直前/直後でGPU<->CPU入れ替え。
        vae.to("cuda")
        apply_whole_module_swap_offload(text_encoder)
    # "model_cpu" はここでは何もしない(呼び出し側でパイプライン単位に処理)


# ============================================================================
# flux2 固有: パイプライン単位オフロード(model_cpu_offload / none / group)
# ============================================================================
def apply_flux2_offload(pipe, mode: str, device: str = "cuda", group_stream: bool = True) -> None:
    """Flux2Pipeline 全体へのオフロード適用。

    抽出元: flux2_diffusers/pipeline_manager.py の _apply_offload()(行138-181)。
    Flux2Pipeline は enable_model_cpu_offload() をパイプライン単位に持つため、Qwen 系の
    transformer 単位オフロード(configure_transformer_offload)とは別関数にしている。

    - "model" / "model_cpu_offload"(既定): enable_model_cpu_offload()
    - "none": 全コンポーネントを GPU 常駐
    - "group" / "group_offload": leaf_level の group offload
      (group_stream=True: CUDA-stream prefetch、CPU RAM にピン止めされるため bf16 では
       RAM を超える可能性がある。group_stream=False: 同期転送、mmap 経由でディスクから
       ストリームするため遅いが RAM 超過時でも動く。flux2_diffusers 行156-172 のコメント参照)
    """
    mode = (mode or "model").lower()
    if mode in ("model", "model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    elif mode == "none":
        pipe.to(device)
    elif mode in ("group", "group_offload"):
        from diffusers.hooks import apply_group_offloading

        onload_device = torch.device(device)
        offload_device = torch.device("cpu")
        common = dict(
            onload_device=onload_device,
            offload_device=offload_device,
            offload_type="leaf_level",
            use_stream=group_stream,
        )
        # transformer は diffusers ModelMixin -> ネイティブメソッド
        pipe.transformer.enable_group_offload(**common)
        # text_encoder(Mistral3 等)は transformers モデル -> functional API
        apply_group_offloading(pipe.text_encoder, **common)
        # VAE は小さいので常駐
        pipe.vae.to(onload_device)
        print(f"[core.optimize] group offload enabled (leaf_level, use_stream={group_stream})")
    else:
        raise ValueError(f"Unknown offload mode: {mode!r}")


# ============================================================================
# Edit系: text_encoder の CPU 退避(プロンプトエンコード後にCPUへ、次回エンコード前にGPUへ)
# ============================================================================
# 背景(REBUILD_PLAN §4.5 / タスク指示): 48GB専有では共有 text_encoder(~16GB)を
# VRAM常駐させたまま Edit の標準解像度(1024²相当)を生成すると、VAE decode の一時ピーク
# (~5GB)が乗って OOM する(CLAUDE.md 3番)。fp8-lightning transformer(~20GB)+ VAE +
# decodeピークだけなら1024²は収まる見込みのため、「テキストエンコード完了後に
# text_encoder を CPU へ退避し、denoise/VAE decode の間は空けておく」方式で対処する。
#
# 実装方式の選定(タスク指示: fp8 layerwise cast 済み transformer と
# enable_model_cpu_offload() の相性は未検証のため、確実に動く方法を実測で選ぶこと):
# QwenImageEditPlusPipeline.__call__()(diffusers/pipelines/qwenimage/
# pipeline_qwenimage_edit_plus.py)はプロンプトエンコード(self.encode_prompt()、
# true_cfg_scale>1 なら正負2回)を prepare_latents/denoise/decode より必ず先に呼ぶ
# 一枚岩の実装で、エンコードと denoise の間にコールバックフックが無い。そのため
# pipe.encode_prompt を実行時にラップし、「呼び出し後に text_encoder を CPU へ退避 +
# empty_cache」を行う手動 swap 方式を採用した(model_cpu_offload はパイプライン全体の
# hook 機構を使うため、transformer 側の enable_layerwise_casting(fp8_e4m3fn) の hook と
# 競合するリスクがあり、手動 swap の方が確実に安全)。
#
# 重要(実機デバッグで特定): text_encoder の GPU 復帰は encode_prompt 呼び出し内では
# 手遅れ。QwenImageEditPlusPipeline.__call__() は encode_prompt を呼ぶより前に
# `device = self._execution_device` を解決し、この device を encode_prompt に明示的に
# 渡す。_execution_device は「最初に見つかった nn.Module の現在地」を返す実装のため、
# text_encoder が CPU に残ったまま pipe(...) を呼ぶと device 自体が "cpu" に解決されて
# しまい、encode_prompt 内で text_encoder を GPU へ戻しても processor の出力
# (input_ids 等)は既に "cpu" 向けに作られてしまう(device mismatch エラーになることを
# 実機で再現・特定した)。そのため GPU 復帰は ensure_text_encoder_on_gpu() として分離し、
# 呼び出し側(families/qwen_image/generate.py)が pipe(...) を呼ぶ直前に明示的に呼ぶ。
#
# encode_prompt は true_cfg_scale>1 かつ negative_prompt 指定時に正負で2回連続して
# 呼ばれる(その間に denoise 等の重い処理は挟まらない)。呼び出しごとに素朴に
# CPU<->GPU を往復すると2回分の往復コストがかかるが、実装を単純にして「denoise/decode
# 開始前に必ず text_encoder が CPU にある」ことを保証する方を優先する(タスク指示:
# 「TE往復のオーバーヘッドは許容」)。


def should_offload_edit_text_encoder(width: "int | None", height: "int | None") -> bool:
    """DS_EDIT_TE_OFFLOAD(auto/on/off)から、この生成で text_encoder CPU退避を使うか判定する。

    auto の主判定は要求解像度(height*width、未指定なら参照画像から自動推定される
    1024²相当とみなす)。EDIT_TE_OFFLOAD_AREA_THRESHOLD(既定 640*640=409600px)以上なら
    退避を有効にする。空きVRAMは「現在どれだけ空いているか」であって「この生成が
    どれだけVRAMを必要とするか」の代理指標にはならない(このプロセスの初回ロード前は
    常に空き~47GBだが、1024²生成自体がTE常駐のままだとOOMする実績があるため、
    空きVRAMの大小で判定を反転させてはいけない)。そのため空きVRAMは「大容量カードでは
    退避自体が不要」というエスケープハッチとしてのみ使う: 空きVRAMが
    EDIT_TE_OFFLOAD_FREE_VRAM_GB_THRESHOLD(既定 44.0GB、この環境の48GB専有相当)を
    大きく超える場合(+TE分16GBの余裕を見て、閾値+16GB以上)は解像度に関わらず退避不要と
    判定する。

    charsheet 経路(既定 CHARSHEET_EDIT_SIZE=1024)は640*640閾値を超えるため auto で
    実質 on になる(タスク指示どおり)。
    """
    mode = get_edit_te_offload_mode("auto")
    if mode == "on":
        return True
    if mode == "off":
        return False
    # auto: 要求解像度から総画素数を見積もる(未指定なら自動推定と同じ1024²相当とみなす)。
    area = (width or 1024) * (height or 1024)
    if area < EDIT_TE_OFFLOAD_AREA_THRESHOLD:
        return False
    # 大容量VRAM環境(空きが閾値+TE分を大きく超える)では退避しなくても収まるため不要と判定。
    free_gb = free_vram_gb()
    return free_gb < (EDIT_TE_OFFLOAD_FREE_VRAM_GB_THRESHOLD + 16.0)


def ensure_text_encoder_on_gpu(pipe) -> None:
    """text_encoder が CPU にあれば GPU へ戻す(呼び出し側が pipe(...) を呼ぶ直前に使う)。

    重要な注意(実機デバッグで判明): QwenImageEditPlusPipeline.__call__() は
    `device = self._execution_device` を **encode_prompt を呼ぶより前**(prompt_embeds
    計算より前の行)で解決する。`self._execution_device`(= `self.device`)は
    `_get_signature_keys()` 順で最初に見つかった `nn.Module`(text_encoder 等)の
    `.device` を返す実装のため、pipe(...) 呼び出し時点で text_encoder が CPU に
    残っていると、__call__ 内で解決される `device` 変数自体が "cpu" になってしまう。
    この `device` は encode_prompt() に明示的に渡され、processor の出力(input_ids 等)を
    `.to(device)` する際に使われるため、encode_prompt 側(_wrapped_encode_prompt)で
    text_encoder を GPU へ戻すだけでは手遅れ(input_ids が cpu、embed_tokens.weight が
    cuda という食い違いで `RuntimeError: Expected all tensors to be on the same device`
    になることを実機で再現・特定した)。そのため text_encoder の GPU 復帰は
    **pipe(...) を呼ぶ直前**に呼び出し側(families/qwen_image/generate.py の
    run_edit()/run_edit_angles())で明示的に行う。
    """
    if not torch.cuda.is_available():
        return
    text_encoder = pipe.text_encoder
    if next(text_encoder.parameters()).device.type != "cuda":
        text_encoder.to("cuda")


def enable_text_encoder_cpu_offload(pipe) -> None:
    """pipe.encode_prompt をラップし、呼び出し後に text_encoder を CPU へ退避する
    (手動 swap 方式)。同一 pipe に複数回呼んでも二重ラップしない。

    呼び出し側の責務(重要): pipe(...) を呼ぶ**前**に ensure_text_encoder_on_gpu(pipe) を
    別途呼ぶこと。__call__ 内の `_execution_device` 解決タイミングの都合上、
    encode_prompt 呼び出し時点で GPU へ戻すのでは遅い(上記 ensure_text_encoder_on_gpu()
    のdocstring参照)。
    """
    if getattr(pipe, "_ds_te_offload_enabled", False):
        return
    if not torch.cuda.is_available():
        return

    original_encode_prompt = pipe.encode_prompt

    def _wrapped_encode_prompt(*args, **kwargs):
        # 2026-07-18バグ修正: 呼び出し冒頭で毎回 text_encoder を GPU へ戻す。
        # true_cfg_scale>1 かつ negative_prompt 指定時、QwenImagePipeline 系は
        # encode_prompt を正・負で2回連続して呼ぶが、旧実装は1回目の finally で
        # TE を CPU へ退避したまま2回目に入るため、2回目の embed_tokens が CPU、
        # input_ids が cuda(__call__ 冒頭で解決済みの device)という食い違いで
        # "Expected all tensors to be on the same device ... (wrapper_CUDA__index_select)"
        # になる(実機再現済み: 無印T2I・lightning=false・UI既定negative_prompt で発症。
        # Lightning時は cfg=1.0 で negative が無効になり1回しか呼ばれないため発症しない)。
        # 冒頭でGPUへ戻すことでN回連続エンコードでも安全になる(呼び出し後にCPUへ退避、
        # は従来どおり維持。往復コストは許容方針)。
        if torch.cuda.is_available() and next(pipe.text_encoder.parameters()).device.type != "cuda":
            pipe.text_encoder.to("cuda")
        try:
            result = original_encode_prompt(*args, **kwargs)
        finally:
            pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()
        return result

    pipe.encode_prompt = _wrapped_encode_prompt
    pipe._ds_te_offload_enabled = True
    pipe._ds_te_offload_original_encode_prompt = original_encode_prompt
    print("[core.optimize] text_encoder CPU offload enabled (manual swap around encode_prompt)")


def disable_text_encoder_cpu_offload(pipe) -> None:
    """enable_text_encoder_cpu_offload() で施したラップを解除し、text_encoder を GPU へ戻す。"""
    if not getattr(pipe, "_ds_te_offload_enabled", False):
        return
    pipe.encode_prompt = pipe._ds_te_offload_original_encode_prompt
    del pipe._ds_te_offload_original_encode_prompt
    pipe._ds_te_offload_enabled = False
    if torch.cuda.is_available():
        pipe.text_encoder.to("cuda")
        torch.cuda.empty_cache()
    print("[core.optimize] text_encoder CPU offload disabled (text_encoder back on GPU)")
