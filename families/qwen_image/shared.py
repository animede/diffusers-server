# -*- coding: utf-8 -*-
"""
共有コンポーネント(vae / text_encoder / tokenizer)のロード。

抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py
  - _load_shared_components_locked()(行774-801)
  - _warn_if_model_cpu_with_quant()(行928-941)

注意: 呼び出し側が families.qwen_image.state.lock を保持している前提("_locked" 系関数の規約)。
"""
import time

import torch
from diffusers import AutoencoderKLQwenImage
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from core.optimize import configure_shared_offload

from families.qwen_image import state
from families.qwen_image.paths import BASE_REPO


def load_shared_components_locked(small_transformer_active: bool = False) -> None:
    """vae / text_encoder / tokenizer を1回だけロードする(呼び出し側でロック保持前提)。

    抽出元: pipeline_manager.py _load_shared_components_locked()(行774-801)。ロジック無変更。
    """
    shared = state.shared
    if shared["loaded"]:
        return
    t0 = time.time()
    offload_mode = state.get_offload_mode(small_transformer_active=small_transformer_active)
    # from_pretrained() は既定で CPU にロードする。配置は configure_shared_offload() に一任する
    # (offload_mode="model_cpu" の場合はパイプライン単位の enable_model_cpu_offload() に任せる)。

    print(f"[families.qwen_image] loading shared components (vae/text_encoder/tokenizer) from {BASE_REPO}")
    vae = AutoencoderKLQwenImage.from_pretrained(BASE_REPO, subfolder="vae", torch_dtype=torch.bfloat16)
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_REPO, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    tokenizer = Qwen2Tokenizer.from_pretrained(BASE_REPO, subfolder="tokenizer")

    # 共有VAEのtiled encode/decode常時化(DS_QWEN_TILED_VAE=1既定、CLAUDE.md 56番)。
    # LTX2(CLAUDE.md 52番)と全く同じ構図: AutoencoderKLQwenImage._encode()/_decode() は
    # use_tiling を尊重してタイル分割経路(tiled_encode/tiled_decode)へ切り替わる
    # (autoencoder_kl_qwenimage.py の _encode()/_decode() 実装参照)。既定タイルサイズ
    # (256px、ストライド192px=64pxオーバーラップでブレンド)を使えば呼び出し側の変更は
    # 不要。1280x720キャンバスのControlNet Inpaint(/api/outpaint)は encode 単体で
    # 4.45GiB要求→OOMを実機確認済み(タイル閾値256pxを大きく超えるため確実にtiled化
    # される)。全パイプライン(t2i/i2i/edit/edit_angles/controlnet/controlnet_inpaint)
    # がこの共有VAEインスタンスを参照するため、ここで一度設定するだけで全モードに効く
    # (Layeredは専用VAEのため対象外。families/qwen_image/layered.py 参照、CLAUDE.md 8番)。
    if state.get_runtime_config().tiled_vae:
        vae.enable_tiling()
        print("[families.qwen_image] shared VAE tiled encode/decode enabled (DS_QWEN_TILED_VAE=1)")

    if offload_mode == "model_cpu":
        # パイプライン単位の enable_model_cpu_offload() に任せるため、ここでは配置しない。
        pass
    else:
        configure_shared_offload(vae, text_encoder, offload_mode)

    shared["vae"] = vae
    shared["text_encoder"] = text_encoder
    shared["tokenizer"] = tokenizer
    shared["loaded"] = True
    shared["load_time_s"] = time.time() - t0
    print(f"[families.qwen_image] shared components loaded in {shared['load_time_s']:.1f}s")


def warn_if_model_cpu_with_quant(quant, offload_mode) -> None:
    """DS_QUANT(GGUF) と DS_OFFLOAD=model_cpu の組み合わせは未サポート。
    共有コンポーネントが CPU に取り残されないよう明示的に GPU へ配置する。

    抽出元: pipeline_manager.py _warn_if_model_cpu_with_quant()(行928-941)。ロジック無変更。
    """
    shared = state.shared
    if quant and offload_mode == "model_cpu" and shared["loaded"]:
        print(
            "[families.qwen_image] warning: DS_QUANT(GGUF)とDS_OFFLOAD=model_cpuの組み合わせは"
            "未サポートです。共有コンポーネントを強制的にGPU常駐にします。"
        )
        if shared["vae"] is not None:
            shared["vae"].to("cuda")
        if shared["text_encoder"] is not None:
            shared["text_encoder"].to("cuda")
