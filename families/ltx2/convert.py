# -*- coding: utf-8 -*-
"""
LTX-2.3 ComfyUI形式 scaled-fp8 チェックポイント(単一ファイル、28GB)を diffusers の
コンポーネント別 state_dict へ変換するロジック。

抽出元: tools/ltx2_regress.py(932行、コミット済みのスタンドアロン回帰確認スクリプト)。
ロジック・コメントとも実証済みの内容を無変更で移植する(CLAUDE.md 新知見番号参照)。

このモジュールが担う変換(diffusers 2.3非対応4件のパッチ):
  1. VAE: up_blocks 4段デコーダ(diffusers本体の convert_ltx2_vae_to_diffusers は
     LTX-2.0の3段用マッピングしか持たず、up_blocks.7/8 が欠落する)
  2. transformer: prompt_adaln_single./audio_prompt_adaln_single. のリネーム漏れ
     (diffusers本体の変換関数は adaln_single./audio_adaln_single. のみ対応)
  3. connectors: video/audio_embeddings_connector.*(transformer本体に埋め込み)を
     LTX2TextConnectors 専用の state_dict へ振り分け(diffusers本体は単純に破棄する)
  4. vocoder: diffusers に single_file 変換関数が一切無いため独自実装
     (conv_pre->conv_in 等のリネーム)

加えて、ComfyUI scaled-fp8 形式のデクオンタイズ(fp8_e4m3fn の weight + 対になる
per-tensor weight_scale(F32スカラー)から bf16 = fp8_weight.to(f32) * scale を復元)を
1テンソルずつ safe_open 経由でストリーミング処理する(RAM安全性、モジュールdocstring
下部参照)。

RAM安全性: 28GBのチェックポイントを丸ごとRAMに読まない。safetensors.safe_open は
mmapベースで get_tensor() 呼び出し時点で該当テンソル分だけ読む実装のため、1テンソル
ずつ処理してすぐ対象モデルへ .to(device) することで、常時RAMに保持するのは
「現在処理中の1テンソル」+「構築中の空(meta)モデルの実体化済み部分」のみに抑える
(core/loaders.py の load_safetensors_streaming と同じ方針)。
"""
import gc
import os
import time

import torch
from safetensors import safe_open

__all__ = [
    "stream_dequant_state_dict",
    "count_fp8_in_prefix",
    "convert_ltx2_vae_to_diffusers_23",
    "load_state_dict_streaming_to_device",
    "build_transformer_state_dict",
    "build_connectors_state_dict",
    "convert_ltx2_vocoder_to_diffusers",
    "convert_gemma_checkpoint_to_transformers",
    "load_vae_pair",
    "load_transformer_and_connectors",
    "load_vocoder",
    "load_text_encoder",
    "load_upsampler",
    "load_iclora_state_dict",
    "LTX2_TE_FP8_SKIP_MODULES_PATTERN",
    "LTX2_UPSAMPLER_CONFIG_OVERRIDES",
]


def log(msg: str) -> None:
    print(f"[families.ltx2.convert] {msg}", flush=True)


def ram_available_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / 1024**2
    except (OSError, ValueError, IndexError):
        pass
    return -1.0


# ---------------------------------------------------------------------------
# 1テンソルずつ読む共通ヘルパ(dequant対応版 load_safetensors_streaming)
# ---------------------------------------------------------------------------

def stream_dequant_state_dict(path: str, prefix: str, strip_prefix: str = ""):
    """safetensors から prefix に一致するキーだけを1テンソルずつ読み、fp8+weight_scale
    ペアがあれば dequant(bf16 = fp8.to(f32) * scale)、それ以外はそのまま bf16 キャストして
    dict[str, torch.Tensor] を返す(cpu上に構築、呼び出し側が個別に .to(device) する)。

    RAM安全性: 返す dict はこの prefix 配下のテンソルのみ保持する(チェックポイント全体は
    保持しない)。呼び出し側で使い終わったら del して解放すること。
    """
    result = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if k.startswith(prefix)]
        scale_keys = set(k for k in keys if k.endswith(".weight_scale"))
        for k in keys:
            if k in scale_keys:
                continue
            new_key = k[len(strip_prefix):] if strip_prefix and k.startswith(strip_prefix) else k
            tensor = f.get_tensor(k)
            base = k.rsplit(".weight", 1)[0] + ".weight_scale" if k.endswith(".weight") else None
            if base is not None and base in scale_keys:
                scale = f.get_tensor(base)
                tensor = (tensor.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)
            else:
                tensor = tensor.to(torch.bfloat16) if tensor.dtype in (torch.float32, torch.bfloat16, torch.float16) else tensor
            result[new_key] = tensor
    return result


def count_fp8_in_prefix(path: str, prefix: str):
    with safe_open(path, framework="pt") as f:
        keys = [k for k in f.keys() if k.startswith(prefix)]
        fp8 = [k for k in keys if f.get_slice(k).get_dtype() == "F8_E4M3"]
    return len(keys), len(fp8)


# ---------------------------------------------------------------------------
# VAE: LTX-2.3 用4段デコーダ変換(diffusers本体の LTX-2.0用3段マッピングを補完)
# ---------------------------------------------------------------------------

# diffusers の convert_ltx2_vae_to_diffusers() は LTX_2_0_VIDEO_VAE_RENAME_DICT の
# up_blocks.* マッピングが 7 エントリ(3段デコーダ、LTX-2.0形状)しか無く、
# LTX-2.3 の AutoencoderKLLTX2Video(decoder_block_out_channels 長=4、実際に4段)には
# 2エントリ不足している(up_blocks.7/8 が変換されないまま残り、期待される
# up_blocks.3.upsamplers.0 / up_blocks.3 が missing になる)。チェックポイント側の
# flat な up_blocks.0..8 の意味(res_blocks/conv が交互、conv=upsampler)を実測して
# 確認済み: 0=mid res, 1=upsample0, 2=stage0 res, 3=upsample1, 4=stage1 res,
# 5=upsample2, 6=stage2 res, 7=upsample3, 8=stage3 res(最終段はupsamplerなし)。
# down_blocks側(encoder)は既存の9エントリ(0-8)のままで4段構造と正しく一致している
# (エンコーダは元から4段対応だった)。ここではデコーダ側だけ2エントリ追加して補完する。
_LTX23_VAE_EXTRA_DECODER_ENTRIES = {
    "up_blocks.7": "up_blocks.3.upsamplers.0",
    "up_blocks.8": "up_blocks.3",
}


def convert_ltx2_vae_to_diffusers_23(checkpoint):
    """diffusers.loaders.single_file_utils.convert_ltx2_vae_to_diffusers のロジックを
    そのまま複製し、LTX-2.3 用に up_blocks.7/8 の欠落エントリだけ補ったローカル版。
    venv の diffusers 本体は変更しない(プロジェクト側コードでのパッチ)。
    """
    LTX_2_0_VIDEO_VAE_RENAME_DICT = {
        "vae.": "",
        "down_blocks.0": "down_blocks.0",
        "down_blocks.1": "down_blocks.0.downsamplers.0",
        "down_blocks.2": "down_blocks.1",
        "down_blocks.3": "down_blocks.1.downsamplers.0",
        "down_blocks.4": "down_blocks.2",
        "down_blocks.5": "down_blocks.2.downsamplers.0",
        "down_blocks.6": "down_blocks.3",
        "down_blocks.7": "down_blocks.3.downsamplers.0",
        "down_blocks.8": "mid_block",
        "up_blocks.0": "mid_block",
        "up_blocks.1": "up_blocks.0.upsamplers.0",
        "up_blocks.2": "up_blocks.0",
        "up_blocks.3": "up_blocks.1.upsamplers.0",
        "up_blocks.4": "up_blocks.1",
        "up_blocks.5": "up_blocks.2.upsamplers.0",
        "up_blocks.6": "up_blocks.2",
        **_LTX23_VAE_EXTRA_DECODER_ENTRIES,
        "res_blocks": "resnets",
        "per_channel_statistics.mean-of-means": "latents_mean",
        "per_channel_statistics.std-of-means": "latents_std",
    }
    LTX_2_0_VAE_SPECIAL_KEYS_REMAP = {
        "per_channel_statistics.channel": lambda key, sd: sd.pop(key),
        "per_channel_statistics.mean-of-stds": lambda key, sd: sd.pop(key),
    }
    converted = {k: checkpoint.pop(k) for k in list(checkpoint.keys())}
    for key in list(converted.keys()):
        new_key = key[:]
        # longest-prefix-first to avoid up_blocks.1 clobbering up_blocks.1.something incorrectly
        for replace_key, rename_key in sorted(
            LTX_2_0_VIDEO_VAE_RENAME_DICT.items(), key=lambda kv: -len(kv[0])
        ):
            new_key = new_key.replace(replace_key, rename_key)
        converted[new_key] = converted.pop(key)
    for key in list(converted.keys()):
        for special_key, handler in LTX_2_0_VAE_SPECIAL_KEYS_REMAP.items():
            if special_key in key:
                handler(key, converted)
    return converted


def load_state_dict_streaming_to_device(model, state_dict: dict, device: str, dtype=torch.bfloat16):
    """meta モデルへ、部分的な state_dict を1テンソルずつ set_module_tensor_to_device() で
    流し込む(model.load_state_dict(..., assign=True) は meta のまま残った未充足パラメータを
    to() で実体化できずに落ちるため、常にこちらを使う)。missing/unexpected を返す。
    """
    from accelerate.utils import set_module_tensor_to_device

    expected = set(model.state_dict().keys())
    got = set(state_dict.keys())
    missing = sorted(expected - got)
    unexpected = sorted(got - expected)
    for key in expected & got:
        tensor = state_dict[key]
        set_module_tensor_to_device(model, key, device, value=tensor, dtype=dtype, clear_cache=False)
    return missing, unexpected


# ---------------------------------------------------------------------------
# transformer + connectors(同一checkpointプレフィックスから振り分け)
# ---------------------------------------------------------------------------

def build_transformer_state_dict(ckpt_path: str):
    """checkpointのmodel.diffusion_model.*から、
      - connector以外 -> transformer用 state_dict (convert_ltx2_transformer_to_diffusersへ)
      - video/audio_embeddings_connector.* -> connectors用の video_connector/audio_connector state_dict
    に振り分けて返す。fp8+weight_scaleはdequant済み。
    """
    n_total, n_fp8 = count_fp8_in_prefix(ckpt_path, "model.diffusion_model.")
    log(f"model.diffusion_model.* : {n_total} tensors, {n_fp8} fp8_e4m3fn (scaled)")

    transformer_raw = {}
    connector_raw = {}  # video_embeddings_connector./audio_embeddings_connector. プレフィックス付きのまま
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if k.startswith("model.diffusion_model.")]
        scale_keys = set(k for k in keys if k.endswith(".weight_scale"))
        # input_scale: per-tensor *activation* quantization scale for fp8 runtime inference.
        # Not needed when dequantizing weights to bf16 and computing in bf16 (no activation
        # quantization happens in that path), so these are dropped entirely.
        input_scale_keys = set(k for k in keys if k.endswith(".input_scale"))
        for k in keys:
            if k in scale_keys or k in input_scale_keys:
                continue
            short = k[len("model.diffusion_model."):]
            tensor = f.get_tensor(k)
            base = k.rsplit(".weight", 1)[0] + ".weight_scale" if k.endswith(".weight") else None
            if base is not None and base in scale_keys:
                scale = f.get_tensor(base)
                tensor = (tensor.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)
            else:
                tensor = tensor.to(torch.bfloat16)

            if short.startswith("video_embeddings_connector.") or short.startswith("audio_embeddings_connector."):
                connector_raw[short] = tensor
            else:
                transformer_raw[short] = tensor

    log(f"split: transformer={len(transformer_raw)} keys, connector={len(connector_raw)} keys")
    return transformer_raw, connector_raw


# diffusers の convert_ltx2_transformer_to_diffusers() の
# LTX_2_0_TRANSFORMER_SPECIAL_KEYS_REMAP["adaln_single"] ハンドラは
# "adaln_single."/"audio_adaln_single." の完全一致プレフィックスしか処理しない
# (2.0時代のキー)。LTX-2.3で新設された prompt_adaln_single./audio_prompt_adaln_single.
# (cross_attn_mod=True 用の追加モジュール、checkpoint実測で12キー確認済み)は
# 素通りしてしまい、"prompt_adaln_single.*" のまま残る(モデル側は
# "prompt_adaln.*"/"audio_prompt_adaln.*" を期待)。ここでは変換後の state_dict に
# 対してこの2種類だけ追加でリネームする(venv本体は変更しない)。
def _fix_ltx23_prompt_adaln_keys(converted: dict) -> dict:
    for key in list(converted.keys()):
        if key.startswith("audio_prompt_adaln_single."):
            new_key = key.replace("audio_prompt_adaln_single.", "audio_prompt_adaln.")
            converted[new_key] = converted.pop(key)
        elif key.startswith("prompt_adaln_single."):
            new_key = key.replace("prompt_adaln_single.", "prompt_adaln.")
            converted[new_key] = converted.pop(key)
    return converted


def build_connectors_state_dict(connector_raw: dict):
    """connector_raw (video_embeddings_connector.*/audio_embeddings_connector.*, prefix stripped
    of model.diffusion_model.) -> LTX2TextConnectors の video_connector/audio_connector
    サブモジュール state_dict へリネーム変換する(text_proj_in系は呼び出し側で別途追加)。
    """
    out = {}
    for modality in ("video", "audio"):
        src_prefix = f"{modality}_embeddings_connector."
        dst_prefix = f"{modality}_connector."
        for k, v in connector_raw.items():
            if not k.startswith(src_prefix):
                continue
            rest = k[len(src_prefix):]
            if rest == "learnable_registers":
                out[dst_prefix + "learnable_registers"] = v
                continue
            # transformer_1d_blocks.N.attn1.q_norm.weight -> transformer_blocks.N.attn1.norm_q.weight
            rest2 = rest.replace("transformer_1d_blocks.", "transformer_blocks.")
            rest2 = rest2.replace(".q_norm.", ".norm_q.")
            rest2 = rest2.replace(".k_norm.", ".norm_k.")
            out[dst_prefix + rest2] = v

    return out


# ---------------------------------------------------------------------------
# vocoder(diffusers に single_file 変換関数が一切無いため独自実装)
# ---------------------------------------------------------------------------
_VOCODER_RENAME_PAIRS = [
    ("conv_pre", "conv_in"),
    ("conv_post", "conv_out"),
    ("act_post", "act_out"),
    ("resblocks", "resnets"),
    ("downsample.lowpass.filter", "downsample.filter"),
    ("ups.", "upsamplers."),
]


def convert_ltx2_vocoder_to_diffusers(raw: dict) -> dict:
    converted = {}
    for k, v in raw.items():
        short = k[len("vocoder."):] if k.startswith("vocoder.") else k
        for old, new in _VOCODER_RENAME_PAIRS:
            short = short.replace(old, new)
        converted[short] = v
    return converted


# ---------------------------------------------------------------------------
# Gemma 3 12B(ローカル bf16 ファイル + HF config)
# ---------------------------------------------------------------------------

# gemma_3_12B_it.safetensors (ComfyUI text_encoders 配布) は transformers の現行
# Gemma3ForConditionalGeneration が期待する state_dict とキー命名が異なる
# (実測、1066テンソルずつで完全に1:1対応することを確認済み):
#   model.embed_tokens.* / model.layers.* / model.norm.*  -> model.language_model.<同名>
#   vision_model.*                                         -> model.vision_tower.vision_model.*
#   multi_modal_projector.*                                -> model.multi_modal_projector.*
#   spiece_model (1個、sentencepieceトークナイザのblob、モデル重みではない)  -> 破棄
# lm_head.weight は checkpoint に存在しない(tie_word_embeddings により
# model.language_model.embed_tokens.weight と共有される transformers 側の実装に依存。
# LTX2Pipeline は output_hidden_states=True で hidden_states しか読まず lm_head を
# 一切使わないため、tie されていれば実害はない)。
_GEMMA_RENAME_PREFIXES = [
    ("model.embed_tokens.", "model.language_model.embed_tokens."),
    ("model.layers.", "model.language_model.layers."),
    ("model.norm.", "model.language_model.norm."),
    ("vision_model.", "model.vision_tower.vision_model."),
    ("multi_modal_projector.", "model.multi_modal_projector."),
]


def convert_gemma_checkpoint_to_transformers(raw: dict) -> dict:
    converted = {}
    for k, v in raw.items():
        if k == "spiece_model":
            continue
        new_key = k
        for old, new in _GEMMA_RENAME_PREFIXES:
            if k.startswith(old):
                new_key = new + k[len(old):]
                break
        converted[new_key] = v
    return converted


# ---------------------------------------------------------------------------
# 各コンポーネントの高レベルロード関数(state_dict組み立て + streamingロードをまとめる)
# ---------------------------------------------------------------------------

def load_vae_pair(ckpt_path: str, device: str, config_repo: str):
    """vae(AutoencoderKLLTX2Video) + audio_vae(AutoencoderKLLTX2Audio) をロードする。"""
    from diffusers import AutoencoderKLLTX2Video, AutoencoderKLLTX2Audio
    from diffusers.loaders.single_file_utils import convert_ltx2_audio_vae_to_diffusers
    from accelerate import init_empty_weights

    t0 = time.time()
    vae_cfg = AutoencoderKLLTX2Video.load_config(config_repo, subfolder="vae")
    with init_empty_weights():
        vae = AutoencoderKLLTX2Video.from_config(vae_cfg)

    raw = stream_dequant_state_dict(ckpt_path, "vae.")
    log(f"read {len(raw)} raw vae.* tensors in {time.time()-t0:.1f}s, RAM avail={ram_available_gb():.1f}GB")
    converted = convert_ltx2_vae_to_diffusers_23(raw)
    missing, unexpected = load_state_dict_streaming_to_device(vae, converted, device)
    if missing:
        raise RuntimeError(f"vae has {len(missing)} unfilled params: {missing[:10]}")
    log(f"vae loaded to {device}")
    del raw, converted
    gc.collect()

    t0 = time.time()
    avae_cfg = AutoencoderKLLTX2Audio.load_config(config_repo, subfolder="audio_vae")
    with init_empty_weights():
        audio_vae = AutoencoderKLLTX2Audio.from_config(avae_cfg)
    raw = stream_dequant_state_dict(ckpt_path, "audio_vae.")
    log(f"read {len(raw)} raw audio_vae.* tensors in {time.time()-t0:.1f}s")
    converted = convert_ltx2_audio_vae_to_diffusers(raw)
    missing, unexpected = load_state_dict_streaming_to_device(audio_vae, converted, device)
    if missing:
        raise RuntimeError(f"audio_vae has {len(missing)} unfilled params: {missing[:10]}")
    log(f"audio_vae loaded to {device}")
    del raw, converted
    gc.collect()

    vae.eval()
    audio_vae.eval()
    return vae, audio_vae


def load_transformer_and_connectors(ckpt_path: str, device: str, config_repo: str):
    """transformer(LTX2VideoTransformer3DModel) + connectors(LTX2TextConnectors) を
    同一チェックポイント読み取りパスから組み立てる(28GBファイルの再ストリームを避ける)。
    """
    from diffusers import LTX2VideoTransformer3DModel
    from diffusers.loaders.single_file_utils import convert_ltx2_transformer_to_diffusers
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
    from accelerate import init_empty_weights

    t_cfg = LTX2VideoTransformer3DModel.load_config(config_repo, subfolder="transformer")
    with init_empty_weights():
        transformer = LTX2VideoTransformer3DModel.from_config(t_cfg)

    t0 = time.time()
    transformer_raw, connector_raw = build_transformer_state_dict(ckpt_path)
    log(f"streamed transformer+connector in {time.time()-t0:.1f}s, RAM avail={ram_available_gb():.1f}GB")

    reprefixed = {"model.diffusion_model." + k: v for k, v in transformer_raw.items()}
    converted_t = convert_ltx2_transformer_to_diffusers(reprefixed)
    converted_t = _fix_ltx23_prompt_adaln_keys(converted_t)
    del transformer_raw, reprefixed
    missing, unexpected = load_state_dict_streaming_to_device(transformer, converted_t, device)
    if missing:
        raise RuntimeError(f"transformer missing={len(missing)}: {missing[:10]}")
    del converted_t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"transformer loaded to {device}, RAM avail={ram_available_gb():.1f}GB")

    c_cfg = LTX2TextConnectors.load_config(config_repo, subfolder="connectors")
    with init_empty_weights():
        connectors = LTX2TextConnectors.from_config(c_cfg)
    converted_c = build_connectors_state_dict(connector_raw)
    proj_raw = stream_dequant_state_dict(ckpt_path, "text_embedding_projection.")
    for k, v in proj_raw.items():
        short = k[len("text_embedding_projection."):]
        short = short.replace("video_aggregate_embed", "video_text_proj_in")
        short = short.replace("audio_aggregate_embed", "audio_text_proj_in")
        converted_c[short] = v
    del connector_raw, proj_raw
    missing, unexpected = load_state_dict_streaming_to_device(connectors, converted_c, device)
    if missing:
        raise RuntimeError(f"connectors missing={len(missing)}: {missing[:10]}")
    del converted_c
    gc.collect()
    log(f"connectors loaded to {device}")

    transformer.eval()
    connectors.eval()
    return transformer, connectors


def load_vocoder(ckpt_path: str, device: str, config_repo: str):
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE, UpSample1d
    from accelerate import init_empty_weights

    cfg = LTX2VocoderWithBWE.load_config(config_repo, subfolder="vocoder")
    with init_empty_weights():
        vocoder = LTX2VocoderWithBWE.from_config(cfg)

    raw = stream_dequant_state_dict(ckpt_path, "vocoder.")
    log(f"read {len(raw)} raw vocoder.* tensors")
    converted = convert_ltx2_vocoder_to_diffusers(raw)

    missing, unexpected = load_state_dict_streaming_to_device(vocoder, converted, device)
    if missing:
        raise RuntimeError(f"vocoder has {len(missing)} unfilled params: {missing[:10]}")

    # LTX2VocoderWithBWE.resampler = UpSample1d(..., persistent=False): its "filter" buffer is
    # registered non-persistent, so it's absent from state_dict() and was never materialized off
    # of the meta device by the streaming load above (confirmed via a standalone repro: it
    # silently stays on meta, only surfacing as "Cannot copy out of meta tensor" on the *next*
    # .to() call). Unlike learned weights this buffer is deterministically derived from
    # (ratio, window_type) at __init__ time, not loaded from the checkpoint, so the fix is to
    # simply reconstruct that one small submodule off-meta and swap it in.
    output_sr = cfg.get("output_sampling_rate", 48000)
    input_sr = cfg.get("input_sampling_rate", 16000)
    vocoder.resampler = UpSample1d(ratio=output_sr // input_sr, window_type="hann", persistent=False).to(device)
    log(f"rebuilt non-persistent resampler.filter buffer off-meta (ratio={output_sr // input_sr})")

    vocoder.eval()
    log(f"vocoder loaded to {device}")
    return vocoder


# ---------------------------------------------------------------------------
# latent upsampler(低解像度生成 -> 潜在空間アップスケール -> 高解像度デコードの2段生成、
# 2026-07-20追加)
# ---------------------------------------------------------------------------

# ローカルの ComfyUI 配布ファイル(ltx-2.3-spatial-upscaler-x2-1.1.safetensors、
# ~995MB、bf16、fp8/weight_scaleペアなし)は diffusers の LTX2LatentUpsamplerModel の
# state_dict とキー名・shapeとも完全一致することを実機確認済み(72キー全一致)。
# HF Hub の Lightricks/LTX-2 リポジトリにある diffusers形式ファイル
# (latent_upsampler/diffusion_pytorch_model.safetensors)はモデル構造そのものが異なる
# (use_rational_resampler=True の SpatialRationalResampler、upsampler.conv/
# upsampler.blur_down/upsampler.pixel_shuffle という別サブモジュール構成)ため、
# ローカルファイルとキー互換ではない(単純なリネームでは変換できない、別モデル)。
# ローカルファイルは use_rational_resampler=False(素の Conv+PixelShuffleND)経路の
# 重みであるため、from_config 時にその設定を明示する。config自体は Lightricks/LTX-2 の
# latent_upsampler/config.json(dims/in_channels/mid_channels/num_blocks_per_stage/
# spatial_upsample/temporal_upsample)をベースに use_rational_resampler だけ上書きする。
LTX2_UPSAMPLER_CONFIG_OVERRIDES = {
    "use_rational_resampler": False,
}


def load_upsampler(upsampler_path: str, device: str, config_repo_upsampler: str = "Lightricks/LTX-2"):
    """LTX2LatentUpsamplerModel をローカル ComfyUI 形式チェックポイントからロードする。

    RAM安全性: stream_dequant_state_dict() と同じ safe_open ベースの1テンソルずつの
    読み取りを使う(このファイルは995MBと小さいが、他コンポーネントと同じ流儀を踏襲する)。
    キー変換は不要(ローカルファイルは diffusers の state_dict と完全一致、モジュール
    docstring参照)。
    """
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
    from accelerate import init_empty_weights

    cfg = LTX2LatentUpsamplerModel.load_config(config_repo_upsampler, subfolder="latent_upsampler")
    cfg = dict(cfg)
    cfg.update(LTX2_UPSAMPLER_CONFIG_OVERRIDES)
    with init_empty_weights():
        upsampler = LTX2LatentUpsamplerModel.from_config(cfg)

    t0 = time.time()
    raw = stream_dequant_state_dict(upsampler_path, "")  # プレフィックス無し、全キー対象
    log(f"read {len(raw)} raw latent_upsampler tensors in {time.time()-t0:.1f}s")

    missing, unexpected = load_state_dict_streaming_to_device(upsampler, raw, device)
    if missing:
        raise RuntimeError(f"latent_upsampler has {len(missing)} unfilled params: {missing[:10]}")
    if unexpected:
        log(f"warning: {len(unexpected)} unexpected latent_upsampler keys ignored: {unexpected[:10]}")
    del raw
    gc.collect()

    upsampler.eval()
    log(f"latent_upsampler loaded to {device}")
    return upsampler


# text_encoder(Gemma3ForConditionalGeneration)の fp8 layerwise casting 用 skip パターン。
# diffusers の apply_layerwise_casting() の既定 skip パターン
# (DEFAULT_SKIP_MODULES_PATTERN = ('pos_embed', 'patch_embed', 'norm', '^proj_in$',
# '^proj_out$'))は diffusers 側 transformer の命名規則(patch_embed 等)向けであり、
# Gemma3 の命名(embed_tokens/lm_head/rotary_emb)には一致しない。実機確認済み:
#   - 'norm' は re.search の部分一致のため、input_layernorm/post_attention_layernorm/
#     post_feedforward_layernorm/pre_feedforward_layernorm/self_attn.q_norm/k_norm/
#     最終 model.language_model.norm は全て skip 対象になる(既定のままで安全)。
#   - 'patch_embed' は vision embeddings の patch_embedding に部分一致で skip される。
#   - 一方 embed_tokens(nn.Embedding、_GO_LC_SUPPORTED_PYTORCH_LAYERS に含まれるため
#     何も指定しないとfp8化されてしまう)、lm_head(nn.Linear)、rotary_emb(バッファのみの
#     モジュールだが念のため)は既定パターンに一致しないため、明示的に追加する。
# CLAUDE.md 37番の meta-init 修正(embed_scale/rotary_emb/position_ids の非persistent
# バッファ再構築)と合わせて、embedding系・norm系・lm_headをfp8化から保護する。
LTX2_TE_FP8_SKIP_MODULES_PATTERN = (
    "pos_embed", "patch_embed", "norm", "^proj_in$", "^proj_out$",  # diffusers既定を維持
    "embed_tokens", "embed_scale", "lm_head", "rotary_emb",  # Gemma3固有の追加保護
)


def load_text_encoder(gemma_path: str, device: str, config_repo: str, te_quant: str = "none", te_qat_dir: "str | None" = None):
    """Gemma 3 12B(ローカル bf16 ファイル)を Gemma3ForConditionalGeneration へロードする。

    te_quant(2026-07-20追加、CLAUDE.md 33番・37番・39番の「丸ごとCPUスワップ禁止」を
    厳守した上でのVRAM削減オプション):
      - "none"(既定): このdocstring以下の従来ロジックのまま(挙動無変更)。
      - "fp8": bf16でロードした後、apply_layerwise_casting() で
        storage_dtype=float8_e4m3fn/compute_dtype=bfloat16を適用する
        (VRAM ~23GB -> ~12GB見込み)。ロード元・ロードロジックは"none"と共通。
      - "nf4": te_qat_dir(QAT版、完全なHF形式ディレクトリ)を
        Gemma3ForConditionalGeneration.from_pretrained(..., quantization_config=
        BitsAndBytesConfig(nf4)) で直接GPUへロードする("none"/"fp8"とは別経路、
        ComfyUI形式チェックポイントのstreamingロードは行わない)。
    """
    if te_quant == "nf4":
        return _load_text_encoder_nf4(te_qat_dir, device)

    import json as _json
    from accelerate import init_empty_weights
    from huggingface_hub import hf_hub_download
    from transformers import Gemma3ForConditionalGeneration, Gemma3Config

    with safe_open(gemma_path, framework="pt") as f:
        keys = list(f.keys())
    log(f"gemma checkpoint: {len(keys)} tensors")

    cfg_path = hf_hub_download(config_repo, "text_encoder/config.json")
    with open(cfg_path) as f:
        cfg_dict = _json.load(f)
    cfg = Gemma3Config(**cfg_dict)

    t0 = time.time()
    with init_empty_weights():
        model = Gemma3ForConditionalGeneration(cfg)
    log(f"RAM avail after meta init: {ram_available_gb():.1f}GB")

    raw = {}
    with safe_open(gemma_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k == "spiece_model":
                continue
            raw[k] = f.get_tensor(k).to(torch.bfloat16)
    log(f"streamed {len(raw)} gemma tensors in {time.time()-t0:.1f}s, RAM avail={ram_available_gb():.1f}GB")

    converted = convert_gemma_checkpoint_to_transformers(raw)
    del raw
    gc.collect()

    missing, unexpected = load_state_dict_streaming_to_device(model, converted, device)
    # lm_head.weight is absent from the checkpoint (tied to embed_tokens per Gemma3's
    # tie_word_embeddings default). LTX2Pipeline never reads logits/lm_head (only
    # output_hidden_states=True), but tie explicitly so no meta tensor remains on the module.
    if missing == ["lm_head.weight"]:
        from accelerate.utils import set_module_tensor_to_device
        embed_weight = model.model.language_model.embed_tokens.weight.detach().clone()
        set_module_tensor_to_device(model, "lm_head.weight", device, value=embed_weight, dtype=torch.bfloat16)
        missing = []
        log("tied lm_head.weight to embed_tokens.weight (missing key resolved)")
    if missing:
        raise RuntimeError(f"text_encoder has {len(missing)} unfilled params: {missing[:10]}")

    del converted
    gc.collect()

    # transformers registers several deterministic (config-derived, not learned)
    # non-persistent buffers -- RoPE inv_freq caches (Gemma3RotaryEmbedding),
    # embed_scale (Gemma3TextScaledWordEmbedding), vision position_ids -- which are
    # absent from state_dict() and therefore never touched by the streaming load above.
    # Building the whole model under meta-init leaves these as meta tensors forever
    # (confirmed via named_buffers() diff against state_dict() keys), which crashes on the
    # *next* unrelated .to(device) call with "Cannot copy out of meta tensor" (same class of
    # bug as LTX2VocoderWithBWE.resampler.filter above). Fix: rebuild just these small
    # submodules for real (off meta) and swap them in -- they are cheap (no large weight
    # tensors, just small deterministic buffers) so this does not reintroduce the
    # large-model-on-CPU RAM risk.
    from transformers.models.gemma3.modeling_gemma3 import Gemma3RotaryEmbedding

    model.model.language_model.rotary_emb = Gemma3RotaryEmbedding(cfg.text_config).to(device)
    model.model.language_model.embed_tokens.register_buffer(
        "embed_scale", torch.tensor(cfg.text_config.hidden_size**0.5, device=device), persistent=False
    )
    if hasattr(model.model, "vision_tower"):
        num_positions = model.model.vision_tower.vision_model.embeddings.num_positions
        model.model.vision_tower.vision_model.embeddings.register_buffer(
            "position_ids", torch.arange(num_positions, device=device).unsqueeze(0), persistent=False
        )
    log("rebuilt non-persistent RoPE/embed_scale/position_ids buffers off-meta")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()
    log(f"text_encoder loaded to {device}, RAM avail={ram_available_gb():.1f}GB")

    if te_quant == "fp8":
        from diffusers.hooks import apply_layerwise_casting

        te_vram_before = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else None
        apply_layerwise_casting(
            model,
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16,
            skip_modules_pattern=LTX2_TE_FP8_SKIP_MODULES_PATTERN,
        )
        # CLAUDE.md 12番: layerwise casting は呼んだ瞬間に圧縮が実行される(遅延評価ではない)。
        # empty_cache() を呼ぶことで解放された bf16 分の VRAM が実際に返却される。
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            te_vram_after = torch.cuda.memory_allocated() / 1024**3
            log(
                f"text_encoder fp8 layerwise casting applied "
                f"(VRAM {te_vram_before:.2f}GB -> {te_vram_after:.2f}GB)"
            )
        else:
            log("text_encoder fp8 layerwise casting applied")

    return model, cfg


def _load_text_encoder_nf4(te_qat_dir: "str | None", device: str):
    """QAT版 Gemma 3 12B(完全なHF形式ディレクトリ)を BitsAndBytesConfig(nf4) で
    直接GPUへロードする。

    QAT版ディレクトリの state_dict キーは標準の transformers 形式のプレフィックス省略版
    ("language_model.*" / "vision_tower.*" / "multi_modal_projector.*"、トップレベルの
    "model." が無い)だが、Gemma3ForConditionalGeneration.from_pretrained() は
    transformers 標準のレガシーキー再マッピング機構でこれを正しく解決することを実機確認
    済み(1065/1066キーが "Materializing param=model.language_model.*" のログとともに
    正常にロードされ、エラーなく完了する)。lm_head.weight はチェックポイントに存在しない
    (tie_word_embeddings によりembed_tokensと共有、from_pretrained が自動的に処理する)。

    tokenizer は現行のもの(LTX2Pipeline既存の GemmaTokenizerFast)を使い続ける
    (QATディレクトリのtokenizerには切り替えない、タスク指示どおり)。

    ホストRAMへの丸ごと退避・常駐は行わない(device_map で最初からGPUへロードする、
    CLAUDE.md 33番・39番の禁止パターンを踏襲)。
    """
    from transformers import BitsAndBytesConfig, Gemma3ForConditionalGeneration

    if not te_qat_dir or not os.path.isdir(te_qat_dir):
        raise RuntimeError(
            f"[families.ltx2.convert] DS_LTX2_TE_QAT_DIR が見つかりません: {te_qat_dir!r} "
            "(QAT版 Gemma 3 12B の完全なHF形式ディレクトリを指定してください)"
        )

    log(f"loading QAT Gemma 3 12B (nf4) from {te_qat_dir}")
    t0 = time.time()
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Gemma3ForConditionalGeneration.from_pretrained(
        te_qat_dir,
        quantization_config=bnb_cfg,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        vram = torch.cuda.memory_allocated() / 1024**3
        log(f"QAT Gemma 3 12B (nf4) loaded in {time.time()-t0:.1f}s, VRAM={vram:.2f}GB")
    else:
        log(f"QAT Gemma 3 12B (nf4) loaded in {time.time()-t0:.1f}s")
    return model, model.config


def load_iclora_state_dict(iclora_path: str) -> dict:
    """IC-LoRA(MergeGreen 等)の ComfyUI 形式 safetensors を読み、diffusers/PEFT 形式の
    state_dict(`transformer.` プレフィックス付き)へ変換して返す。

    ファイルは通常サイズの bf16 LoRA(624MB程度)であり、scaled-fp8 デクオンタイズは
    不要(stream_dequant_state_dict() は weight_scale ペアが無ければ何もしないが、
    このLoRAには元々そのペアが存在しない通常のLoRAのため、素の safe_open 読み取りで足りる)。

    キー変換: diffusers 本体の `LTX2LoraLoaderMixin.lora_state_dict()` が内部で使う
    `_convert_non_diffusers_ltx2_lora_to_diffusers()` をそのまま再利用する(pipeline
    経由の `pipe.load_lora_weights()` と同じ変換ロジック)。MergeGreen のキーは全て
    `diffusion_model.transformer_blocks.N.attn1/attn2/ff.*.lora_A/lora_B.weight`
    (960キー、connector向け `text_embedding_projection.*` キーは無し)であり、
    リネーム対象パターン(patchify_proj 等)には一致しないため、実質的に
    `diffusion_model.` プレフィックスを `transformer.` に置き換えるだけの変換になる
    (実機のキー確認で照合済み)。

    transformer-level の `transformer.load_lora_adapter(sd, prefix="transformer")` は
    `prefix.` を自分で剥がす実装のため、本関数が返す `transformer.` プレフィックス付き
    state_dict をそのまま渡せる。
    """
    from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_ltx2_lora_to_diffusers

    raw_sd = {}
    with safe_open(iclora_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            raw_sd[key] = f.get_tensor(key)
    log(f"read {len(raw_sd)} raw IC-LoRA tensors from {iclora_path}")

    converted = _convert_non_diffusers_ltx2_lora_to_diffusers(raw_sd)
    log(f"converted IC-LoRA state_dict: {len(converted)} keys (prefix='transformer.')")
    return converted
