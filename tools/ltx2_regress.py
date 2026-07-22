# -*- coding: utf-8 -*-
"""
LTX-2.3(動画+音声生成)統合可否のスタンドアロン回帰確認スクリプト。

目的: ローカルの ComfyUI形式 scaled-fp8 チェックポイント
(/home/animede/ComfyUI/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors、28GB)を
venv の diffusers(0.40.0.dev0)の LTX2Pipeline で読み込み、蒸留8ステップの短い動画を
1本生成できるか確認する。families/ や app.py には一切触れない(調査専用の使い捨てスクリプト)。

このチェックポイントは「ComfyUI/Lightricks 純正のモノリシック AVTransformer3DModel 形式」で
あり、diffusers の from_single_file 変換関数(convert_ltx2_transformer_to_diffusers)が
前提とする「connector は最初から独立した state_dict」という形とは異なることが調査で判明した:

  - transformer 本体(model.diffusion_model.*, 7368キー)に
    video_embeddings_connector.* / audio_embeddings_connector.* (各129キー、8層)が
    埋め込まれている。diffusers 側の LTX2TextConnectors は本来これと同じ構造(8層、
    per_modality_projections=True)だが、独立した pipeline component
    (connectors.video_connector / connectors.audio_connector)として設計されている。
    convert_ltx2_transformer_to_diffusers() はこれらのキーを単純に remove_keys_inplace()
    で「捨てる」実装になっており、別コンポーネントへの再配線は行わない
    (diffusers 側は元々別ファイルから来る前提のため)。
  - text_embedding_projection.{video,audio}_aggregate_embed.{weight,bias}
    (4キー)は connectors/config.json の text_proj_in (per_modality_projections=True の
    video_text_proj_in / audio_text_proj_in) に対応する「入力側の集約線形層」。
    lora_conversion_utils.py の rename_dict = {"aggregate_embed": "text_proj_in"} が
    この対応関係の裏付け。
  - vocoder.* (1227キー) は LTX2VocoderWithBWE 完全体(vocoder/bwe_generator/mel_stft)。
    single_file_utils には vocoder 用の checkpoint_mapping_fn が一切存在しない
    (grep で確認済み、SINGLE_FILE_LOADABLE_CLASSES にも LTX2Vocoder 系は無い)。
  - fp8化: F8_E4M3 1462テンソル、対になる *.weight_scale (F32、shape=(), per-tensor
    スカラー)が1462個。全て model.diffusion_model.* (transformer)配下のみ。
    dequant は tensor.to(bf16) * scale(スカラーなのでbroadcastでよい)。

本スクリプトは上記を踏まえ、diffusers の convert_ltx2_*_to_diffusers() をベースに、
「connector キーを捨てずに専用の LTX2TextConnectors state_dict へ振り分ける」
「fp8 + weight_scale ペアを見つけたら dequant する」ロジックを追加した独自の
ストリーミングローダーを実装する。

RAM安全性: 28GBのチェックポイントを丸ごとRAMに読まない。safetensors.safe_open は
mmapベースで get_tensor() 呼び出し時点で該当テンソル分だけ読む実装のため、
1テンソルずつ処理してすぐ対象モデルへ .to(device) することで、常時RAMに保持するのは
「現在処理中の1テンソル」+「構築中の空(meta)モデルの実体化済み部分」のみに抑える
(core/loaders.py の load_safetensors_streaming と同じ方針)。

使い方:
  venv/bin/python tools/ltx2_regress.py --stage configs      # config突き合わせのみ(GPU不要)
  venv/bin/python tools/ltx2_regress.py --stage vae          # vae/audio_vae 単体ロード確認
  venv/bin/python tools/ltx2_regress.py --stage transformer  # transformer単体ロード確認(dequant込み)
  venv/bin/python tools/ltx2_regress.py --stage connectors   # connectors単体ロード確認
  venv/bin/python tools/ltx2_regress.py --stage vocoder      # vocoder単体ロード確認
  venv/bin/python tools/ltx2_regress.py --stage full         # 全部組み立てて8step動画生成
"""
import argparse
import gc
import json
import os
import sys
import time

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CKPT_PATH = "/home/animede/ComfyUI/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"
GEMMA_PATH = "/home/animede/ComfyUI/models/text_encoders/gemma_3_12B_it.safetensors"
CFG_DIR = "/tmp/claude-1000/-home-animede-diffusers-server/004ec3eb-8a16-41bb-90bc-6905fddac044/scratchpad/ltx23_cfg"
OUT_DIR = "/tmp/claude-1000/-home-animede-diffusers-server/004ec3eb-8a16-41bb-90bc-6905fddac044/scratchpad/ltx2_test"
os.makedirs(OUT_DIR, exist_ok=True)


def log(msg):
    print(f"[ltx2_regress] {msg}", flush=True)


def vram_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024**3


def ram_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                return kb / 1024**2
    return -1.0


# ---------------------------------------------------------------------------
# 1テンソルずつ読む共通ヘルパ(dequant対応版 load_safetensors_streaming)
# ---------------------------------------------------------------------------

def stream_dequant_state_dict(path: str, prefix: str, strip_prefix: str = ""):
    """safetensors から prefix に一致するキーだけを1テンソルずつ読み、
    fp8+weight_scale ペアがあれば dequant(bf16 * scale)、それ以外はそのまま bf16 キャストして
    dict[str, torch.Tensor] を返す(cpu上に構築、呼び出し側が個別に .to(device)する)。

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
            scale_key = k + "_scale" if False else None
            # weight_scale key convention: "<weight_key>_scale" replacing trailing ".weight"->".weight_scale"
            if k.endswith(".weight"):
                candidate_scale = k + "_scale"  # not used; real key is k.rsplit(".weight",1)[0] + ".weight_scale"
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
# Stage: configs (静的整合性チェック、GPU不要)
# ---------------------------------------------------------------------------

def stage_configs():
    log("=== stage: configs ===")
    from diffusers import LTX2VideoTransformer3DModel, AutoencoderKLLTX2Video, AutoencoderKLLTX2Audio
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE

    for name, cls, cfgfile in [
        ("transformer", LTX2VideoTransformer3DModel, "transformer/config.json"),
        ("vae", AutoencoderKLLTX2Video, "vae/config.json"),
        ("audio_vae", AutoencoderKLLTX2Audio, "audio_vae/config.json"),
        ("connectors", LTX2TextConnectors, "connectors/config.json"),
        ("vocoder", LTX2VocoderWithBWE, "vocoder/config.json"),
    ]:
        with open(os.path.join(CFG_DIR, cfgfile)) as f:
            cfg = json.load(f)
        cfg.pop("_class_name", None)
        cfg.pop("_diffusers_version", None)
        try:
            with torch.device("meta"):
                model = cls(**cfg)
            n_params = sum(p.numel() for p in model.parameters())
            log(f"OK  {name}: instantiated on meta, {n_params/1e9:.2f}B params, class={cls.__name__}")
        except Exception as e:
            log(f"FAIL {name}: {type(e).__name__}: {e}")
            raise
    log("all 5 component classes accept the downloaded HF configs without error")


# ---------------------------------------------------------------------------
# Stage: vae / audio_vae
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
    venv の diffusers 本体は変更しない(このファイル内でのみ使う調査用パッチ)。
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
    # any remaining meta params (missing keys) must still be materialized or the model
    # cannot run; for a regression test we surface them as an error unless caller accepts partial.
    return missing, unexpected


def stage_vae():
    log("=== stage: vae/audio_vae ===")
    from diffusers import AutoencoderKLLTX2Video, AutoencoderKLLTX2Audio
    from diffusers.loaders.single_file_utils import convert_ltx2_audio_vae_to_diffusers

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    with open(os.path.join(CFG_DIR, "vae/config.json")) as f:
        vae_cfg = json.load(f)
    vae_cfg.pop("_class_name", None)
    vae_cfg.pop("_diffusers_version", None)
    with torch.device("meta"):
        vae = AutoencoderKLLTX2Video(**vae_cfg)

    raw = stream_dequant_state_dict(CKPT_PATH, "vae.")
    log(f"read {len(raw)} raw vae.* tensors in {time.time()-t0:.1f}s, RAM avail={ram_available_gb():.1f}GB")
    converted = convert_ltx2_vae_to_diffusers_23(raw)
    missing, unexpected = load_state_dict_streaming_to_device(vae, converted, device)
    log(f"vae load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        log(f"  missing sample: {missing[:10]}")
    if unexpected:
        log(f"  unexpected sample: {unexpected[:10]}")
    if missing:
        raise RuntimeError(f"vae has {len(missing)} unfilled params, cannot run")
    log(f"vae loaded to {device}, VRAM={vram_gb():.2f}GB")
    del raw, converted
    gc.collect()

    t0 = time.time()
    with open(os.path.join(CFG_DIR, "audio_vae/config.json")) as f:
        avae_cfg = json.load(f)
    avae_cfg.pop("_class_name", None)
    avae_cfg.pop("_diffusers_version", None)
    with torch.device("meta"):
        audio_vae = AutoencoderKLLTX2Audio(**avae_cfg)
    raw = stream_dequant_state_dict(CKPT_PATH, "audio_vae.")
    log(f"read {len(raw)} raw audio_vae.* tensors in {time.time()-t0:.1f}s")
    converted = convert_ltx2_audio_vae_to_diffusers(raw)
    missing, unexpected = load_state_dict_streaming_to_device(audio_vae, converted, device)
    log(f"audio_vae load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        log(f"  missing sample: {missing[:10]}")
    if unexpected:
        log(f"  unexpected sample: {unexpected[:10]}")
    if missing:
        raise RuntimeError(f"audio_vae has {len(missing)} unfilled params, cannot run")
    log(f"audio_vae loaded to {device}, VRAM={vram_gb():.2f}GB")
    return vae, audio_vae


# ---------------------------------------------------------------------------
# Stage: transformer (dequant + connector-stripping)
# ---------------------------------------------------------------------------

def build_transformer_state_dict():
    """checkpointのmodel.diffusion_model.*から、
      - connector以外 -> transformer用 state_dict (convert_ltx2_transformer_to_diffusersへ)
      - video/audio_embeddings_connector.* -> connectors用の video_connector/audio_connector state_dict
    に振り分けて返す。fp8+weight_scaleはdequant済み。
    """
    n_total, n_fp8 = count_fp8_in_prefix(CKPT_PATH, "model.diffusion_model.")
    log(f"model.diffusion_model.* : {n_total} tensors, {n_fp8} fp8_e4m3fn (scaled)")

    transformer_raw = {}
    connector_raw = {}  # will hold keys still prefixed with video_embeddings_connector./audio_embeddings_connector.
    with safe_open(CKPT_PATH, framework="pt", device="cpu") as f:
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


def stage_transformer():
    log("=== stage: transformer ===")
    from diffusers import LTX2VideoTransformer3DModel
    from diffusers.loaders.single_file_utils import convert_ltx2_transformer_to_diffusers

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(CFG_DIR, "transformer/config.json")) as f:
        cfg = json.load(f)
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    with torch.device("meta"):
        transformer = LTX2VideoTransformer3DModel(**cfg)

    t0 = time.time()
    transformer_raw, connector_raw = build_transformer_state_dict()
    log(f"streamed+dequantized in {time.time()-t0:.1f}s, RAM avail={ram_available_gb():.1f}GB")

    # convert_ltx2_transformer_to_diffusers expects "model.diffusion_model." prefix present
    # (it strips it via the rename dict), and drops embeddings_connector keys via remove_keys_inplace.
    # We already stripped the prefix ourselves above, so re-add it just for this call, OR
    # replicate the same rename logic without the prefix. Simpler: re-add prefix.
    reprefixed = {"model.diffusion_model." + k: v for k, v in transformer_raw.items()}
    converted = convert_ltx2_transformer_to_diffusers(reprefixed)
    converted = _fix_ltx23_prompt_adaln_keys(converted)
    log(f"converted transformer state_dict: {len(converted)} keys (connector keys already excluded)")

    missing, unexpected = load_state_dict_streaming_to_device(transformer, converted, device)
    log(f"transformer load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        log(f"  missing sample ({len(missing)}): {missing[:20]}")
    if unexpected:
        log(f"  unexpected sample ({len(unexpected)}): {unexpected[:20]}")
    if missing:
        raise RuntimeError(f"transformer has {len(missing)} unfilled params, cannot run")

    del transformer_raw, converted, reprefixed
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log(f"transformer loaded to {device}, VRAM={vram_gb():.2f}GB")
    return transformer, connector_raw


# ---------------------------------------------------------------------------
# Stage: connectors (LTX2TextConnectors, built from transformer-embedded connector keys
# + text_embedding_projection.* aggregate_embed keys)
# ---------------------------------------------------------------------------

def build_connectors_state_dict(connector_raw: dict):
    """connector_raw (video_embeddings_connector.*/audio_embeddings_connector.*, prefix stripped
    of model.diffusion_model.) + text_embedding_projection.*aggregate_embed -> LTX2TextConnectors
    state_dict.

    LTX2TextConnectors attributes:
      - text_proj_in (per_modality_projections=False) OR
        video_text_proj_in / audio_text_proj_in (per_modality_projections=True, our case)
      - video_connector / audio_connector (LTX2ConnectorTransformer1d):
          learnable_registers
          rope (no params)
          transformer_blocks.N.norm1 (no params, RMSNorm elementwise_affine=False)
          transformer_blocks.N.attn1.{to_q,to_k,to_v,to_out.0,to_gate_logits}
          transformer_blocks.N.attn1.{q_norm->?, k_norm->?}  (LTX2Attention: norm_q/norm_k)
          transformer_blocks.N.norm2 (no params)
          transformer_blocks.N.ff.net.0.proj / net.2
          norm_out (no params)

    checkpoint side (video_embeddings_connector.*):
      learnable_registers
      transformer_1d_blocks.N.attn1.{to_q,to_k,to_v,to_out.0,to_gate_logits,q_norm,k_norm}
      transformer_1d_blocks.N.ff.net.0.proj / net.2
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


def stage_connectors(connector_raw=None):
    log("=== stage: connectors ===")
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(CFG_DIR, "connectors/config.json")) as f:
        cfg = json.load(f)
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    with torch.device("meta"):
        connectors = LTX2TextConnectors(**cfg)

    if connector_raw is None:
        _, connector_raw = build_transformer_state_dict()

    converted = build_connectors_state_dict(connector_raw)

    # text_embedding_projection.*aggregate_embed -> video_text_proj_in / audio_text_proj_in
    proj_raw = stream_dequant_state_dict(CKPT_PATH, "text_embedding_projection.")
    for k, v in proj_raw.items():
        # text_embedding_projection.video_aggregate_embed.weight -> video_text_proj_in.weight
        short = k[len("text_embedding_projection."):]
        short = short.replace("video_aggregate_embed", "video_text_proj_in")
        short = short.replace("audio_aggregate_embed", "audio_text_proj_in")
        converted[short] = v

    log(f"connectors converted state_dict: {len(converted)} keys")
    missing, unexpected = load_state_dict_streaming_to_device(connectors, converted, device)
    log(f"connectors load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        log(f"  missing sample ({len(missing)}): {missing[:20]}")
    if unexpected:
        log(f"  unexpected sample ({len(unexpected)}): {unexpected[:20]}")
    if missing:
        raise RuntimeError(f"connectors has {len(missing)} unfilled params, cannot run")

    log(f"connectors loaded to {device}, VRAM={vram_gb():.2f}GB")
    return connectors


# ---------------------------------------------------------------------------
# Stage: vocoder
# ---------------------------------------------------------------------------
#
# vocoder には diffusers 側の single_file 変換関数が一切存在しない(grep で確認済み)。
# checkpoint のネイティブ命名と diffusers の LTX2Vocoder/LTX2VocoderWithBWE 実装の命名が
# 以下の点で異なっており、単純なプレフィックス除去だけでは load_state_dict が通らない:
#   conv_pre    -> conv_in
#   conv_post   -> conv_out
#   act_post    -> act_out
#   resblocks   -> resnets
#   *.downsample.lowpass.filter -> *.downsample.filter (AntiAliasAct1d.downsampleは
#       DownSample1d を直接保持し、DownSample1d 自体が buffer 名 "filter" を持つ実装
#       (diffusers側)。checkpoint側は同じ構造に "lowpass" という中間名を挟んでいる。
# convs1/convs2/acts1/acts2/upsample.filter はそのまま一致(ResBlock/AntiAliasAct1d実装は
# diffusers側もネイティブ側もほぼ同一)。

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
        # strip "vocoder." prefix -> matches submodule names
        # (vocoder.vocoder.*, vocoder.bwe_generator.*, vocoder.mel_stft.*)
        short = k[len("vocoder."):] if k.startswith("vocoder.") else k
        for old, new in _VOCODER_RENAME_PAIRS:
            short = short.replace(old, new)
        converted[short] = v
    return converted


def stage_vocoder():
    log("=== stage: vocoder ===")
    from diffusers.pipelines.ltx2.vocoder import LTX2VocoderWithBWE

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with open(os.path.join(CFG_DIR, "vocoder/config.json")) as f:
        cfg = json.load(f)
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    with torch.device("meta"):
        vocoder = LTX2VocoderWithBWE(**cfg)

    raw = stream_dequant_state_dict(CKPT_PATH, "vocoder.")
    log(f"read {len(raw)} raw vocoder.* tensors")
    converted = convert_ltx2_vocoder_to_diffusers(raw)

    missing, unexpected = load_state_dict_streaming_to_device(vocoder, converted, device)
    log(f"vocoder load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        log(f"  missing sample ({len(missing)}): {missing[:20]}")
    if unexpected:
        log(f"  unexpected sample ({len(unexpected)}): {unexpected[:20]}")
    if missing:
        raise RuntimeError(f"vocoder has {len(missing)} unfilled params, cannot run")

    # LTX2VocoderWithBWE.resampler = UpSample1d(..., persistent=False) (vocoder.py line ~571):
    # its "filter" buffer is registered non-persistent, so it's absent from state_dict() and
    # was never materialized off of the meta device by the streaming load above (confirmed via
    # a standalone repro: .state_dict() does not contain "resampler.filter", so it silently
    # stayed on meta, only surfacing as "Cannot copy out of meta tensor" on the *next* .to()
    # call). Unlike learned weights this buffer is deterministically derived from
    # (ratio, window_type) at __init__ time, not loaded from the checkpoint, so the fix is to
    # simply reconstruct that one small submodule off-meta and swap it in.
    from diffusers.pipelines.ltx2.vocoder import UpSample1d
    output_sr = cfg.get("output_sampling_rate", 48000)
    input_sr = cfg.get("input_sampling_rate", 16000)
    vocoder.resampler = UpSample1d(ratio=output_sr // input_sr, window_type="hann", persistent=False).to(device)
    log(f"rebuilt non-persistent resampler.filter buffer off-meta (ratio={output_sr // input_sr})")

    log(f"vocoder loaded to {device}, VRAM={vram_gb():.2f}GB")
    return vocoder


# ---------------------------------------------------------------------------
# Stage: text encoder (Gemma 3 12B, local bf16 file + HF config)
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


def stage_text_encoder():
    log("=== stage: text_encoder (gemma) ===")
    import json as _json
    from transformers import Gemma3ForConditionalGeneration, Gemma3Config

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with safe_open(GEMMA_PATH, framework="pt") as f:
        keys = list(f.keys())
    log(f"gemma checkpoint: {len(keys)} tensors")

    from huggingface_hub import hf_hub_download
    cfg_path = hf_hub_download("diffusers/LTX-2.3-Diffusers", "text_encoder/config.json")
    with open(cfg_path) as f:
        cfg_dict = _json.load(f)
    cfg = Gemma3Config(**cfg_dict)

    t0 = time.time()
    with torch.device("meta"):
        model = Gemma3ForConditionalGeneration(cfg)
    log(f"RAM avail after meta init: {ram_available_gb():.1f}GB")

    raw = {}
    with safe_open(GEMMA_PATH, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k == "spiece_model":
                continue
            raw[k] = f.get_tensor(k).to(torch.bfloat16)
    log(f"streamed {len(raw)} gemma tensors in {time.time()-t0:.1f}s, RAM avail={ram_available_gb():.1f}GB")

    converted = convert_gemma_checkpoint_to_transformers(raw)
    del raw
    gc.collect()

    missing, unexpected = load_state_dict_streaming_to_device(model, converted, device)
    log(f"text_encoder load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        log(f"  missing sample ({len(missing)}): {missing[:20]}")
    if unexpected:
        log(f"  unexpected sample ({len(unexpected)}): {unexpected[:20]}")
    # lm_head.weight is absent from the checkpoint (tied to embed_tokens per Gemma3's
    # tie_word_embeddings default). LTX2Pipeline never reads logits/lm_head (only
    # output_hidden_states=True), but tie explicitly so no meta tensor remains on the module.
    if missing == ["lm_head.weight"]:
        from accelerate.utils import set_module_tensor_to_device
        embed_weight = model.model.language_model.embed_tokens.weight.detach().clone()
        set_module_tensor_to_device(model, "lm_head.weight", device, value=embed_weight, dtype=torch.bfloat16)
        missing = []
        log("tied lm_head.weight to embed_tokens.weight (missing key resolved)")

    del converted
    gc.collect()

    # transformers registers several deterministic (config-derived, not learned)
    # non-persistent buffers -- RoPE inv_freq caches (Gemma3RotaryEmbedding),
    # embed_scale (Gemma3TextScaledWordEmbedding), vision position_ids -- which are
    # absent from state_dict() and therefore never touched by the streaming load above.
    # Building the whole model under torch.device("meta") leaves these as meta tensors
    # forever (confirmed via named_buffers() diff against state_dict() keys), which
    # crashes on the *next* unrelated .to(device) call with "Cannot copy out of meta
    # tensor" (same class of bug hit and fixed for LTX2VocoderWithBWE.resampler.filter
    # above). Fix: rebuild just these small submodules for real (off meta) and swap
    # them in -- they are cheap (no large weight tensors, just small deterministic
    # buffers) so this does not reintroduce the large-model-on-CPU RAM risk.
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

    log(f"text_encoder loaded to {device}, VRAM={vram_gb():.2f}GB, RAM avail={ram_available_gb():.1f}GB")
    return model, cfg


# ---------------------------------------------------------------------------
# Stage: full pipeline assembly + generation
# ---------------------------------------------------------------------------

def stage_full():
    log("=== stage: full (assemble LTX2Pipeline + generate) ===")
    import json as _json
    from diffusers import LTX2Pipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.utils import export_to_video
    from transformers import GemmaTokenizerFast

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    log(f"RAM avail before any load: {ram_available_gb():.1f}GB")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        log(f"VRAM free before any load: {free/1024**3:.1f}GB / {total/1024**3:.1f}GB")

    # NOTE on VRAM strategy (2nd attempt): the shared diffusers-server (port 8601) was
    # restarted with no models loaded, so ~95GB VRAM is free. All 6 components summed
    # (~66GB bf16) + activation memory fits GPU-resident, which is both the fastest and
    # the simplest configuration ("収まるならGPU常駐が最速で正解", CLAUDE.md 33).
    # Each component is streamed 1-tensor-at-a-time directly to cuda:0 (RAM-safe,
    # proven in the per-stage runs).

    # 1. VAE + audio_vae (small, load first)
    vae, audio_vae = stage_vae()
    log(f"after vae+audio_vae: VRAM={vram_gb():.2f}GB, RAM avail={ram_available_gb():.1f}GB")

    # 2. transformer + connectors (share the same checkpoint read pass to avoid re-streaming
    # the 28GB file twice; build_transformer_state_dict() already splits transformer/connector)
    from diffusers import LTX2VideoTransformer3DModel
    from diffusers.loaders.single_file_utils import convert_ltx2_transformer_to_diffusers
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

    with open(os.path.join(CFG_DIR, "transformer/config.json")) as f:
        t_cfg = _json.load(f)
    t_cfg.pop("_class_name", None)
    t_cfg.pop("_diffusers_version", None)
    with torch.device("meta"):
        transformer = LTX2VideoTransformer3DModel(**t_cfg)

    t0 = time.time()
    transformer_raw, connector_raw = build_transformer_state_dict()
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
    log(f"transformer loaded to GPU: VRAM={vram_gb():.2f}GB, RAM avail={ram_available_gb():.1f}GB")

    with open(os.path.join(CFG_DIR, "connectors/config.json")) as f:
        c_cfg = _json.load(f)
    c_cfg.pop("_class_name", None)
    c_cfg.pop("_diffusers_version", None)
    with torch.device("meta"):
        connectors = LTX2TextConnectors(**c_cfg)
    converted_c = build_connectors_state_dict(connector_raw)
    proj_raw = stream_dequant_state_dict(CKPT_PATH, "text_embedding_projection.")
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
    log(f"connectors loaded: VRAM={vram_gb():.2f}GB, RAM avail={ram_available_gb():.1f}GB")

    # 3. vocoder
    vocoder = stage_vocoder()
    log(f"after vocoder: VRAM={vram_gb():.2f}GB, RAM avail={ram_available_gb():.1f}GB")

    # 4. text encoder + tokenizer
    text_encoder, gemma_cfg = stage_text_encoder()
    log(f"after text_encoder: VRAM={vram_gb():.2f}GB, RAM avail={ram_available_gb():.1f}GB")

    tok_dir_file = hf_hub_download_tokenizer()
    tokenizer = GemmaTokenizerFast.from_pretrained(tok_dir_file)

    # 5. scheduler
    with open(os.path.join(CFG_DIR, "scheduler/scheduler_config.json")) as f:
        sched_cfg = _json.load(f)
    sched_cfg.pop("_class_name", None)
    sched_cfg.pop("_diffusers_version", None)
    scheduler = FlowMatchEulerDiscreteScheduler(**sched_cfg)

    for m in (vae, audio_vae, transformer, connectors, vocoder, text_encoder):
        m.eval()

    pipe = LTX2Pipeline(
        scheduler=scheduler,
        vae=vae,
        audio_vae=audio_vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        connectors=connectors,
        transformer=transformer,
        vocoder=vocoder,
    )
    log(f"LTX2Pipeline assembled (all components GPU-resident). "
        f"VRAM={vram_gb():.2f}GB, RAM avail={ram_available_gb():.1f}GB")

    # Distilled model: use guidance_scale=1.0 (no CFG) matching Lightning/turbo-style distillation
    # convention elsewhere in this codebase (CLAUDE.md 35, Z-Image-Turbo uses guidance_scale=0.0;
    # LTX2 CFG semantics use 1.0 = disabled per encode_prompt's do_classifier_free_guidance flag).
    prompt = "A red panda walking on a tree branch, cinematic lighting, 4k"
    negative_prompt = "blurry, low quality, distorted"

    height, width = 320, 512  # multiples of 32, small for a smoke test
    num_frames = 25  # 8n+1 form, small
    num_inference_steps = 8

    log(f"generating: {width}x{height}, frames={num_frames}, steps={num_inference_steps}")
    log(f"VRAM free right before pipe() call: {torch.cuda.mem_get_info()[0]/1024**3:.1f}GB")
    t0 = time.time()
    generator = torch.Generator(device=device).manual_seed(12345)
    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=1.0,
            generator=generator,
        )
    elapsed = time.time() - t0
    log(f"generation done in {elapsed:.1f}s, VRAM peak={torch.cuda.max_memory_allocated()/1024**3:.2f}GB")

    frames = output.frames[0]
    video_path = os.path.join(OUT_DIR, "ltx23_smoketest.mp4")
    export_to_video(frames, video_path, fps=24)
    log(f"video saved: {video_path} ({len(frames)} frames)")

    # save first/middle/last frame as PNG for visual inspection
    for idx, name in [(0, "first"), (len(frames) // 2, "middle"), (len(frames) - 1, "last")]:
        frames[idx].save(os.path.join(OUT_DIR, f"ltx23_smoketest_{name}.png"))
    log(f"frame PNGs saved to {OUT_DIR}")

    if hasattr(output, "audio") and output.audio is not None:
        audio = output.audio
        if isinstance(audio, (list, tuple)):
            audio = audio[0]
        log(f"audio output present: type={type(audio)}, shape={getattr(audio, 'shape', None)}, "
            f"dtype={getattr(audio, 'dtype', None)}")
        try:
            import soundfile as sf
            wav = audio.detach().float().cpu()
            if wav.ndim == 3:
                wav = wav[0]
            # (channels, samples) -> (samples, channels)
            if wav.ndim == 2 and wav.shape[0] <= 8:
                wav = wav.transpose(0, 1)
            sample_rate = 48000  # LTX2VocoderWithBWE output_sampling_rate per vocoder config
            audio_path = os.path.join(OUT_DIR, "ltx23_smoketest.wav")
            sf.write(audio_path, wav.numpy(), sample_rate)
            log(f"audio saved: {audio_path} ({wav.shape[0]/sample_rate:.2f}s, {wav.shape})")
            # mux into mp4 alongside for a complete artifact
            muxed = os.path.join(OUT_DIR, "ltx23_smoketest_av.mp4")
            os.system(
                f"ffmpeg -y -loglevel error -i {video_path} -i {audio_path} "
                f"-c:v copy -c:a aac -shortest {muxed}"
            )
            if os.path.exists(muxed):
                log(f"muxed audio+video: {muxed}")
        except Exception as e:
            log(f"audio save failed (non-fatal): {type(e).__name__}: {e}")
    else:
        log("no audio output field on result (or None)")

    return video_path


def hf_hub_download_tokenizer():
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(
        "diffusers/LTX-2.3-Diffusers",
        allow_patterns=["tokenizer/*"],
    )
    return os.path.join(local_dir, "tokenizer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=[
        "configs", "vae", "transformer", "connectors", "vocoder", "text_encoder", "full",
    ])
    args = parser.parse_args()

    log(f"RAM available at start: {ram_available_gb():.1f}GB")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        log(f"VRAM free at start: {free/1024**3:.1f}GB / {total/1024**3:.1f}GB")

    if args.stage == "configs":
        stage_configs()
    elif args.stage == "vae":
        stage_vae()
    elif args.stage == "transformer":
        stage_transformer()
    elif args.stage == "connectors":
        stage_connectors()
    elif args.stage == "vocoder":
        stage_vocoder()
    elif args.stage == "text_encoder":
        stage_text_encoder()
    elif args.stage == "full":
        stage_full()

    log(f"RAM available at end: {ram_available_gb():.1f}GB")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        log(f"VRAM free at end: {free/1024**3:.1f}GB / {total/1024**3:.1f}GB")


if __name__ == "__main__":
    main()
