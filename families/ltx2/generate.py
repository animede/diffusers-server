# -*- coding: utf-8 -*-
"""
LTX-2.3 生成本体(T2V / I2V / FLF / IA2V)。音声(vocoder出力、またはIA2Vでは入力wav原音)を
wav 化して ffmpeg で mp4 へ mux する。

蒸留(distilled fp8)チェックポイント既定: steps=8, guidance_scale=1.0(CFGなし)。
negative_prompt は蒸留モデルでは実質無効(guidance_scale=1.0だとdo_classifier_free_guidance
がFalseになりnegative_prompt_embedsが計算されない、diffusers共通のCFG設計)だが、
UIから渡された場合はそのままパイプラインへ渡す(実害はない)。

IA2V(Image+Audio to Video、リップシンク動画、2026-07-19追加): tools/ltx2_ia2v_probe.py で
実証済みのカスタムdenoiseループ(diffusers本体無改変)を run_ia2v() として移植したもの。
機構の詳細は同スクリプトのモジュールdocstring参照。要点:
  - 音声latentを入力wav由来のclean latentへ毎ステップx0固定(mask=1.0)し、映像側だけを
    denoiseする(ComfyUI IA2Vの noise_mask=0 ハードロックと同等の効果を、diffusers本体を
    変更せずに呼び出し側のループで再現)。
  - use_cross_timestep=True を採用(A/B検証の結果、diffusers本体のdocstring
    ("True is the newer (e.g. LTX-2.3) behavior; False is the legacy LTX-2.0 behavior")と
    整合したため。動画フレームの差分は認められたが、リップシンク品質そのものに優劣は
    見られず、チェックポイント世代(2.3)に合わせた設定を正とした)。
  - audio_out=original(既定): 入力wavの原音をそのままmux(ボコーダ再合成音ではない)。
    audio latentは同期のためだけに使い、聴かせる音は元wavから直接持ってくる。
  - デコード前クリーンアップは「不要になった大型テンソル(CFGバッファ・prompt embeds等)の
    del + empty_cache()」のみ。text_encoder/connectors の丸ごとCPU退避は行わない
    (CLAUDE.md 33番の禁止パターン。groupモードでは transformer ~35GB がホストRAM常駐の
    ため、TE ~23GB の退避を重ねると計~58GBでホストRAM 62GB を食い潰し、OOMキラーによる
    プロセス強制終了を実機で引き起こした(2026-07-20))。
  - 代わりに、groupモード(48GB級)では VAE decode を時間方向 tiled decode で行い、
    decode の一時ピーク(実測: 512x320・89フレームの一括 decode で OOM)を分割して
    VRAM に収める(run_ia2v() の decode 部参照)。noneモード(96GB級)は一括 decode。

latent upsampler(低解像度生成 -> 潜在空間アップスケール -> 高解像度デコードの2段生成、
2026-07-20追加): T2V/I2V の任意パラメータ upscale=1(既定0=無効、従来経路は完全無変更)。
_run_upscaled() が共通実装(詳細はその関数の直前のモジュールレベルdocstring参照)。要点:
  - pipe(..., output_type="latent") で得られる result.frames は「denormalize済みの
    5D動画latent」(diffusersの_unpack_latents/_denormalize_latents適用後、pixel変換前)
    であり、LTX2LatentUpsamplerModel が要求する「正規化前のlatent」とそのまま一致する。
  - アップスケール後は decode 対象のトークン数が4倍になるため、offload_mode(none/group)
    に関わらず常時 tiled decode(vae.enable_tiling() + use_framewise_decoding=True)を
    適用する(none モードでも 512x320 の 2x で溢れることを実測で確認済み、
    _decode_video_latents_tiled() 参照)。
"""
import copy
import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from PIL import Image

from core import progress as progress_mod
from families.ltx2 import pipeline as pipeline_mod
from families.ltx2 import state
from families.ltx2.runtime import (
    DEFAULT_AUDIO_OUT,
    DEFAULT_FLF_FRAME_RATE,
    DEFAULT_FLF_STRENGTH,
    DEFAULT_FPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_IA2V_AUDIO_OUT,
    DEFAULT_IA2V_FRAME_RATE,
    DEFAULT_IA2V_NEGATIVE_PROMPT,
    DEFAULT_IA2V_PROMPT,
    DEFAULT_ICLORA_MASK_COLOR,
    DEFAULT_ICLORA_NEGATIVE_PROMPT,
    DEFAULT_ICLORA_REF_STRENGTH,
    DEFAULT_NUM_FRAMES,
    DEFAULT_STEPS,
    DEFAULT_V2A_MAX_NUM_FRAMES,
    DEFAULT_V2A_NEGATIVE_PROMPT,
    DEFAULT_V2A_PROMPT,
    DEFAULT_WIDTH,
)

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

AUDIO_SAMPLE_RATE = 48000  # LTX2VocoderWithBWE output_sampling_rate(vocoder config既定)

FFMPEG_BIN = shutil.which("ffmpeg")


def _new_output_path(mode: str, ext: str, ts: "str | None" = None, job_id: "str | None" = None):
    ts = ts or datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = job_id or uuid.uuid4().hex[:8]
    name = f"{mode}_{ts}_{job_id}.{ext}"
    path = os.path.join(OUTPUTS_DIR, name)
    return path, name


def make_generator(seed: Optional[int]):
    if seed is None or int(seed) < 0:
        return None, None
    seed = int(seed)
    return torch.Generator(device="cpu").manual_seed(seed), seed


def _reset_vram_stats():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_vram_gb():
    return torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else None


def _progress_callback(mode: str, total_steps: int):
    progress_mod.start_generating(mode, total_steps)

    def _cb(pipe, step_index, timestep, callback_kwargs):
        progress_mod.update_step(step_index + 1, total_steps)
        return callback_kwargs

    return _cb


def _pipeline_meta() -> dict:
    bps = state.base_pipeline_state
    return {
        "model": "ltx-2.3-22b-distilled-fp8",
        "offload_mode": bps.get("offload_mode"),
    }


# ============================================================================
# latent upsampler(低解像度生成 -> 潜在空間アップスケール -> 高解像度デコードの2段生成、
# 2026-07-20追加)。T2V/I2V の任意パラメータ upscale=1 で有効化する(既定0=無効、
# 従来経路は完全に無変更)。
#
# 機構: base/I2V パイプラインを output_type="latent" で実行すると、
# LTX2Pipeline.__call__()/LTX2ImageToVideoPipeline.__call__() は denoise 後にまず
# _unpack_latents() で [B,S,D] パック形式を [B,C,F,H,W] 動画テンソルへ戻し、その後
# _denormalize_latents() を適用してから result.frames として返す(video_processor
# による pixel 変換はスキップされる、diffusersソース確認済み)。つまり
# output_type="latent" 時の result.frames は「denormalize済みの5D動画latent」であり、
# LTX2LatentUpsamplePipeline が要求する入力形式(vae.encode()直後、正規化前の生latent)と
# そのまま一致する(diffusersのEXAMPLE_DOC_STRINGでもこの前提でupsample_pipeを呼んでいる)。
# audio側もresult.audioとして「denormalize済み・unpack済みのaudio latent」が返る
# (audio_vae.decode()がそのまま受け付ける形式、通常のvocoderデコード経路と同一)。
#
# よって upscale=1 時は:
#   1. pipe(..., output_type="latent") で低解像度の denoise を実行
#   2. video latentsを upsampler(unnormalizedのまま)に通す(2x spatial)
#   3. upsampler出力を正規化せず **そのまま** vae.decode() へ渡す(diffusers本家
#      LTX2LatentUpsamplePipeline.__call__() と同じ。ここで _normalize_latents() を
#      挟むと色空間が崩壊する — 実際に発症したバグ、2026-07-20修正)
#   4. audio latentsはそのまま audio_vae.decode() -> vocoder(アップスケール対象外)
#
# VRAM対策: アップスケール後の latents はトークン数(H*W)4倍になるため、decode の
# 一時ピークも約4倍になる。実測(下記docstring・CLAUDE.md参照)で group モードはもちろん
# none モードでも 512x320 の 2x(1024x640相当)で tiled decode が必要と判明したため、
# upscale=1 時は offload_mode に関わらず vae.enable_tiling() +
# use_framewise_decoding=True を常時適用する(CLAUDE.md 39番の罠: enable_tiling() だけ
# では空間タイルしか有効にならず、時間方向は use_framewise_decoding の手動設定が必須)。
# ============================================================================


def _decode_video_latents_tiled(pipe, latents: torch.Tensor) -> torch.Tensor:
    """vae.decode() を時間方向 tiled decode で実行し、常に元のフラグへ戻す。

    upscale時は decode の一時ピークが大きい(4倍のトークン数)ため、offload_mode に
    関わらず常時 tiled decode を適用する(run_ia2v()/run_keyframes()等は group モード
    時のみ適用しているが、upscale decode は none モードでも溢れることを実測で確認済み)。
    """
    prev_tiling = (pipe.vae.use_tiling, pipe.vae.use_framewise_decoding)
    pipe.vae.enable_tiling()
    pipe.vae.use_framewise_decoding = True
    try:
        with torch.no_grad():
            video = pipe.vae.decode(latents, None, return_dict=False)[0]
    finally:
        pipe.vae.use_tiling, pipe.vae.use_framewise_decoding = prev_tiling
    return video


def _run_upscaled(pipe, call_kwargs: dict, fps: float):
    """base/I2V パイプラインを output_type="latent" で実行し、latent upsampler で
    2x spatial アップスケールしてから手動デコードする。戻り値は通常の pipe(...) 呼び出しと
    互換の (frames_pil_list, audio_tensor) タプル(frames_pil_list は export_to_video に
    そのまま渡せる PIL フレーム列)。
    """
    upsampler = pipeline_mod.get_upsampler()

    with torch.no_grad():
        result = pipe(**call_kwargs, output_type="latent")

    video_latents = result.frames  # denormalize済み [B,C,F,H,W](vae.encode直後と同じ形式)
    audio_latents = result.audio  # denormalize済み・unpack済み audio latent

    progress_mod.set_phase("decoding")

    with torch.no_grad():
        video_latents = video_latents.to(upsampler.dtype)
        upsampled = upsampler(video_latents)
        del video_latents
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 重要: upsampler出力は unnormalized latent のまま vae.decode() へ直接渡す。
        # diffusers本家 LTX2LatentUpsamplePipeline.__call__() も upsample後に正規化せず
        # そのまま decode している。ここに _normalize_latents() を挟むと decode が
        # 正規化済み潜在を受け取って色空間が崩壊する(サイケデリックな色化けとして
        # 実際に発症・修正済み、2026-07-20)。
        upsampled = upsampled.to(pipe.vae.dtype)

        # アップスケール後は decode 対象のトークン数が4倍(H*Wが2x*2x)になるため、
        # offload_mode に関わらず常に tiled decode を適用する(モジュールdocstring参照)。
        video = _decode_video_latents_tiled(pipe, upsampled)
        del upsampled
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    video_frames = pipe.video_processor.postprocess_video(video, output_type="pil")

    with torch.no_grad():
        audio_latents = audio_latents.to(pipe.audio_vae.dtype)
        generated_mel = pipe.audio_vae.decode(audio_latents, return_dict=False)[0]
        audio = pipe.vocoder(generated_mel)

    return video_frames[0], audio


def _mux_wav_to_video(video_path: str, audio_path: str, mode: str, ts: str, job_id: str) -> "str | None":
    """wav(既に保存済み)を ffmpeg で mp4 へ mux する共通ヘルパ。失敗時は None(非致命的)。"""
    if not FFMPEG_BIN:
        print("[families.ltx2] ffmpeg が見つからないため mux をスキップします(映像のみmp4+wav別ファイル)")
        return None
    muxed_path, muxed_name = _new_output_path(f"{mode}_av", "mp4", ts=ts, job_id=job_id)
    try:
        subprocess.run(
            [
                FFMPEG_BIN, "-y", "-loglevel", "error",
                "-i", video_path, "-i", audio_path,
                "-c:v", "copy", "-c:a", "aac", "-shortest", muxed_path,
            ],
            check=True,
            timeout=120,
        )
        return f"/outputs/{muxed_name}"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"[families.ltx2] ffmpeg mux に失敗しました(非致命的): {exc}")
        return None


def _save_audio_and_mux(
    audio, video_path: str, ts: str, job_id: str, mode: str = "ltx2_t2v", audio_out: str = "on"
) -> "tuple[str | None, str | None, bool]":
    """vocoder出力(audio tensor)を wav 保存し、ffmpeg があれば mp4 へ mux する。

    audio_out="off" の場合は音声保存自体を行わず、映像のみのmp4(video_path)を返す
    (無音mp4、タスク要件2番)。

    戻り値: (audio_url, muxed_video_url, has_audio)。ffmpeg が無い/失敗した場合は
    映像のみのmp4(video_path)と別ファイルのwavをそれぞれ返す(タスク指示: 「なければ
    映像のみmp4+wav別ファイルで可」)。
    """
    if audio_out == "off" or audio is None:
        return None, None, False
    try:
        import soundfile as sf
    except ImportError:
        print("[families.ltx2] soundfile が無いため音声保存をスキップします")
        return None, None, False

    if isinstance(audio, (list, tuple)):
        audio = audio[0]
    wav = audio.detach().float().cpu()
    if wav.ndim == 3:
        wav = wav[0]
    # (channels, samples) -> (samples, channels)
    if wav.ndim == 2 and wav.shape[0] <= 8:
        wav = wav.transpose(0, 1)

    audio_path, audio_name = _new_output_path(mode, "wav", ts=ts, job_id=job_id)
    sf.write(audio_path, wav.numpy(), AUDIO_SAMPLE_RATE)
    audio_url = f"/outputs/{audio_name}"

    muxed_url = _mux_wav_to_video(video_path, audio_path, mode, ts, job_id)
    return audio_url, muxed_url, True


def run_t2v(req: dict) -> dict:
    pipe = pipeline_mod.get_base_pipeline()
    generator, used_seed = make_generator(req.get("seed", -1))

    width = int(req.get("width") or DEFAULT_WIDTH)
    height = int(req.get("height") or DEFAULT_HEIGHT)
    num_frames = int(req.get("num_frames") or DEFAULT_NUM_FRAMES)
    fps = int(req.get("fps") or DEFAULT_FPS)
    steps = int(req.get("steps") or DEFAULT_STEPS)
    guidance_scale = float(req.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
    negative_prompt = req.get("negative_prompt") or None
    audio_out = (req.get("audio_out") or DEFAULT_AUDIO_OUT).strip().lower()
    if audio_out not in ("on", "off"):
        raise ValueError(f"audio_out は on/off のいずれかです(t2v): {audio_out!r}")
    upscale = bool(int(req.get("upscale") or 0))

    call_kwargs = dict(
        prompt=req["prompt"],
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=float(fps),
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        callback_on_step_end=_progress_callback("ltx2_t2v", steps),
    )

    _reset_vram_stats()
    t0 = time.time()
    if upscale:
        frames, audio = _run_upscaled(pipe, call_kwargs, fps)
    else:
        with torch.no_grad():
            result = pipe(**call_kwargs)
        frames = result.frames[0]
        audio = getattr(result, "audio", None)
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    from diffusers.utils import export_to_video

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = uuid.uuid4().hex[:8]
    video_path, video_name = _new_output_path("ltx2_t2v", "mp4", ts=ts, job_id=job_id)
    export_to_video(frames, video_path, fps=fps)

    audio_url, muxed_url, has_audio = _save_audio_and_mux(audio, video_path, ts, job_id, audio_out=audio_out)

    meta = {
        "mode": "ltx2_t2v",
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", ""),
        "width": width * 2 if upscale else width,
        "height": height * 2 if upscale else height,
        "num_frames": num_frames,
        "fps": fps,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": used_seed,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "video_url": muxed_url or f"/outputs/{video_name}",
        "video_only_url": f"/outputs/{video_name}",
        "audio_url": audio_url,
        "has_audio": has_audio,
        "audio_out": audio_out,
        "upscale": upscale,
    }
    meta.update(_pipeline_meta())
    return meta


def run_flf(req: dict, first_image: Image.Image, last_image: Image.Image) -> dict:
    """FLF(First-Last-Frame): 最初と最後のフレーム画像を指定し、間の動画を生成する。

    ComfyUIワークフロー(video_ltx2_3-22b-flf-bf8.json)の LTXVAddGuide ×2
    (条件1: 画像A, frame_idx=0, strength=0.7 / 条件2: 画像B, frame_idx=-1, strength=0.7)
    に対応する。LTX2ConditionPipeline.prepare_conditions() が画像を「長辺基準でリサイズして
    target(height, width)へ中央クロップ」する実装(ComfyUIの ResizeImageMaskNode
    "scale dimensions" + center crop と同じ挙動)のため、呼び出し側で事前リサイズは行わない。

    upscale=1 で T2V/I2V と同じ latent upsampler 2段生成に切り替わる(2026-07-23追加、
    CLAUDE.md 51番)。LTX2ConditionPipeline も output_type="latent" 時は
    「条件トークン除去 → _unpack_latents() → _denormalize_latents()」の順で処理した
    5D latent を返す(pipeline_ltx2_condition.py で1行単位に確認済み、43番の色化け事故対策)。
    audio latents も output_type 分岐の前に無条件で denormalize+unpack されるため、
    どちらも _run_upscaled() が前提とする形式と完全一致する。
    """
    pipe = pipeline_mod.get_flf_pipeline()
    generator, used_seed = make_generator(req.get("seed", -1))

    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition

    width = int(req.get("width") or DEFAULT_WIDTH)
    height = int(req.get("height") or DEFAULT_HEIGHT)
    num_frames = int(req.get("num_frames") or DEFAULT_NUM_FRAMES)
    fps = float(req.get("fps") or DEFAULT_FLF_FRAME_RATE)
    steps = int(req.get("steps") or DEFAULT_STEPS)
    guidance_scale = float(req.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
    strength = float(req.get("strength") or DEFAULT_FLF_STRENGTH)
    negative_prompt = req.get("negative_prompt") or None
    audio_out = (req.get("audio_out") or DEFAULT_AUDIO_OUT).strip().lower()
    if audio_out not in ("on", "off"):
        raise ValueError(f"audio_out は on/off のいずれかです(flf): {audio_out!r}")
    upscale = bool(int(req.get("upscale") or 0))

    conditions = [
        LTX2VideoCondition(frames=first_image, index=0, strength=strength),
        LTX2VideoCondition(frames=last_image, index=-1, strength=strength),
    ]

    call_kwargs = dict(
        conditions=conditions,
        prompt=req["prompt"],
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=fps,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        callback_on_step_end=_progress_callback("ltx2_flf", steps),
    )

    _reset_vram_stats()
    t0 = time.time()
    if upscale:
        # 低解像度denoise → latent 2x upsample → tiled decode(i2v/t2vと共通機構)。
        # 高解像度の直接denoiseはattention中間テンソルがシーケンス長のほぼ2乗で膨らみ、
        # 1280×704×121フレームで87GB超のCUDA OOMを起こす(実測、CLAUDE.md 51番)ため
        # 必ずこの2段経路を使うこと。
        frames, audio = _run_upscaled(pipe, call_kwargs, fps)
    else:
        with torch.no_grad():
            result = pipe(**call_kwargs)
        frames = result.frames[0]
        audio = getattr(result, "audio", None)
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    from diffusers.utils import export_to_video

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = uuid.uuid4().hex[:8]
    video_path, video_name = _new_output_path("ltx2_flf", "mp4", ts=ts, job_id=job_id)
    export_to_video(frames, video_path, fps=fps)

    audio_url, muxed_url, has_audio = _save_audio_and_mux(
        audio, video_path, ts, job_id, mode="ltx2_flf", audio_out=audio_out
    )

    meta = {
        "mode": "ltx2_flf",
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", ""),
        "width": width * 2 if upscale else width,
        "height": height * 2 if upscale else height,
        "num_frames": num_frames,
        "fps": fps,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "strength": strength,
        "seed": used_seed,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "video_url": muxed_url or f"/outputs/{video_name}",
        "video_only_url": f"/outputs/{video_name}",
        "audio_url": audio_url,
        "has_audio": has_audio,
        "audio_out": audio_out,
        "upscale": upscale,
    }
    meta.update(_pipeline_meta())
    return meta


def run_keyframes(
    req: dict,
    images: "list[Image.Image]",
    indices: "list[int]",
    strengths: "list[float]",
) -> dict:
    """任意キーフレーム条件付け: N枚の画像を任意フレーム位置・任意strengthで条件付ける。

    FLF(run_flf())の一般化。LTX2ConditionPipeline.prepare_conditions() は
    conditions=[LTX2VideoCondition(frames, index, strength), ...] を任意個数受け付ける
    汎用実装で、FLF はその特殊ケース(index=0とindex=-1の2条件のみ)にすぎない
    (CLAUDE.md 38番参照)。index は負値で末尾からの指定にネイティブ対応する
    (prepare_conditions() が latent_start_idx % latent_num_frames で解決)。

    画像の事前リサイズは行わない(prepare_conditions() が長辺基準リサイズ+中央クロップを
    自動で行う、FLFと同じ挙動)。
    """
    pipe = pipeline_mod.get_flf_pipeline()
    generator, used_seed = make_generator(req.get("seed", -1))

    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition

    if not images:
        raise ValueError("keyframes には画像が1枚以上必要です")
    if len(indices) != len(images):
        raise ValueError(
            f"indices の個数({len(indices)})が images の個数({len(images)})と一致していません"
        )
    if len(strengths) == 1:
        strengths = strengths * len(images)
    if len(strengths) != len(images):
        raise ValueError(
            f"strengths の個数({len(strengths)})が images の個数({len(images)})と一致していません"
            "(1個だけ指定した場合は全画像に適用されます)"
        )

    width = int(req.get("width") or DEFAULT_WIDTH)
    height = int(req.get("height") or DEFAULT_HEIGHT)
    num_frames = int(req.get("num_frames") or DEFAULT_NUM_FRAMES)
    fps = float(req.get("fps") or DEFAULT_FLF_FRAME_RATE)
    steps = int(req.get("steps") or DEFAULT_STEPS)
    guidance_scale = float(req.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
    negative_prompt = req.get("negative_prompt") or None
    audio_out = (req.get("audio_out") or DEFAULT_AUDIO_OUT).strip().lower()
    if audio_out not in ("on", "off"):
        raise ValueError(f"audio_out は on/off のいずれかです(keyframes): {audio_out!r}")

    conditions = [
        LTX2VideoCondition(frames=img, index=idx, strength=strength)
        for img, idx, strength in zip(images, indices, strengths)
    ]

    _reset_vram_stats()
    t0 = time.time()

    # groupモード(48GB級)では VAE decode の一時ピークが収まらないため、run_ia2v() の
    # decode 部と同じパターン(時間方向tiled decode)で pipe(...) 呼び出し全体を包む
    # (CLAUDE.md 39番: enable_tiling() だけでは時間方向タイルは有効にならず、
    # use_framewise_decoding を別途 True にする必要がある)。vae は他モードと共有のため
    # finally で必ず元のフラグへ戻す。
    tiled_decode = state.base_pipeline_state.get("offload_mode") == "group"
    if tiled_decode:
        prev_tiling = (pipe.vae.use_tiling, pipe.vae.use_framewise_decoding)
        pipe.vae.enable_tiling()
        pipe.vae.use_framewise_decoding = True
    try:
        with torch.no_grad():
            result = pipe(
                conditions=conditions,
                prompt=req["prompt"],
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=fps,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                callback_on_step_end=_progress_callback("ltx2_keyframes", steps),
            )
    finally:
        if tiled_decode:
            pipe.vae.use_tiling, pipe.vae.use_framewise_decoding = prev_tiling
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    from diffusers.utils import export_to_video

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = uuid.uuid4().hex[:8]
    frames = result.frames[0]
    video_path, video_name = _new_output_path("ltx2_keyframes", "mp4", ts=ts, job_id=job_id)
    export_to_video(frames, video_path, fps=fps)

    audio = getattr(result, "audio", None)
    audio_url, muxed_url, has_audio = _save_audio_and_mux(
        audio, video_path, ts, job_id, mode="ltx2_keyframes", audio_out=audio_out
    )

    meta = {
        "mode": "ltx2_keyframes",
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", ""),
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": fps,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "indices": indices,
        "strengths": strengths,
        "num_keyframes": len(images),
        "seed": used_seed,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "video_url": muxed_url or f"/outputs/{video_name}",
        "video_only_url": f"/outputs/{video_name}",
        "audio_url": audio_url,
        "has_audio": has_audio,
        "audio_out": audio_out,
    }
    meta.update(_pipeline_meta())
    return meta


def run_i2v(req: dict, src_image: Image.Image) -> dict:
    pipe = pipeline_mod.get_i2v_pipeline()
    generator, used_seed = make_generator(req.get("seed", -1))

    width = int(req.get("width") or DEFAULT_WIDTH)
    height = int(req.get("height") or DEFAULT_HEIGHT)
    num_frames = int(req.get("num_frames") or DEFAULT_NUM_FRAMES)
    fps = int(req.get("fps") or DEFAULT_FPS)
    steps = int(req.get("steps") or DEFAULT_STEPS)
    guidance_scale = float(req.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
    negative_prompt = req.get("negative_prompt") or None
    audio_out = (req.get("audio_out") or DEFAULT_AUDIO_OUT).strip().lower()
    if audio_out not in ("on", "off"):
        raise ValueError(f"audio_out は on/off のいずれかです(i2v): {audio_out!r}")
    upscale = bool(int(req.get("upscale") or 0))

    call_kwargs = dict(
        image=src_image,
        prompt=req["prompt"],
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=float(fps),
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        callback_on_step_end=_progress_callback("ltx2_i2v", steps),
    )

    _reset_vram_stats()
    t0 = time.time()
    if upscale:
        frames, audio = _run_upscaled(pipe, call_kwargs, fps)
    else:
        with torch.no_grad():
            result = pipe(**call_kwargs)
        frames = result.frames[0]
        audio = getattr(result, "audio", None)
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    from diffusers.utils import export_to_video

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = uuid.uuid4().hex[:8]
    video_path, video_name = _new_output_path("ltx2_i2v", "mp4", ts=ts, job_id=job_id)
    export_to_video(frames, video_path, fps=fps)

    audio_url, muxed_url, has_audio = _save_audio_and_mux(
        audio, video_path, ts, job_id, mode="ltx2_i2v", audio_out=audio_out
    )

    meta = {
        "mode": "ltx2_i2v",
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", ""),
        "width": width * 2 if upscale else width,
        "height": height * 2 if upscale else height,
        "num_frames": num_frames,
        "fps": fps,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": used_seed,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "video_url": muxed_url or f"/outputs/{video_name}",
        "video_only_url": f"/outputs/{video_name}",
        "audio_url": audio_url,
        "has_audio": has_audio,
        "audio_out": audio_out,
        "upscale": upscale,
    }
    meta.update(_pipeline_meta())
    return meta


# ============================================================================
# IA2V(Image + Audio to Video、リップシンク動画)
# tools/ltx2_ia2v_probe.py の stage_full() を移植・整理したもの。ロジックは実証済みの
# ままで、以下の点を本番向けに変更している:
#   - 音声latentの生成元を「入力wavファイル」から都度読む(スクリプトの固定パス定数を
#     廃止し、run_ia2v(req, image, audio_path) の引数で受け取る)。
#   - audio_out(original/vocoder/none)による出力音声の切り替え(タスク要件1・2番)。
#   - probe にあったデコード段の text_encoder/connectors CPU退避は削除(モジュール
#     docstring参照。del + empty_cache() のみ維持)。
#   - use_cross_timestep=True 固定(A/B検証の結論、モジュールdocstring参照)。
# ============================================================================


def _load_wav_soundfile(audio_path: str):
    """wav を [C, T] の float32 tensor として読む(soundfile、torchaudio.load() は
    venv の torchaudio が新backend(torchcodec必須)のため使わない、probe と同じ理由)。
    原音mux用(audio_out="original")にも、mel変換の入力(リサンプリング前)にも
    同じ関数を使い、呼び出し側でそれぞれ別変数に保持する(混同防止)。
    """
    import soundfile as sf

    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)  # [T, C]
    waveform = torch.from_numpy(data).transpose(0, 1).contiguous()  # [C, T]
    return waveform, sr


def _wav_to_mel_and_audio_latents(audio_path: str, pipe, device: str, dtype: torch.dtype):
    """wav -> mel spectrogram -> audio_vae.encode() で clean audio latent(5D)を得る。

    抽出元: tools/ltx2_ia2v_probe.py の load_wav_as_mel()(mel変換ロジックは
    ComfyUI AudioPreprocessor.waveform_to_mel と同一)。戻り値は
    (clean_audio_latents_5d, wav_duration_s, waveform_for_original_mux, sr_orig)。
    waveform_for_original_mux は audio_out="original" 用に、リサンプリング前の
    原音そのまま([C, T], 元sr)を別途返す(mel変換用に16kHzへリサンプルした
    ものと混同しないよう明確に分離する)。
    """
    import torchaudio

    waveform_orig, sr_orig = _load_wav_soundfile(audio_path)  # 原音mux用(未加工)

    waveform, sr = _load_wav_soundfile(audio_path)
    target_sr = int(pipe.audio_vae.config.sample_rate)
    mel_bins = int(pipe.audio_vae.config.mel_bins)
    hop_length = int(pipe.audio_vae.config.mel_hop_length)
    n_fft = 1024
    win_length = 1024
    mel_fmin = 0
    mel_fmax = target_sr / 2.0

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    if waveform.shape[0] == 1:
        waveform = waveform.expand(2, -1)

    waveform = waveform.unsqueeze(0).to(device)  # [B=1, C=2, T]

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=mel_fmin,
        f_max=mel_fmax,
        n_mels=mel_bins,
        power=1.0,
        mel_scale="slaney",
    ).to(device)

    mel = mel_transform(waveform)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    mel = mel.permute(0, 1, 3, 2).contiguous()  # [B, C, T_mel, mel_bins]
    mel = mel.to(pipe.audio_vae.dtype)

    with torch.no_grad():
        posterior = pipe.audio_vae.encode(mel).latent_dist
        clean_audio_latents_5d = posterior.mode()

    wav_duration_s = waveform.shape[-1] / target_sr
    return clean_audio_latents_5d, wav_duration_s, waveform_orig, sr_orig


def _frames_from_audio_duration(duration_s: float, fps: float) -> int:
    """音声長(秒)×fps から動画フレーム数を自動計算し、8n+1へ丸める(タスク要件5番)。"""
    raw = duration_s * fps
    n = max(1, round((raw - 1) / 8))
    return int(8 * n + 1)


def _save_original_audio_and_mux(
    audio_path: str, video_path: str, video_num_frames: int, fps: float, ts: str, job_id: str
) -> "tuple[str | None, str | None, bool]":
    """audio_out="original": 入力wavの原音をそのまま動画長に合わせてmuxする。

    動画長(video_num_frames/fps)と音声長の不一致は ffmpeg 側で吸収する:
    音声が動画より長い場合は "-shortest" 相当の -t トリム、動画が音声より
    長い場合は音声側を無音パディングする(-af apad で不足分を無音埋め)。
    """
    if not FFMPEG_BIN:
        print("[families.ltx2] ffmpeg が見つからないため原音muxをスキップします")
        return None, None, False

    video_duration_s = video_num_frames / float(fps)
    muxed_path, muxed_name = _new_output_path("ltx2_ia2v_av", "mp4", ts=ts, job_id=job_id)
    try:
        subprocess.run(
            [
                FFMPEG_BIN, "-y", "-loglevel", "error",
                "-i", video_path, "-i", audio_path,
                "-c:v", "copy", "-c:a", "aac",
                "-af", "apad",  # 音声が動画より短い場合は無音パディング
                "-t", f"{video_duration_s:.6f}",  # 動画長にトリム/延長を揃える
                muxed_path,
            ],
            check=True,
            timeout=120,
        )
        return f"/outputs/{os.path.basename(audio_path)}", f"/outputs/{muxed_name}", True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"[families.ltx2] 原音muxに失敗しました(非致命的): {exc}")
        return None, None, False


def run_ia2v(req: dict, image: Image.Image, audio_path: str) -> dict:
    """IA2V(Image + Audio to Video): 参照画像 + wav音声 からリップシンク動画を生成する。

    機構: tools/ltx2_ia2v_probe.py の stage_full() を移植。diffusers本体
    (LTX2ConditionPipeline.__call__())は改変せず、そのdenoiseループをコピー・改修した
    カスタムループをここに実装する。音声latentを入力wav由来のclean latentへ毎ステップ
    x0固定(audio_conditioning_mask=1.0)し、映像側だけをdenoiseすることで、
    ComfyUIのnoise_mask=0ハードロックと同等の「音声は既知信号、映像だけが同期して
    生成される」条件付けを実現する(詳細は同スクリプトのdocstring参照)。
    """
    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import (
        LTX2VideoCondition,
        calculate_shift,
        retrieve_timesteps,
    )
    from diffusers.utils import export_to_video

    pipe = pipeline_mod.get_flf_pipeline()  # LTX2ConditionPipeline、base components 参照共有
    generator, used_seed = make_generator(req.get("seed", -1))

    device = pipe._execution_device
    dtype = torch.float32

    width = int(req.get("width") or DEFAULT_WIDTH)
    height = int(req.get("height") or DEFAULT_HEIGHT)
    fps = float(req.get("fps") or DEFAULT_IA2V_FRAME_RATE)
    steps = int(req.get("steps") or DEFAULT_STEPS)
    guidance_scale = float(req.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
    audio_guidance_scale = float(req.get("audio_guidance_scale", guidance_scale))
    prompt = req.get("prompt") or DEFAULT_IA2V_PROMPT
    negative_prompt = req.get("negative_prompt") or DEFAULT_IA2V_NEGATIVE_PROMPT
    audio_out = (req.get("audio_out") or DEFAULT_IA2V_AUDIO_OUT).strip().lower()
    if audio_out not in ("original", "vocoder", "none"):
        raise ValueError(f"audio_out は original/vocoder/none のいずれかです(ia2v): {audio_out!r}")

    _reset_vram_stats()
    progress_mod.start_loading("ltx2_ia2v")

    # --- 音声 -> mel -> audio_vae.encode() ---
    (
        clean_audio_latents_5d,
        wav_duration_s,
        waveform_orig,
        sr_orig,
    ) = _wav_to_mel_and_audio_latents(audio_path, pipe, device, dtype)

    num_frames = req.get("num_frames")
    if num_frames:
        num_frames = int(num_frames)
    else:
        num_frames = _frames_from_audio_duration(wav_duration_s, fps)

    do_cfg = guidance_scale > 1.0 or audio_guidance_scale > 1.0

    t0 = time.time()

    # 重要(tools/ltx2_ia2v_probe.py には無かった修正、48GB group offload検証で発見):
    # encode_prompt(Gemma3-12B forward、max_length=1024・output_hidden_states=True)を
    # torch.no_grad() なしで呼ぶと autograd グラフ用の中間活性化が保持され、通常より
    # 大幅に多い一時VRAMを消費する(96GB機のnoneモードでは表面化しなかったが、48GB相当の
    # groupモードでは大幅な余分ピークとなりOOMを実機で再現した)。denoiseループを含む
    # 以降の処理はすべて勾配計算が不要な推論専用コードのため、この関数の残り全体を
    # 1つの torch.no_grad() で包む。
    with torch.no_grad():
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            device=device,
        )
        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = pipe.connectors(
            prompt_embeds, prompt_attention_mask, padding_side=pipe.tokenizer_padding_side
        )

        # --- video latents(first-frame画像条件、strength=1.0) ---
        conditions = [LTX2VideoCondition(frames=image, index=0, strength=1.0)]
        num_channels_latents = pipe.transformer.config.in_channels
        noise_scale = 1.0

        latents, conditioning_mask, clean_latents, keyframe_coords = pipe.prepare_latents(
            conditions=conditions,
            batch_size=1,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=fps,
            noise_scale=noise_scale,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        conditioning_mask_full = torch.cat([conditioning_mask, conditioning_mask]) if do_cfg else conditioning_mask

        latent_num_frames = (num_frames - 1) // pipe.vae_temporal_compression_ratio + 1
        latent_height = height // pipe.vae_spatial_compression_ratio
        latent_width = width // pipe.vae_spatial_compression_ratio

        # --- audio latents: 与えられた wav の latent に常に固定する(hard lock) ---
        duration_s = num_frames / fps
        audio_latents_per_second = (
            pipe.audio_sampling_rate / pipe.audio_hop_length / float(pipe.audio_vae_temporal_compression_ratio)
        )
        audio_num_frames = round(duration_s * audio_latents_per_second)

        L = clean_audio_latents_5d.shape[2]
        if L >= audio_num_frames:
            clean_audio_latents_5d = clean_audio_latents_5d[:, :, :audio_num_frames]
        else:
            pad = audio_num_frames - L
            clean_audio_latents_5d = torch.nn.functional.pad(clean_audio_latents_5d, (0, 0, 0, pad))
        clean_audio_latents_5d = clean_audio_latents_5d.to(device=device, dtype=dtype)

        clean_audio_packed = pipe._normalize_audio_latents(
            pipe._pack_audio_latents(clean_audio_latents_5d), pipe.audio_vae.latents_mean, pipe.audio_vae.latents_std
        )

        audio_conditioning_mask = torch.full(
            (clean_audio_packed.shape[0], clean_audio_packed.shape[1], 1),
            1.0, device=device, dtype=dtype,  # 常にclean(=入力wav)に固定(hard lock)
        )
        audio_latents = clean_audio_packed.clone()
        audio_conditioning_mask_full = (
            torch.cat([audio_conditioning_mask, audio_conditioning_mask]) if do_cfg else audio_conditioning_mask
        )

        # --- timesteps / scheduler ---
        sigmas = np.linspace(1.0, 1 / steps, steps)
        mu = calculate_shift(
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_image_seq_len", 1024),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.95),
            pipe.scheduler.config.get("max_shift", 2.05),
        )
        audio_scheduler = copy.deepcopy(pipe.scheduler)
        audio_timesteps, _ = retrieve_timesteps(audio_scheduler, steps, device, None, sigmas=sigmas, mu=mu)
        timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, steps, device, None, sigmas=sigmas, mu=mu)

        video_coords = pipe.transformer.rope.prepare_video_coords(
            latents.shape[0], latent_num_frames, latent_height, latent_width, latents.device, fps=fps
        )
        audio_coords = pipe.transformer.audio_rope.prepare_audio_coords(
            audio_latents.shape[0], audio_num_frames, audio_latents.device
        )
        if keyframe_coords is not None:
            video_coords = torch.cat([video_coords, keyframe_coords], dim=2)
        if do_cfg:
            video_coords = video_coords.repeat((2,) + (1,) * (video_coords.ndim - 1))
            audio_coords = audio_coords.repeat((2,) + (1,) * (audio_coords.ndim - 1))

        progress_mod.start_generating("ltx2_ia2v", steps)

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = latent_model_input.to(prompt_embeds.dtype)
            audio_latent_model_input = torch.cat([audio_latents] * 2) if do_cfg else audio_latents
            audio_latent_model_input = audio_latent_model_input.to(prompt_embeds.dtype)

            timestep = t.expand(latent_model_input.shape[0])
            video_timestep = timestep.unsqueeze(-1) * (1 - conditioning_mask_full.squeeze(-1))

            t_audio = audio_timesteps[i]
            audio_timestep_scalar = t_audio.expand(latent_model_input.shape[0])
            audio_timestep = audio_timestep_scalar.unsqueeze(-1) * (1 - audio_conditioning_mask_full.squeeze(-1))

            noise_pred_video, noise_pred_audio = pipe.transformer(
                hidden_states=latent_model_input,
                audio_hidden_states=audio_latent_model_input,
                encoder_hidden_states=connector_prompt_embeds,
                audio_encoder_hidden_states=connector_audio_prompt_embeds,
                timestep=video_timestep,
                audio_timestep=audio_timestep,
                sigma=timestep,
                audio_sigma=audio_timestep_scalar,
                encoder_attention_mask=connector_attention_mask,
                audio_encoder_attention_mask=connector_attention_mask,
                num_frames=latent_num_frames,
                height=latent_height,
                width=latent_width,
                fps=fps,
                audio_num_frames=audio_num_frames,
                video_coords=video_coords,
                audio_coords=audio_coords,
                isolate_modalities=False,
                spatio_temporal_guidance_blocks=None,
                perturbation_mask=None,
                use_cross_timestep=True,  # A/B検証の結論(モジュールdocstring参照)
                return_dict=False,
            )
            noise_pred_video = noise_pred_video.float()
            noise_pred_audio = noise_pred_audio.float()

            if do_cfg:
                npv_uncond, npv_cond = noise_pred_video.chunk(2)
                npv_cond_x0 = pipe.convert_velocity_to_x0(latents, npv_cond, i, pipe.scheduler)
                npv_uncond_x0 = pipe.convert_velocity_to_x0(latents, npv_uncond, i, pipe.scheduler)
                video_cfg_delta = (guidance_scale - 1) * (npv_cond_x0 - npv_uncond_x0)

                npa_uncond, npa_cond = noise_pred_audio.chunk(2)
                npa_cond_x0 = pipe.convert_velocity_to_x0(audio_latents, npa_cond, i, audio_scheduler)
                npa_uncond_x0 = pipe.convert_velocity_to_x0(audio_latents, npa_uncond, i, audio_scheduler)
                audio_cfg_delta = (audio_guidance_scale - 1) * (npa_cond_x0 - npa_uncond_x0)

                noise_pred_video = npv_cond_x0 + video_cfg_delta
                noise_pred_audio = npa_cond_x0 + audio_cfg_delta
            else:
                noise_pred_video = pipe.convert_velocity_to_x0(latents, noise_pred_video, i, pipe.scheduler)
                noise_pred_audio = pipe.convert_velocity_to_x0(audio_latents, noise_pred_audio, i, audio_scheduler)

            bsz = noise_pred_video.size(0)
            denoised_video_cond = (
                noise_pred_video * (1 - conditioning_mask[:bsz]) + clean_latents[:bsz] * conditioning_mask[:bsz]
            ).to(noise_pred_video.dtype)
            noise_pred_video = pipe.convert_x0_to_velocity(latents, denoised_video_cond, i, pipe.scheduler)
            latents = pipe.scheduler.step(noise_pred_video, t, latents, return_dict=False)[0]

            abz = noise_pred_audio.size(0)
            denoised_audio_cond = (
                noise_pred_audio * (1 - audio_conditioning_mask[:abz])
                + clean_audio_packed[:abz] * audio_conditioning_mask[:abz]
            ).to(noise_pred_audio.dtype)
            noise_pred_audio = pipe.convert_x0_to_velocity(audio_latents, denoised_audio_cond, i, audio_scheduler)
            audio_latents = audio_scheduler.step(noise_pred_audio, t, audio_latents, return_dict=False)[0]

            progress_mod.update_step(i + 1, steps)

    gen_elapsed = time.time() - t0
    progress_mod.set_phase("decoding")

    # --- デコード前クリーンアップ: 不要になった大型テンソルを解放してVAE decodeの
    # VRAM余裕を作る。text_encoder/connectorsのCPU退避はしない(モジュールdocstring参照) ---
    compute_dtype = prompt_embeds.dtype
    del noise_pred_video, noise_pred_audio, latent_model_input, audio_latent_model_input
    del connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask
    del prompt_embeds, prompt_attention_mask
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- decode ---
    base_token_count = latent_num_frames * latent_height * latent_width
    latents = latents[:, :base_token_count]
    latents = pipe._unpack_latents(
        latents, latent_num_frames, latent_height, latent_width,
        pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size,
    )
    latents = latents.to(compute_dtype)
    latents = pipe._denormalize_latents(latents, pipe.vae.latents_mean, pipe.vae.latents_std, pipe.vae.config.scaling_factor)
    latents = latents.to(pipe.vae.dtype)

    # groupモード(48GB級)では VAE decode の一時ピークが収まらない(実測: 512x320・
    # 89フレームで conv3d が 5.87GB 要求 → OOM)ため、時間方向の tiled decode
    # (use_framewise_decoding、enable_tiling() では有効にならず別途フラグ設定が必要)で
    # ピークを分割する。probe 時代の「text_encoder 丸ごとCPU退避」の代替であり、
    # CLAUDE.md 33番の禁止パターン(ホストRAM退避)を使わずに decode の VRAM を賄う。
    # noneモード(96GB級)では従来どおり一括 decode(出力を変えないため)。
    tiled_decode = state.base_pipeline_state.get("offload_mode") == "group"
    if tiled_decode:
        prev_tiling = (pipe.vae.use_tiling, pipe.vae.use_framewise_decoding)
        pipe.vae.enable_tiling()
        pipe.vae.use_framewise_decoding = True
    try:
        with torch.no_grad():
            video = pipe.vae.decode(latents, None, return_dict=False)[0]
    finally:
        if tiled_decode:
            pipe.vae.use_tiling, pipe.vae.use_framewise_decoding = prev_tiling
    video_frames = pipe.video_processor.postprocess_video(video, output_type="pil")

    vocoder_audio = None
    if audio_out == "vocoder":
        audio_latents_denorm = pipe._denormalize_audio_latents(
            audio_latents, pipe.audio_vae.latents_mean, pipe.audio_vae.latents_std
        )
        latent_mel_bins = pipe.audio_mel_bins // pipe.audio_vae_mel_compression_ratio
        audio_latents_unpacked = pipe._unpack_audio_latents(
            audio_latents_denorm, audio_num_frames, num_mel_bins=latent_mel_bins
        )
        with torch.no_grad():
            audio_latents_unpacked = audio_latents_unpacked.to(pipe.audio_vae.dtype)
            generated_mel = pipe.audio_vae.decode(audio_latents_unpacked, return_dict=False)[0]
            vocoder_audio = pipe.vocoder(generated_mel)

    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = uuid.uuid4().hex[:8]
    video_path, video_name = _new_output_path("ltx2_ia2v", "mp4", ts=ts, job_id=job_id)
    export_to_video(video_frames[0], video_path, fps=fps)

    audio_url = None
    muxed_url = None
    has_audio = False
    if audio_out == "original":
        # 入力wavの原音をそのままmux(タスク要件1番)。動画長にトリム/パディングする。
        orig_audio_path, orig_audio_name = _new_output_path("ltx2_ia2v", "wav", ts=ts, job_id=job_id)
        import soundfile as sf

        wav_for_save = waveform_orig.transpose(0, 1).numpy() if waveform_orig.shape[0] <= 8 else waveform_orig.numpy()
        sf.write(orig_audio_path, wav_for_save, sr_orig)
        audio_url, muxed_url, has_audio = _save_original_audio_and_mux(
            orig_audio_path, video_path, num_frames, fps, ts, job_id
        )
        if audio_url is None:
            audio_url = f"/outputs/{orig_audio_name}"
    elif audio_out == "vocoder":
        audio_url, muxed_url, has_audio = _save_audio_and_mux(
            vocoder_audio, video_path, ts, job_id, mode="ltx2_ia2v", audio_out="on"
        )
    # audio_out == "none": 音声保存もmuxも行わない(無音mp4のまま)

    meta = {
        "mode": "ltx2_ia2v",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": fps,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "audio_guidance_scale": audio_guidance_scale,
        "seed": used_seed,
        "elapsed_s": elapsed,
        "gen_elapsed_s": gen_elapsed,
        "peak_vram_gb": peak_vram_gb,
        "video_url": muxed_url or f"/outputs/{video_name}",
        "video_only_url": f"/outputs/{video_name}",
        "audio_url": audio_url,
        "has_audio": has_audio,
        "audio_out": audio_out,
        "audio_duration_s": wav_duration_s,
    }
    meta.update(_pipeline_meta())
    return meta


# ============================================================================
# V2A(Video to Audio、IA2Vの鏡像。2026-07-20追加)
#
# 機構: run_ia2v() が「音声latentを入力wav由来のclean latentへ毎ステップx0固定(mask=1.0)
# し、映像側だけをdenoiseする」のに対し、V2Aはその逆で「映像latentを入力動画由来の
# clean latentへ毎ステップx0固定(mask=1.0)し、音声latentだけをdenoiseする」。
# ループの骨格(scheduler、use_cross_timestep=True、CFG分岐、torch.no_grad()、進捗
# コールバック、デコード前del+empty_cache())はrun_ia2v()を最大限流用する。
# diffusers本体は無改変(pipe.prepare_latents/pipe.vae.encode等の既存メソッドのみ使用)。
#
# 入力動画は cv2.VideoCapture でフレーム列として読み込み、要求解像度へ「長辺基準リサイズ+
# 中央クロップ」(LTX2ConditionPipeline.preprocess_conditions()と同じ規約、bilinear補間)
# した上で video VAE でエンコードする。フレーム数は8n+1へ切り詰める(末尾側を切る、
# 8n+1丸めの標準的な扱い)。音声latentの初期形状はLTX2Pipeline.__call__()のT2V実装
# (prepare_audio_latents、乱数初期化)と同一の計算式を使う。
#
# 出力: 生成した音声(vocoder出力)をwav化し、**元の入力動画(リサイズ前のオリジナル)**へ
# ffmpeg でmuxする(_mux_wav_to_video()を再利用)。映像側のVAE decodeは不要(出力映像は
# 入力そのもの)なため、run_ia2v()よりVRAM・時間とも軽い。
# ============================================================================


def _read_video_frames(video_path: str, max_num_frames: int):
    """cv2.VideoCapture で動画を読み込み、フレーム列(RGB, uint8, [F,H,W,C])・fps・
    元の全フレーム数を返す。要求解像度へのリサイズは呼び出し側(_encode_video_to_clean_latents())
    で行う(生フレームはオリジナル解像度のまま保持し、mux用の元動画パスとは別に扱う)。
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"動画を読み込めませんでした: {video_path!r}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = float(DEFAULT_FPS)
        frames = []
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            if len(frames) >= max_num_frames:
                # 上限に達したら打ち切る(タスク要件: 長い動画への保険)
                break
    finally:
        cap.release()

    if not frames:
        raise ValueError(f"動画からフレームを1枚も読み込めませんでした: {video_path!r}")

    total_frames_read = len(frames)
    # 8n+1 へ切り詰める(末尾を切る)。1フレームしか読めなかった場合は1のまま。
    if total_frames_read > 1:
        n = (total_frames_read - 1) // 8
        target = max(1, 8 * n + 1)
        frames = frames[:target]

    return frames, float(fps), total_frames_read


def _encode_video_to_clean_latents(frames: "list", height: int, width: int, pipe, device, dtype):
    """フレーム列(RGB, uint8)を「長辺基準リサイズ+中央クロップ」でtarget解像度へ揃え、
    video VAE でエンコードして正規化・パック済みのclean video latent(3D)を返す。

    LTX2ConditionPipeline.preprocess_conditions()(pipeline_ltx2_condition.py)と同じ
    前処理規約(F.interpolateのbilinear、アンチエイリアスなし。PIL resizeは使わない)を
    踏襲し、[0,255] -> [-1,1] へマップしてから pipe.vae.encode() する。
    """
    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import retrieve_latents

    arr = np.stack(frames)  # [F, H, W, C] uint8
    src_h, src_w = arr.shape[1], arr.shape[2]
    num_frames = arr.shape[0]

    pixels = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float32)
    pixels = pixels.permute(3, 0, 1, 2).unsqueeze(0).to(device)  # [1, C, F, H, W]

    scale = max(height / src_h, width / src_w)
    new_h = int(np.ceil(src_h * scale))
    new_w = int(np.ceil(src_w * scale))
    pixels = pixels.permute(0, 2, 1, 3, 4).reshape(num_frames, 3, src_h, src_w)
    pixels = torch.nn.functional.interpolate(pixels, size=(new_h, new_w), mode="bilinear", align_corners=False)
    top = (new_h - height) // 2
    left = (new_w - width) // 2
    pixels = pixels[:, :, top: top + height, left: left + width]
    pixels = pixels.reshape(1, num_frames, 3, height, width).permute(0, 2, 1, 3, 4)  # [1, C, F, H, W]

    video_tensor = (pixels / 127.5 - 1.0).to(dtype=pipe.vae.dtype, device=device)

    with torch.no_grad():
        posterior = pipe.vae.encode(video_tensor).latent_dist
        clean_latents_5d = posterior.mode()

    clean_latents_5d = clean_latents_5d.to(device=device, dtype=dtype)
    clean_latents_norm = pipe._normalize_latents(
        clean_latents_5d, pipe.vae.latents_mean, pipe.vae.latents_std, pipe.vae.config.scaling_factor
    )
    clean_latents_packed = pipe._pack_latents(
        clean_latents_norm, pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size
    )
    return clean_latents_packed


def run_v2a(req: dict, video_path: str) -> dict:
    """V2A(Video to Audio): 既存動画に合った音声を生成して付与する(IA2Vの鏡像)。

    映像latentを入力動画由来のclean latentへ毎ステップx0固定(conditioning_mask=1.0)
    し、音声latentだけをdenoiseする。diffusers本体(LTX2ConditionPipeline)は改変せず、
    そのdenoiseループをコピー・改修したカスタムループをここに実装する(run_ia2v()と対)。
    """
    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import calculate_shift, retrieve_timesteps

    pipe = pipeline_mod.get_flf_pipeline()  # LTX2ConditionPipeline、base components 参照共有
    generator, used_seed = make_generator(req.get("seed", -1))

    device = pipe._execution_device
    dtype = torch.float32

    width = int(req.get("width") or DEFAULT_WIDTH)
    height = int(req.get("height") or DEFAULT_HEIGHT)
    steps = int(req.get("steps") or DEFAULT_STEPS)
    guidance_scale = float(req.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
    prompt = req.get("prompt") or DEFAULT_V2A_PROMPT
    negative_prompt = req.get("negative_prompt") or DEFAULT_V2A_NEGATIVE_PROMPT

    # num_frames(タスク要件): 0(未指定)なら動画全長を8n+1へ丸める(上限
    # DEFAULT_V2A_MAX_NUM_FRAMES まで自動で切り詰め)。明示指定がその上限を超える場合は
    # 明確な ValueError にする(長い動画への保険)。
    requested_num_frames = req.get("num_frames")
    if requested_num_frames:
        requested_num_frames = int(requested_num_frames)
        if requested_num_frames > DEFAULT_V2A_MAX_NUM_FRAMES:
            raise ValueError(
                f"num_frames は上限 {DEFAULT_V2A_MAX_NUM_FRAMES} を超えられません(v2a): "
                f"{requested_num_frames!r}"
            )
        read_limit = requested_num_frames
    else:
        read_limit = DEFAULT_V2A_MAX_NUM_FRAMES

    _reset_vram_stats()
    progress_mod.start_loading("ltx2_v2a")

    # --- 動画 -> フレーム列(元fps・元フレーム数を取得) ---
    frames, video_fps, total_frames_read = _read_video_frames(video_path, read_limit)
    num_frames = len(frames)
    video_duration_s = num_frames / video_fps

    # groupモード(48GB級)の VAE encode 対策(CLAUDE.md 39番・タスク指示):
    # autoencoder_kl_ltx2.py の _encode() は use_framewise_decoding フラグで
    # _temporal_tiled_encode に切り替わる実装(decodeと同じフラグを共有)。
    tiled_vae = state.base_pipeline_state.get("offload_mode") == "group"
    if tiled_vae:
        prev_tiling = (pipe.vae.use_tiling, pipe.vae.use_framewise_decoding)
        pipe.vae.enable_tiling()
        pipe.vae.use_framewise_decoding = True
    try:
        clean_video_packed = _encode_video_to_clean_latents(frames, height, width, pipe, device, dtype)
    finally:
        if tiled_vae:
            pipe.vae.use_tiling, pipe.vae.use_framewise_decoding = prev_tiling

    do_cfg = guidance_scale > 1.0

    t0 = time.time()

    # encode_prompt を torch.no_grad() なしで呼ぶと余分な一時VRAMを消費する
    # (run_ia2v()と同じ理由、48GB相当のgroupモードで実機OOMを誘発した経緯があるため
    # 以降すべてを1つのtorch.no_grad()で包む)。
    with torch.no_grad():
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            device=device,
        )
        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = pipe.connectors(
            prompt_embeds, prompt_attention_mask, padding_side=pipe.tokenizer_padding_side
        )

        # --- video latents: 全域clean固定(hard lock、conditioning_mask=1.0) ---
        latent_num_frames = (num_frames - 1) // pipe.vae_temporal_compression_ratio + 1
        latent_height = height // pipe.vae_spatial_compression_ratio
        latent_width = width // pipe.vae_spatial_compression_ratio

        clean_latents = clean_video_packed.clone()
        latents = clean_video_packed.clone()
        conditioning_mask = torch.full(
            (latents.shape[0], latents.shape[1], 1), 1.0, device=device, dtype=dtype,  # 常にclean(=入力動画)固定
        )
        conditioning_mask_full = torch.cat([conditioning_mask, conditioning_mask]) if do_cfg else conditioning_mask

        # --- audio latents: 乱数初期化(T2Vのprepare_audio_latentsと同一計算式) ---
        duration_s = num_frames / video_fps
        audio_latents_per_second = (
            pipe.audio_sampling_rate / pipe.audio_hop_length / float(pipe.audio_vae_temporal_compression_ratio)
        )
        audio_num_frames = round(duration_s * audio_latents_per_second)
        latent_mel_bins = pipe.audio_mel_bins // pipe.audio_vae_mel_compression_ratio
        num_channels_latents_audio = pipe.audio_vae.config.latent_channels

        audio_shape = (1, num_channels_latents_audio, audio_num_frames, latent_mel_bins)
        from diffusers.utils.torch_utils import randn_tensor

        audio_latents = randn_tensor(audio_shape, generator=generator, device=device, dtype=dtype)
        audio_latents = pipe._pack_audio_latents(audio_latents)
        audio_conditioning_mask = torch.zeros(
            (audio_latents.shape[0], audio_latents.shape[1], 1), device=device, dtype=dtype,  # 常にnoise(=denoise対象)
        )
        audio_conditioning_mask_full = (
            torch.cat([audio_conditioning_mask, audio_conditioning_mask]) if do_cfg else audio_conditioning_mask
        )

        # --- timesteps / scheduler ---
        sigmas = np.linspace(1.0, 1 / steps, steps)
        mu = calculate_shift(
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_image_seq_len", 1024),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.95),
            pipe.scheduler.config.get("max_shift", 2.05),
        )
        audio_scheduler = copy.deepcopy(pipe.scheduler)
        audio_timesteps, _ = retrieve_timesteps(audio_scheduler, steps, device, None, sigmas=sigmas, mu=mu)
        timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, steps, device, None, sigmas=sigmas, mu=mu)

        video_coords = pipe.transformer.rope.prepare_video_coords(
            latents.shape[0], latent_num_frames, latent_height, latent_width, latents.device, fps=video_fps
        )
        audio_coords = pipe.transformer.audio_rope.prepare_audio_coords(
            audio_latents.shape[0], audio_num_frames, audio_latents.device
        )
        if do_cfg:
            video_coords = video_coords.repeat((2,) + (1,) * (video_coords.ndim - 1))
            audio_coords = audio_coords.repeat((2,) + (1,) * (audio_coords.ndim - 1))

        progress_mod.start_generating("ltx2_v2a", steps)

        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = latent_model_input.to(prompt_embeds.dtype)
            audio_latent_model_input = torch.cat([audio_latents] * 2) if do_cfg else audio_latents
            audio_latent_model_input = audio_latent_model_input.to(prompt_embeds.dtype)

            timestep = t.expand(latent_model_input.shape[0])
            video_timestep = timestep.unsqueeze(-1) * (1 - conditioning_mask_full.squeeze(-1))

            t_audio = audio_timesteps[i]
            audio_timestep_scalar = t_audio.expand(latent_model_input.shape[0])
            audio_timestep = audio_timestep_scalar.unsqueeze(-1) * (1 - audio_conditioning_mask_full.squeeze(-1))

            noise_pred_video, noise_pred_audio = pipe.transformer(
                hidden_states=latent_model_input,
                audio_hidden_states=audio_latent_model_input,
                encoder_hidden_states=connector_prompt_embeds,
                audio_encoder_hidden_states=connector_audio_prompt_embeds,
                timestep=video_timestep,
                audio_timestep=audio_timestep,
                sigma=timestep,
                audio_sigma=audio_timestep_scalar,
                encoder_attention_mask=connector_attention_mask,
                audio_encoder_attention_mask=connector_attention_mask,
                num_frames=latent_num_frames,
                height=latent_height,
                width=latent_width,
                fps=video_fps,
                audio_num_frames=audio_num_frames,
                video_coords=video_coords,
                audio_coords=audio_coords,
                isolate_modalities=False,
                spatio_temporal_guidance_blocks=None,
                perturbation_mask=None,
                use_cross_timestep=True,  # run_ia2v()と同じ設定(モジュールdocstring参照)
                return_dict=False,
            )
            noise_pred_video = noise_pred_video.float()
            noise_pred_audio = noise_pred_audio.float()

            if do_cfg:
                npv_uncond, npv_cond = noise_pred_video.chunk(2)
                npv_cond_x0 = pipe.convert_velocity_to_x0(latents, npv_cond, i, pipe.scheduler)
                npv_uncond_x0 = pipe.convert_velocity_to_x0(latents, npv_uncond, i, pipe.scheduler)
                video_cfg_delta = (guidance_scale - 1) * (npv_cond_x0 - npv_uncond_x0)

                npa_uncond, npa_cond = noise_pred_audio.chunk(2)
                npa_cond_x0 = pipe.convert_velocity_to_x0(audio_latents, npa_cond, i, audio_scheduler)
                npa_uncond_x0 = pipe.convert_velocity_to_x0(audio_latents, npa_uncond, i, audio_scheduler)
                audio_cfg_delta = (guidance_scale - 1) * (npa_cond_x0 - npa_uncond_x0)

                noise_pred_video = npv_cond_x0 + video_cfg_delta
                noise_pred_audio = npa_cond_x0 + audio_cfg_delta
            else:
                noise_pred_video = pipe.convert_velocity_to_x0(latents, noise_pred_video, i, pipe.scheduler)
                noise_pred_audio = pipe.convert_velocity_to_x0(audio_latents, noise_pred_audio, i, audio_scheduler)

            bsz = noise_pred_video.size(0)
            denoised_video_cond = (
                noise_pred_video * (1 - conditioning_mask[:bsz]) + clean_latents[:bsz] * conditioning_mask[:bsz]
            ).to(noise_pred_video.dtype)
            noise_pred_video = pipe.convert_x0_to_velocity(latents, denoised_video_cond, i, pipe.scheduler)
            latents = pipe.scheduler.step(noise_pred_video, t, latents, return_dict=False)[0]

            # audio側は conditioning_mask が常に0(全域denoise対象)のため、
            # video側と違いclean値とのブレンドは不要(noise_predをそのままx0として使う)。
            noise_pred_audio = pipe.convert_x0_to_velocity(audio_latents, noise_pred_audio, i, audio_scheduler)
            audio_latents = audio_scheduler.step(noise_pred_audio, t, audio_latents, return_dict=False)[0]

            progress_mod.update_step(i + 1, steps)

    gen_elapsed = time.time() - t0
    progress_mod.set_phase("decoding")

    # --- デコード前クリーンアップ(run_ia2v()と同じ、text_encoder/connectorsのCPU退避はしない) ---
    compute_dtype = prompt_embeds.dtype
    del noise_pred_video, noise_pred_audio, latent_model_input, audio_latent_model_input
    del connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask
    del prompt_embeds, prompt_attention_mask, latents, clean_latents, clean_video_packed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- audio decode(映像側はdecode不要、出力映像は入力そのもの) ---
    audio_latents_denorm = pipe._denormalize_audio_latents(
        audio_latents, pipe.audio_vae.latents_mean, pipe.audio_vae.latents_std
    )
    latent_mel_bins = pipe.audio_mel_bins // pipe.audio_vae_mel_compression_ratio
    audio_latents_unpacked = pipe._unpack_audio_latents(
        audio_latents_denorm, audio_num_frames, num_mel_bins=latent_mel_bins
    )
    with torch.no_grad():
        audio_latents_unpacked = audio_latents_unpacked.to(pipe.audio_vae.dtype)
        generated_mel = pipe.audio_vae.decode(audio_latents_unpacked, return_dict=False)[0]
        vocoder_audio = pipe.vocoder(generated_mel)

    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = uuid.uuid4().hex[:8]

    # V2Aは映像を変更しない(出力映像は入力そのもの)ため、outputs/配下へコピーしてから
    # 音声をmuxする(video_path はアップロード直後の一時ファイルパスであり、
    # /outputs/ 配下ではないため URL として返せない。呼び出し側は本関数からの
    # 復帰後に一時ファイルを削除する規約のため、mux前にコピーが必要)。
    src_ext = os.path.splitext(video_path)[1] or ".mp4"
    video_only_path, video_only_name = _new_output_path("ltx2_v2a", src_ext.lstrip("."), ts=ts, job_id=job_id)
    shutil.copyfile(video_path, video_only_path)

    audio_url, muxed_url, has_audio = _save_audio_and_mux(
        vocoder_audio, video_only_path, ts, job_id, mode="ltx2_v2a", audio_out="on"
    )

    meta = {
        "mode": "ltx2_v2a",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": video_fps,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": used_seed,
        "elapsed_s": elapsed,
        "gen_elapsed_s": gen_elapsed,
        "peak_vram_gb": peak_vram_gb,
        "video_url": muxed_url or f"/outputs/{video_only_name}",
        "video_only_url": f"/outputs/{video_only_name}",
        "audio_url": audio_url,
        "has_audio": has_audio,
        "video_duration_s": video_duration_s,
        "source_total_frames": total_frames_read,
    }
    meta.update(_pipeline_meta())
    return meta


# ============================================================================
# IC-LoRA(In-Context LoRA、MergeGreen 動画編集モード、2026-07-20追加)
#
# 機構: HF repo siraxe/MergeGreen_IC-lora_ltx2.3 の LoRA アダプタは、中央フレームを
# 緑 RGB(0,191,0) でマスクした参照動画を LTX2InContextPipeline.reference_conditions へ
# 渡すと、緑マスク領域を「プロンプトで指示した変化」に沿って生成し直す(前後の非マスク
# 領域は参照動画の内容をおおむね維持する)動画編集アダプタ。ComfyUIワークフロー
# (MergeGreen_IC-lora_ltx2.3.json)の実際のノード構成(LTXICLoRALoaderModelOnly
# strength=0.9、LTXAddVideoICLoRAGuide strength=1、LTXVConditioning frame_rate=25)を
# 実機で確認し、そのまま踏襲した:
#   - LoRA adapter weight: 0.9(README「0.9 blends better」とも整合)
#   - reference_conditions の strength: 1.0(参照動画にそのまま従う、conditioning_
#     attention_strength は既定の1.0のまま=マスキングなし。マスクは参照動画の
#     "ピクセル" に緑色として焼き込まれているため、attention側のマスクは不要)
#   - frame_rate: 25(FLF/keyframesと同じ DEFAULT_FLF_FRAME_RATE を流用)
#
# diffusers本体(LTX2InContextPipeline.__call__())は無改変で使う(IA2V/V2Aと違い
# 独自denoiseループは不要、標準の reference_conditions 機構だけで実現できる)。
# ============================================================================


def _apply_green_mask(
    frames: "list", mask_start: int, mask_end: int, mask_color: "tuple[int, int, int]" = DEFAULT_ICLORA_MASK_COLOR
) -> "list":
    """frames(RGB, uint8, list of HxWxC ndarray)の [mask_start, mask_end) 範囲のフレームを
    指定色で塗りつぶした新しいリストを返す(元のリストは変更しない)。

    HFリポジトリ README の「set middle frames color to RGB 0,191,0 (75% green fill)」に
    従い、フレーム全体を単色で塗りつぶす(75%はComfyUI側ノードの合成透過率の話であり、
    塗りつぶし色そのものはRGB(0,191,0)のベタ塗りでよいことをワークフローの
    VideoPrepAB ノード('0,191,0', ...) の指定からも確認済み)。
    """
    masked = list(frames)  # shallow copy、対象範囲だけ差し替える
    color = np.array(mask_color, dtype=np.uint8)
    for i in range(max(0, mask_start), min(mask_end, len(masked))):
        h, w = masked[i].shape[:2]
        masked[i] = np.broadcast_to(color, (h, w, 3)).copy()
    return masked


def run_iclora(req: dict, video_path: str) -> dict:
    """IC-LoRA(MergeGreen): 参照動画の中央部を緑マスクし、プロンプトで指示した変化に
    沿って動画を生成し直す(動画編集モード)。

    mask_mode:
      "middle"(既定): サーバ側で中央フレームを緑マスクする。mask_start/mask_end
        (フレームインデックス、両端未指定なら動画全体の中央1/3)で範囲を明示できる。
      "prefilled": ユーザーが既に緑マスク済みの動画をアップロードした前提で、
        サーバ側でのマスク適用は行わない(そのまま reference_conditions へ渡す)。
    """
    from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import LTX2ReferenceCondition

    pipe = pipeline_mod.get_iclora_pipeline()
    generator, used_seed = make_generator(req.get("seed", -1))

    width = int(req.get("width") or DEFAULT_WIDTH)
    height = int(req.get("height") or DEFAULT_HEIGHT)
    fps = float(req.get("fps") or DEFAULT_FLF_FRAME_RATE)
    steps = int(req.get("steps") or DEFAULT_STEPS)
    guidance_scale = float(req.get("guidance_scale", DEFAULT_GUIDANCE_SCALE))
    negative_prompt = req.get("negative_prompt") or DEFAULT_ICLORA_NEGATIVE_PROMPT
    ref_strength = float(req.get("ref_strength") or DEFAULT_ICLORA_REF_STRENGTH)
    mask_mode = (req.get("mask_mode") or "middle").strip().lower()
    if mask_mode not in ("middle", "prefilled"):
        raise ValueError(f"mask_mode は middle/prefilled のいずれかです(iclora): {mask_mode!r}")
    if not req.get("prompt"):
        raise ValueError("iclora には prompt(何を変えるかの指示)が必須です")

    requested_num_frames = req.get("num_frames")
    if requested_num_frames:
        requested_num_frames = int(requested_num_frames)
        if requested_num_frames > DEFAULT_V2A_MAX_NUM_FRAMES:
            raise ValueError(
                f"num_frames は上限 {DEFAULT_V2A_MAX_NUM_FRAMES} を超えられません(iclora): "
                f"{requested_num_frames!r}"
            )
        read_limit = requested_num_frames
    else:
        read_limit = DEFAULT_V2A_MAX_NUM_FRAMES

    _reset_vram_stats()
    progress_mod.start_loading("ltx2_iclora")

    frames, video_fps, total_frames_read = _read_video_frames(video_path, read_limit)
    num_frames = len(frames)

    if mask_mode == "middle":
        default_start = num_frames // 3
        default_end = num_frames - num_frames // 3
        mask_start = int(req.get("mask_start")) if req.get("mask_start") is not None else default_start
        mask_end = int(req.get("mask_end")) if req.get("mask_end") is not None else default_end
        mask_start = max(0, min(mask_start, num_frames))
        mask_end = max(mask_start, min(mask_end, num_frames))
        ref_frames = _apply_green_mask(frames, mask_start, mask_end)
    else:
        mask_start, mask_end = None, None
        ref_frames = frames

    reference_conditions = [LTX2ReferenceCondition(frames=ref_frames, strength=ref_strength)]

    # groupモード(48GB級)では VAE decode/encode の一時ピークが収まらないため、
    # run_keyframes()/run_v2a() と同じパターン(時間方向tiled decode)で pipe(...)
    # 呼び出し全体を包む(reference_conditions のVAEエンコードもこのフラグの影響を
    # 受ける、CLAUDE.md 39番・41番参照)。
    tiled_decode = state.base_pipeline_state.get("offload_mode") == "group"
    if tiled_decode:
        prev_tiling = (pipe.vae.use_tiling, pipe.vae.use_framewise_decoding)
        pipe.vae.enable_tiling()
        pipe.vae.use_framewise_decoding = True

    t0 = time.time()
    try:
        with torch.no_grad():
            result = pipe(
                prompt=req["prompt"],
                negative_prompt=negative_prompt,
                reference_conditions=reference_conditions,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=fps,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                callback_on_step_end=_progress_callback("ltx2_iclora", steps),
            )
    finally:
        if tiled_decode:
            pipe.vae.use_tiling, pipe.vae.use_framewise_decoding = prev_tiling
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    from diffusers.utils import export_to_video

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = uuid.uuid4().hex[:8]
    video_frames = result.frames[0]
    video_path_out, video_name = _new_output_path("ltx2_iclora", "mp4", ts=ts, job_id=job_id)
    export_to_video(video_frames, video_path_out, fps=fps)

    audio = getattr(result, "audio", None)
    audio_url, muxed_url, has_audio = _save_audio_and_mux(
        audio, video_path_out, ts, job_id, mode="ltx2_iclora", audio_out="on"
    )

    meta = {
        "mode": "ltx2_iclora",
        "prompt": req["prompt"],
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": fps,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": used_seed,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "video_url": muxed_url or f"/outputs/{video_name}",
        "video_only_url": f"/outputs/{video_name}",
        "audio_url": audio_url,
        "has_audio": has_audio,
        "mask_mode": mask_mode,
        "mask_start": mask_start,
        "mask_end": mask_end,
        "ref_strength": ref_strength,
        "iclora_strength": state.get_runtime_config().iclora_strength,
        "source_total_frames": total_frames_read,
    }
    meta.update(_pipeline_meta())
    return meta
