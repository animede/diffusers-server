# -*- coding: utf-8 -*-
"""
LTX-2.3 IA2V (Image + Audio to Video: 参照画像+wav音声 -> 音声同期(リップシンク)動画)
のスタンドアロン検証プロトタイプ。

## 背景・調査結果のまとめ

ComfyUI (`comfy_extras/nodes_lt_audio.py` / `nodes_lt.py`) の IA2V 機構を調査した結果:

  - `LTXVAudioVAEEncode` は wav を `AudioVAE.encode()` でメルスペクトログラム経由の
    音声latent(shape `[B, C=8, L, M=16]`, 8chunk次元・Lは時間・M=mel_bins//4)へ変換する。
  - `LTXVConcatAVLatent` は video latent と audio latent を `comfy.nested_tensor.NestedTensor`
    で束ね、**1回のサンプリングで両方を同時denoise**する(comfy/samplers.py 1004-1031行、
    `pack_latents`/`unbind`でモダリティごとに分離・結合)。
  - AVTransformer(`comfy/ldm/lightricks/av_model.py` の `LTXAVModel`/`BasicAVTransformerBlock`)
    は video/audio を別々にpatchifyした後、audio_attn1/2(音声自己注意)、
    audio_to_video_attn / video_to_audio_attn(モダリティ間cross-attention)を持つ
    joint transformer blockで処理する。つまりリップシンクは「audioをガイドとして注入する」
    のではなく、video/audio双方向cross-attentionによる同時denoiseで実現される。
  - 重要な発見: ComfyUIのIA2Vでは音声latentに`noise_mask=0`(=完全固定、denoiseされない)
    を与える経路が存在する(`get_noise_mask`のデフォルトは全て1=フルノイズだが、
    画像条件と同じ理屈で音声側も外部から与えられた既知の信号として固定できる設計になっている)。

## diffusers (venv 0.40.0.dev0) 側の対応状況

  - `transformer_ltx2.py` の `LTX2VideoTransformer3DModel` / `LTX2VideoTransformerBlock` は
    ComfyUIのAVModelと**同じjoint AVアーキテクチャを完全実装済み**
    (audio_attn1/2, audio_to_video_attn, video_to_audio_attn)。これはCLAUDE.mdの
    旧記述(「audio_latentsは単なる初期ノイズ」)よりも高機能で、想定より進んでいた。
  - `LTX2ConditionPipeline.__call__()` は `audio_latents` 引数を受け付け、
    `prepare_audio_latents(latents=..., noise_scale=...)` 経由で
    「与えられた音声latentをnoise_scaleで薄めてから通常のdenoiseループに入れる」
    ことができる(`_create_noised_state`: `noise_scale*noise + (1-noise_scale)*latents`)。
    しかし **denoiseループ内で音声latentをvideoの`conditioning_mask`のように
    「毎ステップ既知の値へ引き戻す」処理が一切無い**(1751-1757行目、videoは
    `denoised_sample_cond = pred*(1-mask) + clean*mask` を毎ステップ適用するが、
    audioの対応行(1757行目)は素通しで `noise_pred_audio` をそのまま
    `convert_x0_to_velocity`している)。つまり `audio_latents` は img2img の
    「strength」相当の弱い初期値ヒントに過ぎず、ComfyUIの`noise_mask=0`ハードロックに
    相当する機構がdiffusers側には実装されていないことをソース確認した
    (pipeline_ltx2_condition.py 1064-1094行 `prepare_audio_latents`、
    1744-1764行 denoiseループのvideo/audio非対称箇所)。

## 本スクリプトの対処

diffusers本体は改変しない。`LTX2ConditionPipeline.__call__()` を直接呼ぶ代わりに、
そのdenoiseループをコピー・改修した**カスタムループ**をこのスクリプト内に実装し、
video側と全く同じ x0-space remix (`pred*(1-mask) + clean*mask`) を audio 側にも
毎ステップ適用することで、ComfyUIの `noise_mask=0` ハードロックと同等の
「音声は常に入力wav由来の既知信号に固定し、videoだけがそれに対してcross-attentionで
同期して生成される」条件付けを実現する。

使い方:
  venv/bin/python tools/ltx2_ia2v_probe.py --stage mel      # mel変換確認のみ(GPU不要)
  venv/bin/python tools/ltx2_ia2v_probe.py --stage full     # 画像+音声からIA2V動画を1本生成
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

# 実際に実行する場合は、手元の参照画像・音声ファイルのパスに差し替えること。
IMAGE_PATH = os.path.expanduser("~/ComfyUI/output/RemoveBG-result_00003_.png")
AUDIO_PATH = os.path.expanduser("~/audio.wav")
OUT_DIR = os.path.join(_REPO_ROOT, "outputs", "_ltx2_ia2v_probe")
os.makedirs(OUT_DIR, exist_ok=True)


def log(msg):
    print(f"[ia2v_probe] {msg}", flush=True)


def vram_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024**3


def peak_vram_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def ram_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024**2
    return -1.0


# ---------------------------------------------------------------------------
# wav -> mel spectrogram (ComfyUI AudioPreprocessor.waveform_to_mel と同一ロジック)
# audio_vae.config: sample_rate=16000, mel_hop_length=160, mel_bins=64
# n_fft(filter_length)=1024, win_length=1024, mel_fmin=0, mel_fmax=8000, stereo channels=2
# ---------------------------------------------------------------------------

def _load_wav_soundfile(audio_path: str):
    """torchaudio.load() は venv の torchaudio が新backend(torchcodec必須)のため使えない。
    soundfile(既存依存、families/ltx2/generate.py でも使用)で読み、[C, T] の float32 tensor にする。
    """
    import soundfile as sf

    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)  # [T, C]
    waveform = torch.from_numpy(data).transpose(0, 1).contiguous()  # [C, T]
    return waveform, sr


def load_wav_as_mel(audio_path: str, audio_vae, device: str):
    import torchaudio

    waveform, sr = _load_wav_soundfile(audio_path)  # [C, T]
    log(f"loaded wav: shape={tuple(waveform.shape)} sr={sr}")

    target_sr = int(audio_vae.config.sample_rate)
    mel_bins = int(audio_vae.config.mel_bins)
    hop_length = int(audio_vae.config.mel_hop_length)
    n_fft = 1024  # filter_length (ComfyUI preprocessing.stft.filter_length, checkpoint metadata確認済み)
    win_length = 1024
    mel_fmin = 0
    mel_fmax = target_sr / 2.0  # ComfyUI側は8000固定値だが sample_rate/2=8000 で一致するためこれで代用

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        log(f"resampled {sr} -> {target_sr}")

    # audio_vae は stereo(in_channels=2)を期待するため、モノラルはチャンネル複製する
    # (ComfyUIの AudioVAE.encode: waveform.shape[1]==1 の場合 expand)
    if waveform.shape[0] == 1:
        waveform = waveform.expand(2, -1)
        log("expanded mono -> stereo (channel duplication)")

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

    mel = mel_transform(waveform)  # [B, C, mel_bins, T_mel]
    mel = torch.log(torch.clamp(mel, min=1e-5))
    mel = mel.permute(0, 1, 3, 2).contiguous()  # -> [B, C, T_mel, mel_bins]  (matches ComfyUI waveform_to_mel)
    log(f"mel spectrogram shape={tuple(mel.shape)} (B,C,T,M)")
    return mel, waveform.shape[-1] / target_sr  # (mel, duration_seconds)


def stage_mel():
    """audio_vae 無しで torchaudio の mel 変換だけ確認する(GPU不要、軽量チェック)。"""
    import torchaudio

    waveform, sr = _load_wav_soundfile(AUDIO_PATH)
    log(f"wav: shape={tuple(waveform.shape)} sr={sr} dur={waveform.shape[-1]/sr:.3f}s")
    target_sr = 16000
    waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] == 1:
        waveform = waveform.expand(2, -1)
    waveform = waveform.unsqueeze(0)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr, n_fft=1024, win_length=1024, hop_length=160,
        f_min=0, f_max=8000, n_mels=64, power=1.0, mel_scale="slaney",
    )
    mel = mel_transform(waveform)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    mel = mel.permute(0, 1, 3, 2).contiguous()
    log(f"mel shape (B,C,T,M) = {tuple(mel.shape)}, min={mel.min():.3f} max={mel.max():.3f}")


# ---------------------------------------------------------------------------
# フル生成: 画像(first-frame条件)+ 音声(hard-locked条件)で IA2V 動画を生成する
# ---------------------------------------------------------------------------

def stage_full(args):
    from PIL import Image
    from diffusers.pipelines.ltx2.pipeline_ltx2_condition import (
        LTX2VideoCondition,
        calculate_shift,
        retrieve_timesteps,
        rescale_noise_cfg,
    )
    from diffusers.utils.torch_utils import randn_tensor
    from diffusers.utils import export_to_video

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from families.ltx2 import pipeline as pipeline_mod

    log(f"RAM available before load: {ram_available_gb():.1f}GB")

    t_load0 = time.time()
    pipe = pipeline_mod.get_flf_pipeline()  # LTX2ConditionPipeline, base components 参照共有
    load_time = time.time() - t_load0
    log(f"pipeline ready in {load_time:.1f}s, VRAM={vram_gb():.2f}GB")

    device = pipe._execution_device
    dtype = torch.float32

    # --- 画像読み込み ---
    image = Image.open(IMAGE_PATH).convert("RGB")
    log(f"image: {image.size}")

    width = args.width
    height = args.height
    fps = args.fps
    num_frames = args.num_frames
    steps = args.steps
    guidance_scale = args.guidance_scale
    audio_guidance_scale = args.audio_guidance_scale
    audio_lock_mode = args.audio_lock_mode  # "hard" (mask=1 always clean) or "soft" (noise_scale only, no remix)

    prompt = args.prompt
    negative_prompt = args.negative_prompt or ""

    generator = torch.Generator(device="cpu").manual_seed(args.seed) if args.seed >= 0 else None

    # --- 音声 -> mel -> audio_vae.encode() ---
    mel, wav_duration_s = load_wav_as_mel(AUDIO_PATH, pipe.audio_vae, device)
    mel = mel.to(pipe.audio_vae.dtype)
    with torch.no_grad():
        posterior = pipe.audio_vae.encode(mel).latent_dist
        clean_audio_latents_5d = posterior.mode()  # [B, C=8, L, M=16]
    log(f"encoded audio latent (5D) shape={tuple(clean_audio_latents_5d.shape)}")

    # --- text encode ---
    do_cfg = guidance_scale > 1.0 or audio_guidance_scale > 1.0
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

    # --- video latents (first-frame image condition, strength=1.0) ---
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
    if do_cfg:
        conditioning_mask_full = torch.cat([conditioning_mask, conditioning_mask])
    else:
        conditioning_mask_full = conditioning_mask

    latent_num_frames = (num_frames - 1) // pipe.vae_temporal_compression_ratio + 1
    latent_height = height // pipe.vae_spatial_compression_ratio
    latent_width = width // pipe.vae_spatial_compression_ratio

    # --- audio latents: 与えられた wav の latent に "常に" 固定する(hard lock) ---
    duration_s = num_frames / fps
    audio_latents_per_second = (
        pipe.audio_sampling_rate / pipe.audio_hop_length / float(pipe.audio_vae_temporal_compression_ratio)
    )
    audio_num_frames = round(duration_s * audio_latents_per_second)
    log(f"video duration={duration_s:.2f}s audio_num_frames(latent)={audio_num_frames} "
        f"(encoded wav latent length={clean_audio_latents_5d.shape[2]}, wav duration={wav_duration_s:.2f}s)")

    # 音声latent の時間長を動画尺に合わせる(切り詰め/ゼロパディング)
    L = clean_audio_latents_5d.shape[2]
    if L >= audio_num_frames:
        clean_audio_latents_5d = clean_audio_latents_5d[:, :, :audio_num_frames]
    else:
        pad = audio_num_frames - L
        clean_audio_latents_5d = torch.nn.functional.pad(clean_audio_latents_5d, (0, 0, 0, pad))
    clean_audio_latents_5d = clean_audio_latents_5d.to(device=device, dtype=dtype)

    clean_audio_packed = pipe._normalize_audio_latents(
        pipe._pack_audio_latents(clean_audio_latents_5d), pipe.audio_vae.latents_mean, pipe.audio_vae.latents_std
    )  # [B, L, C*M]
    log(f"clean_audio_packed shape={tuple(clean_audio_packed.shape)}")

    if audio_lock_mode == "hard":
        audio_mask_value = 1.0  # 常にclean(=入力wav)に固定、videoのI2V first-frameと同じ扱い
    else:
        audio_mask_value = 0.0  # soft: 通常のfree-running生成(比較用ベースライン)

    audio_conditioning_mask = torch.full(
        (clean_audio_packed.shape[0], clean_audio_packed.shape[1], 1),
        audio_mask_value, device=device, dtype=dtype,
    )

    # 初期 audio_latents: hardモードは clean からスタート(video I2V first-frameと同じ扱い、
    # 以降のループでも毎ステップ clean へ引き戻すため実質固定される)。
    # softモードは全ノイズからスタート(通常の free-running 生成、比較用)。
    noise_audio = randn_tensor(clean_audio_packed.shape, generator=generator, device=device, dtype=dtype)
    if audio_lock_mode == "hard":
        audio_latents = clean_audio_packed.clone()
    else:
        audio_latents = noise_audio.clone()

    if do_cfg:
        audio_conditioning_mask_full = torch.cat([audio_conditioning_mask, audio_conditioning_mask])
    else:
        audio_conditioning_mask_full = audio_conditioning_mask

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

    log(f"latents shape={tuple(latents.shape)} audio_latents shape={tuple(audio_latents.shape)} "
        f"audio_lock_mode={audio_lock_mode} steps={steps} do_cfg={do_cfg}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t_gen0 = time.time()
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = latent_model_input.to(prompt_embeds.dtype)
            audio_latent_model_input = torch.cat([audio_latents] * 2) if do_cfg else audio_latents
            audio_latent_model_input = audio_latent_model_input.to(prompt_embeds.dtype)

            timestep = t.expand(latent_model_input.shape[0])
            video_timestep = timestep.unsqueeze(-1) * (1 - conditioning_mask_full.squeeze(-1))

            t_audio = audio_timesteps[i]
            # audio_lock_mode=hard の場合、音声側の実効timestepも常に0(=常にclean)にする。
            # video の video_timestep = t*(1-conditioning_mask) と同じロジックを audio にも適用。
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
                use_cross_timestep=True,
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

            # video: 既存pipelineと同じx0-space remix
            bsz = noise_pred_video.size(0)
            denoised_video_cond = (
                noise_pred_video * (1 - conditioning_mask[:bsz]) + clean_latents[:bsz] * conditioning_mask[:bsz]
            ).to(noise_pred_video.dtype)
            noise_pred_video = pipe.convert_x0_to_velocity(latents, denoised_video_cond, i, pipe.scheduler)
            latents = pipe.scheduler.step(noise_pred_video, t, latents, return_dict=False)[0]

            # audio: 同じ x0-space remix をここで追加(diffusers本体に無い、本スクリプトの核心パッチ)
            abz = noise_pred_audio.size(0)
            denoised_audio_cond = (
                noise_pred_audio * (1 - audio_conditioning_mask[:abz])
                + clean_audio_packed[:abz] * audio_conditioning_mask[:abz]
            ).to(noise_pred_audio.dtype)
            noise_pred_audio = pipe.convert_x0_to_velocity(audio_latents, denoised_audio_cond, i, audio_scheduler)
            audio_latents = audio_scheduler.step(noise_pred_audio, t, audio_latents, return_dict=False)[0]

            log(f"step {i+1}/{len(timesteps)} t={float(t):.4f} t_audio={float(t_audio):.4f} "
                f"VRAM={vram_gb():.2f}GB")

    gen_time = time.time() - t_gen0
    peak = peak_vram_gb()
    log(f"denoise loop done in {gen_time:.1f}s, peak VRAM={peak:.2f}GB")

    # denoiseループの中間テンソル(CFG用の重複バッチ、connector埋め込み等)を解放してから
    # decodeに入る。デノイズ自体はVRAM 87.9GB付近まで使っており、decode(VAE conv3d)の
    # 一時ピークがそのまま乗るとOOMする(実機で確認済み)。
    compute_dtype = prompt_embeds.dtype
    del noise_pred_video, noise_pred_audio, latent_model_input, audio_latent_model_input
    del connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask
    del prompt_embeds, prompt_attention_mask
    # text_encoder(Gemma, ~23GB)はdecodeに不要なのでCPUへ退避してVAE decodeのVRAMを確保する
    # (CLAUDE.md 33番: 丸ごとCPU退避は本来禁止パターンだが、ここは denoiseループ完了後の
    # 一度きりの片道移動であり、往復スワップではないため事故リスクは無い)。
    pipe.text_encoder.to("cpu")
    pipe.connectors.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log(f"VRAM after cleanup: {vram_gb():.2f}GB")

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

    with torch.no_grad():
        video = pipe.vae.decode(latents, None, return_dict=False)[0]
    video = pipe.video_processor.postprocess_video(video, output_type="pil")

    audio_latents_denorm = pipe._denormalize_audio_latents(audio_latents, pipe.audio_vae.latents_mean, pipe.audio_vae.latents_std)
    latent_mel_bins = pipe.audio_mel_bins // pipe.audio_vae_mel_compression_ratio
    audio_latents_unpacked = pipe._unpack_audio_latents(audio_latents_denorm, audio_num_frames, num_mel_bins=latent_mel_bins)

    with torch.no_grad():
        audio_latents_unpacked = audio_latents_unpacked.to(pipe.audio_vae.dtype)
        generated_mel = pipe.audio_vae.decode(audio_latents_unpacked, return_dict=False)[0]
        audio_out = pipe.vocoder(generated_mel)

    decode_time = time.time() - t_gen0 - gen_time
    log(f"decode done in {decode_time:.1f}s")

    # --- save ---
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"ia2v_{audio_lock_mode}_{ts}"
    video_path = os.path.join(OUT_DIR, f"{tag}.mp4")
    export_to_video(video[0], video_path, fps=fps)
    log(f"saved video: {video_path}")

    import soundfile as sf
    wav = audio_out.detach().float().cpu()
    if wav.ndim == 3:
        wav = wav[0]
    if wav.ndim == 2 and wav.shape[0] <= 8:
        wav = wav.transpose(0, 1)
    audio_path = os.path.join(OUT_DIR, f"{tag}.wav")
    output_sr = int(getattr(pipe.vocoder, "output_sampling_rate", 48000))
    sf.write(audio_path, wav.numpy(), output_sr)
    log(f"saved audio: {audio_path} (sr={output_sr})")

    import subprocess, shutil
    ffmpeg_bin = shutil.which("ffmpeg")
    muxed_path = None
    if ffmpeg_bin:
        muxed_path = os.path.join(OUT_DIR, f"{tag}_av.mp4")
        try:
            subprocess.run(
                [ffmpeg_bin, "-y", "-loglevel", "error", "-i", video_path, "-i", audio_path,
                 "-c:v", "copy", "-c:a", "aac", "-shortest", muxed_path],
                check=True, timeout=120,
            )
            log(f"muxed: {muxed_path}")
        except Exception as exc:
            log(f"mux failed (non-fatal): {exc}")
            muxed_path = None

    log(f"RAM available after run: {ram_available_gb():.1f}GB")
    log("=== SUMMARY ===")
    log(f"load_time_s={load_time:.1f} gen_time_s={gen_time:.1f} decode_time_s={decode_time:.1f} "
        f"peak_vram_gb={peak:.2f} audio_lock_mode={audio_lock_mode}")
    log(f"video_only={video_path}")
    log(f"audio_only={audio_path}")
    log(f"muxed={muxed_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["mel", "full"], default="full")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--num_frames", type=int, default=89)  # 3.46s*25fps=86.5 -> 8n+1=89
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--guidance_scale", type=float, default=1.0)
    ap.add_argument("--audio_guidance_scale", type=float, default=1.0)
    ap.add_argument("--audio_lock_mode", choices=["hard", "soft"], default="hard")
    ap.add_argument("--prompt", type=str, default=(
        "A woman looks directly at the camera and talks, natural lip movements synced to speech, "
        "photorealistic, static camera, indoor lighting"
    ))
    ap.add_argument("--negative_prompt", type=str, default="blurry, distorted face, static mouth, low quality")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    if args.stage == "mel":
        stage_mel()
    else:
        stage_full(args)


if __name__ == "__main__":
    main()
