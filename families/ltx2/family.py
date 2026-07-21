# -*- coding: utf-8 -*-
"""
LTX2Family: core.registry.ModelFamily の実装(LTX-2.3 T2V / I2V)。

load(mode) / generate(request) / unload() / status() の共通インターフェースで
T2V / I2V をまとめる(families/z_image/family.py と対になる実装)。

mode:
  "t2v"       -> pipeline.get_base_pipeline()
  "i2v"       -> pipeline.get_i2v_pipeline()(base の components から遅延構築)
  "flf"       -> pipeline.get_flf_pipeline()(base の components から遅延構築、First-Last-Frame)
  "ia2v"      -> pipeline.get_flf_pipeline()(LTX2ConditionPipeline を流用、Image+Audio to Video、
                 2026-07-19追加。run_ia2v() が独自denoiseループでカスタム処理するため、
                 ロード自体は flf と同一パイプラインを使う)
  "keyframes" -> pipeline.get_flf_pipeline()(FLFの一般化、任意枚数・任意フレーム位置・
                 任意strengthのキーフレーム条件付け。ロードは flf と同一パイプラインを使う)
  "v2a"       -> pipeline.get_flf_pipeline()(Video to Audio、IA2Vの鏡像、2026-07-20追加。
                 run_v2a() が独自denoiseループでカスタム処理するため、ロード自体は
                 flf/ia2v と同一パイプラインを使う。追加モデルDLは不要)
  "iclora"    -> pipeline.get_iclora_pipeline()(IC-LoRA、MergeGreen 動画編集、2026-07-20追加。
                 LTX2InContextPipeline を base の components から遅延構築し、base の
                 transformer に MergeGreen LoRA アダプタをロードする。他モードへの
                 切替時は pipeline.disable_iclora() で無効化のみ行い、アンロードは
                 しない(再度 iclora が要求されたら再有効化のみで再ロード不要)。
"""
from core.registry import ModelFamily

from families.ltx2 import generate as generate_mod
from families.ltx2 import lifecycle
from families.ltx2 import pipeline as pipeline_mod

_ICLORA_MODE = "iclora"


class LTX2Family(ModelFamily):
    name = "ltx2"

    def load(self, mode: str, **kwargs) -> None:
        if mode == "t2v":
            pipeline_mod.get_base_pipeline()
        elif mode == "i2v":
            pipeline_mod.get_i2v_pipeline()
        elif mode in ("flf", "ia2v", "keyframes", "v2a"):
            pipeline_mod.get_flf_pipeline()
        elif mode == _ICLORA_MODE:
            pipeline_mod.get_iclora_pipeline()
        else:
            raise ValueError(f"unknown ltx2 mode: {mode!r}")

    def generate(self, request: dict) -> dict:
        mode = request["mode"]
        if mode == "t2v":
            return generate_mod.run_t2v(request)
        if mode == "i2v":
            return generate_mod.run_i2v(request, request["_image"])
        if mode == "flf":
            return generate_mod.run_flf(request, request["_first_image"], request["_last_image"])
        if mode == "ia2v":
            return generate_mod.run_ia2v(request, request["_image"], request["_audio_path"])
        if mode == "keyframes":
            return generate_mod.run_keyframes(
                request, request["_images"], request["_indices"], request["_strengths"]
            )
        if mode == "v2a":
            return generate_mod.run_v2a(request, request["_video_path"])
        if mode == _ICLORA_MODE:
            return generate_mod.run_iclora(request, request["_video_path"])
        raise ValueError(f"unknown ltx2 mode: {mode!r}")

    def unload(self) -> dict:
        return lifecycle.unload("all")

    def status(self) -> dict:
        return lifecycle.get_status()

    def is_loaded(self) -> bool:
        return lifecycle.is_loaded()
