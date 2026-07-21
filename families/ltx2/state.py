# -*- coding: utf-8 -*-
"""
LTX-2.3 ファミリーのシングルトン状態(プロセス内ロック + ロード済みパイプライン)。

T2V(LTX2Pipeline)を base としてロードし、I2V(LTX2ImageToVideoPipeline)は base の
components から遅延構築する(families/z_image/state.py と同じ設計。参照共有のため
VRAM 使用量は増えない)。
"""
import threading

from families.ltx2.runtime import LTX2RuntimeConfig

lock = threading.Lock()

_runtime_config: "LTX2RuntimeConfig | None" = None

# base(T2V)。LTX2Pipeline。I2V はこのパイプラインの components から遅延構築される。
# lora_loaded / lora_enabled(IC-LoRA、2026-07-20追加): base transformer に MergeGreen
# IC-LoRA アダプタがロード済みか(load_lora_adapter() 実行済みか)、現在有効化されて
# いるか(set_adapters() vs disable_lora())を追跡する。ロードはロード時(group offload
# 登録前)にしか行えないため、「未ロードの base が既に常駐している」場合は一旦 unload
# してから load_lora=True で再ロードする設計(families/ltx2/pipeline.py 参照)。
base_pipeline_state = {
    "pipe": None,
    "loaded": False,
    "load_time_s": None,
    "offload_mode": None,
    "load_time_breakdown": None,
    "lora_loaded": False,
    "lora_enabled": False,
}

# I2V(LTX2ImageToVideoPipeline)。base_pipeline_state["pipe"].components から派生。
i2v_pipeline_state = {
    "pipe": None,
    "loaded": False,
}

# FLF(LTX2ConditionPipeline、First-Last-Frame)。base_pipeline_state["pipe"].components
# から派生(I2V と同じ参照共有パターン、VRAM追加なし)。
flf_pipeline_state = {
    "pipe": None,
    "loaded": False,
}

# IC-LoRA(LTX2InContextPipeline、MergeGreen 動画編集、2026-07-20追加)。
# base_pipeline_state["pipe"].components から派生(I2V/FLF と同じ参照共有パターン)。
# LoRA自体は base の transformer に直接ロードされる(families/ltx2/pipeline.py の
# get_iclora_pipeline()・base_pipeline_state["lora_loaded"]/["lora_enabled"] 参照)。
iclora_pipeline_state = {
    "pipe": None,
    "loaded": False,
}

# latent upsampler(LTX2LatentUpsamplerModel、2026-07-20追加)。T2V/I2Vの upscale=1
# オプション用に初回要求時だけ遅延ロードする(GPU常駐、~0.93GBと小さい)。base/I2V/FLFとは
# 独立したモデル(VAEのみ参照共有、transformer等とは無関係)のため専用のstateを持つ。
upsampler_state = {
    "model": None,
    "loaded": False,
}


def get_runtime_config() -> LTX2RuntimeConfig:
    global _runtime_config
    if _runtime_config is None:
        _runtime_config = LTX2RuntimeConfig()
    return _runtime_config


def reset_runtime_config_cache() -> None:
    """テスト用: 環境変数を変えて再読込したい場合に使う(通常の unload では呼ばない)。"""
    global _runtime_config
    _runtime_config = None
