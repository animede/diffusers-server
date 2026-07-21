# -*- coding: utf-8 -*-
"""
JoyAI-Image-Edit-Plus ファミリーのシングルトン状態(プロセス内ロック + ロード済みパイプライン)。

families/z_image/state.py・families/ltx2/state.py と同じパターン: 単一の
JoyImageEditPlusPipeline を pipeline_state に保持する(このファミリーは Edit のみを
提供するため、base/i2i/inpaint のような複数ラッパー構造は持たない)。
"""
import threading

from families.joyai.runtime import JoyAIRuntimeConfig

lock = threading.Lock()

_runtime_config: "JoyAIRuntimeConfig | None" = None

pipeline_state = {
    "pipe": None,
    "loaded": False,
    "load_time_s": None,
    "te_offload": None,
    "patched": False,  # PatchifyLinear ランタイムパッチ適用済みか
}


def get_runtime_config() -> JoyAIRuntimeConfig:
    global _runtime_config
    if _runtime_config is None:
        _runtime_config = JoyAIRuntimeConfig()
    return _runtime_config


def reset_runtime_config_cache() -> None:
    global _runtime_config
    _runtime_config = None
