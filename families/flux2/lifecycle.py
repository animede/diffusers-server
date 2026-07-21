# -*- coding: utf-8 -*-
"""
unload() / get_status() の共通実装(FLUX.2 ファミリー)。

抽出元: 概念的には flux2_diffusers に unload/status 相当の実装はない
(Gradio版はプロセス終了までロードしっぱなしの単純設計だった)。Qwen系
(families/qwen_image/lifecycle.py)の設計パターン(dict をNoneに戻す + gc.collect() +
empty_cache() + reset_peak_stats())をそのまま踏襲して新規実装する。

2026-07-19: ecocoro廃止に伴い dev 単一モデル構成に戻した。target は "dev" 単体のみ持つ
(旧実装の "ecocoro" は削除、"all"/"t2i"/"i2i" 等の未知の値は従来どおり dev を解放する)。
"""
import gc

from core import gpu

from families.flux2 import state


def unload(target: str = "all") -> dict:
    """VRAM解放。target の値に関わらず dev グループ(唯一のグループ)を解放する。"""
    freed = []
    with state.lock:
        ps = state.pipeline_state
        if ps["loaded"]:
            ps["pipe"] = None
            ps["loaded"] = False
            ps["precision"] = None
            ps["offload_mode"] = None
            freed.append("flux2-dev")

        gc.collect()
        gpu.empty_cache()
        gpu.reset_peak_stats()
    print(f"[families.flux2] unloaded: {freed}")
    return {"freed": freed}


def get_status() -> dict:
    ps = state.pipeline_state
    runtime_config = state.get_runtime_config()
    return {
        "active_model": "dev",
        "loaded": ps["loaded"],
        "load_time_s": ps["load_time_s"],
        "precision": ps["precision"],
        "offload_mode": ps["offload_mode"],
        "dev_loaded": ps["loaded"],
        "dev_load_time_s": ps["load_time_s"],
        "runtime_config": repr(runtime_config),
        "gpu_busy": gpu.generation_lock.locked(),
        "vram": gpu.vram_snapshot(),
    }


def is_loaded() -> bool:
    return state.pipeline_state["loaded"]
