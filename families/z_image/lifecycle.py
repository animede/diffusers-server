# -*- coding: utf-8 -*-
"""
unload() / get_status() の共通実装(Z-Image ファミリー)。

Qwen系(families/qwen_image/lifecycle.py)・flux2系(families/flux2/lifecycle.py)の
設計パターン(dict をNoneに戻す + gc.collect() + empty_cache() + reset_peak_stats())を
踏襲する。base(T2I)/ I2I / Inpaint は同一の nn.Module 群を参照するグループのため、
target に関わらず常に3つとも一括で解放する(base だけ解放して I2I/Inpaint の参照を
残すと、components 経由の共有参照が宙に浮いた状態になるため)。
"""
import gc

from core import gpu

from families.z_image import state


def unload(target: str = "all") -> dict:
    """VRAM解放。target の値に関わらず base/I2I/Inpaint を一括解放する
    (3パイプラインは同一コンポーネントを共有するグループのため個別解放は行わない)。
    """
    freed = []
    with state.lock:
        bps = state.base_pipeline_state
        ips = state.i2i_pipeline_state
        nps = state.inpaint_pipeline_state
        if bps["loaded"]:
            bps["pipe"] = None
            bps["loaded"] = False
            bps["precision"] = None
            bps["offload_mode"] = None
            freed.append("z_image-base")
        if ips["loaded"]:
            ips["pipe"] = None
            ips["loaded"] = False
            freed.append("z_image-i2i")
        if nps["loaded"]:
            nps["pipe"] = None
            nps["loaded"] = False
            freed.append("z_image-inpaint")

        gc.collect()
        gpu.empty_cache()
        gpu.reset_peak_stats()
    print(f"[families.z_image] unloaded: {freed}")
    return {"freed": freed}


def get_status() -> dict:
    bps = state.base_pipeline_state
    ips = state.i2i_pipeline_state
    nps = state.inpaint_pipeline_state
    runtime_config = state.get_runtime_config()
    return {
        "loaded": bps["loaded"],
        "load_time_s": bps["load_time_s"],
        "precision": bps["precision"],
        "offload_mode": bps["offload_mode"],
        "i2i_loaded": ips["loaded"],
        "inpaint_loaded": nps["loaded"],
        "runtime_config": repr(runtime_config),
        "gpu_busy": gpu.generation_lock.locked(),
        "vram": gpu.vram_snapshot(),
    }


def is_loaded() -> bool:
    return state.base_pipeline_state["loaded"]
