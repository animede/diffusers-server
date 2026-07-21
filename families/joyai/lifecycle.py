# -*- coding: utf-8 -*-
"""
unload() / get_status() の共通実装(JoyAI ファミリー)。

families/z_image/lifecycle.py・families/ltx2/lifecycle.py と同じパターン
(dict を None に戻す + gc.collect() + empty_cache() + reset_peak_stats())。
"""
import gc

from core import gpu

from families.joyai import state


def unload(target: str = "all") -> dict:
    freed = []
    with state.lock:
        ps = state.pipeline_state
        if ps["loaded"]:
            ps["pipe"] = None
            ps["loaded"] = False
            ps["load_time_s"] = None
            ps["te_offload"] = None
            ps["patched"] = False
            freed.append("joyai-edit")

        gc.collect()
        gpu.empty_cache()
        gpu.reset_peak_stats()
    print(f"[families.joyai] unloaded: {freed}")
    return {"freed": freed}


def get_status() -> dict:
    ps = state.pipeline_state
    runtime_config = state.get_runtime_config()
    return {
        "loaded": ps["loaded"],
        "load_time_s": ps["load_time_s"],
        "te_offload": ps["te_offload"],
        "patched": ps["patched"],
        "runtime_config": repr(runtime_config),
        "gpu_busy": gpu.generation_lock.locked(),
        "vram": gpu.vram_snapshot(),
    }


def is_loaded() -> bool:
    return state.pipeline_state["loaded"]
