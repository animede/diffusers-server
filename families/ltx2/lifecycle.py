# -*- coding: utf-8 -*-
"""
unload() / get_status() の共通実装(LTX-2.3 ファミリー)。

base(T2V)/ I2V は同一の nn.Module 群を参照するグループのため、target に関わらず
常に両方とも一括で解放する(families/z_image/lifecycle.py と同じ設計)。latent
upsampler(2026-07-20追加)は base の vae 等とは独立した別モデルだが、同様に target に
関わらず一括解放する(個別解放を許す設計上の利点がないため、他グループと同じ扱いにする)。

group offload(CLAUDE.md 33番・34番)を使っていた場合、transformer は CPU 側に静的
確保されていたため、unload() で参照を切って gc.collect() すればホストRAMも解放される
(丸ごとスワップではなく「参照を切ってGC」であることに注意。CPU常駐分のRAM解放は
free -h の available/free の回復で確認できる)。
"""
import gc

from core import gpu

from families.ltx2 import state


def unload(target: str = "all") -> dict:
    """VRAM/RAM解放。target の値に関わらず base/I2V/FLF/IC-LoRA/upsampler を一括解放する。"""
    freed = []
    with state.lock:
        bps = state.base_pipeline_state
        ips = state.i2v_pipeline_state
        fps = state.flf_pipeline_state
        ics = state.iclora_pipeline_state
        ups = state.upsampler_state
        if bps["loaded"]:
            bps["pipe"] = None
            bps["loaded"] = False
            bps["offload_mode"] = None
            bps["load_time_s"] = None
            bps["load_time_breakdown"] = None
            bps["lora_loaded"] = False
            bps["lora_enabled"] = False
            freed.append("ltx2-base")
        if ips["loaded"]:
            ips["pipe"] = None
            ips["loaded"] = False
            freed.append("ltx2-i2v")
        if fps["loaded"]:
            fps["pipe"] = None
            fps["loaded"] = False
            freed.append("ltx2-flf")
        if ics["loaded"]:
            ics["pipe"] = None
            ics["loaded"] = False
            freed.append("ltx2-iclora")
        if ups["loaded"]:
            ups["model"] = None
            ups["loaded"] = False
            freed.append("ltx2-upsampler")

        gc.collect()
        gpu.empty_cache()
        gpu.reset_peak_stats()
    print(f"[families.ltx2] unloaded: {freed}")
    return {"freed": freed}


def get_status() -> dict:
    bps = state.base_pipeline_state
    ips = state.i2v_pipeline_state
    fps = state.flf_pipeline_state
    ics = state.iclora_pipeline_state
    ups = state.upsampler_state
    runtime_config = state.get_runtime_config()
    return {
        "loaded": bps["loaded"],
        "load_time_s": bps["load_time_s"],
        "offload_mode": bps["offload_mode"],
        "load_time_breakdown": bps["load_time_breakdown"],
        "i2v_loaded": ips["loaded"],
        "flf_loaded": fps["loaded"],
        "iclora_loaded": ics["loaded"],
        "iclora_lora_loaded": bps.get("lora_loaded", False),
        "iclora_lora_enabled": bps.get("lora_enabled", False),
        "upsampler_loaded": ups["loaded"],
        "runtime_config": repr(runtime_config),
        "gpu_busy": gpu.generation_lock.locked(),
        "vram": gpu.vram_snapshot(),
        "ram_available_gb": gpu.available_ram_gb(),
    }


def is_loaded() -> bool:
    return state.base_pipeline_state["loaded"]
