# -*- coding: utf-8 -*-
"""
unload() / get_status() の共通実装。

抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py
  - unload()(行1540-1621)
  - get_status()(行1624-1669)

ロジック(解放順序含む)は無変更。core.gpu 経由で empty_cache / reset_peak_stats を呼ぶ点のみ
core 共通化に合わせて変更(挙動は同一)。
"""
import gc

from core import gpu

from families.qwen_image import state


def unload(target: str = "all") -> dict:
    """VRAM 解放。target: "t2i"(T2I/I2I グループ。2512/無印どちらのモデルでも同一グループ) /
    "edit"(Editグループ) /
    "edit_angles"(charsheet専用 fp8-lightning-angles グループ) /
    "edit_angles_bf16"(charsheet専用 fp8-base-adapters グループ) /
    "edit_angles_bf16group"(charsheet専用 bf16-group グループ、旧charsheet完全再現) /
    "controlnet"(ControlNet Union/Inpaintグループ) / "layered"(Layeredグループ) /
    "all"(共有含め全て)。

    ControlNetパイプラインは t2i_group["transformer"] を直接参照している(コピーではない)ため、
    先にControlNetグループを解放してから t2i グループを解放する(順序が逆だと transformer への
    参照がControlNet側に残ったままになり、VRAMが実際には解放されない)。

    抽出元: pipeline_manager.py unload()(行1540-1621)。
    """
    freed = []
    with state.lock:
        controlnet_union_group = state.controlnet_union_group
        controlnet_inpaint_group = state.controlnet_inpaint_group
        t2i_group = state.t2i_group
        edit_group = state.edit_group
        edit_angles_group = state.edit_angles_group
        edit_angles_bf16_group = state.edit_angles_bf16_group
        edit_angles_bf16group_group = state.edit_angles_bf16group_group
        layered_group = state.layered_group
        shared = state.shared

        if target in ("t2i", "controlnet", "all") and controlnet_union_group["loaded"]:
            controlnet_union_group["controlnet"] = None
            controlnet_union_group["pipe"] = None
            controlnet_union_group["scheduler"] = None
            controlnet_union_group["loaded"] = False
            freed.append("controlnet-union")
        if target in ("t2i", "controlnet", "all") and controlnet_inpaint_group["loaded"]:
            controlnet_inpaint_group["controlnet"] = None
            controlnet_inpaint_group["pipe"] = None
            controlnet_inpaint_group["scheduler"] = None
            controlnet_inpaint_group["loaded"] = False
            freed.append("controlnet-inpaint")
        if target in ("t2i", "all") and t2i_group["loaded"]:
            t2i_group["transformer"] = None
            t2i_group["t2i_pipe"] = None
            t2i_group["i2i_pipe"] = None
            t2i_group["scheduler"] = None
            t2i_group["loaded"] = False
            t2i_group["quant"] = None
            t2i_group["lora_available"] = None
            t2i_group["lora_unavailable_reason"] = None
            t2i_group["lightning_merged"] = False
            freed.append("t2i")
        if target in ("edit", "all") and edit_group["loaded"]:
            edit_group["transformer"] = None
            edit_group["edit_pipe"] = None
            edit_group["scheduler"] = None
            edit_group["processor"] = None
            edit_group["loaded"] = False
            edit_group["quant"] = None
            edit_group["lora_available"] = None
            edit_group["lora_unavailable_reason"] = None
            edit_group["lightning_merged"] = False
            freed.append("edit")
        if target in ("edit_angles", "all") and edit_angles_group["loaded"]:
            edit_angles_group["transformer"] = None
            edit_angles_group["edit_pipe"] = None
            edit_angles_group["scheduler"] = None
            edit_angles_group["processor"] = None
            edit_angles_group["loaded"] = False
            edit_angles_group["quant"] = None
            edit_angles_group["lightning_merged"] = False
            freed.append("edit_angles")
        if target in ("edit_angles_bf16", "all") and edit_angles_bf16_group["loaded"]:
            edit_angles_bf16_group["transformer"] = None
            edit_angles_bf16_group["edit_pipe"] = None
            edit_angles_bf16_group["scheduler"] = None
            edit_angles_bf16_group["processor"] = None
            edit_angles_bf16_group["loaded"] = False
            edit_angles_bf16_group["quant"] = None
            edit_angles_bf16_group["lightning_merged"] = False
            freed.append("edit_angles_bf16")
        if target in ("edit_angles_bf16group", "all") and edit_angles_bf16group_group["loaded"]:
            edit_angles_bf16group_group["transformer"] = None
            edit_angles_bf16group_group["edit_pipe"] = None
            edit_angles_bf16group_group["scheduler"] = None
            edit_angles_bf16group_group["processor"] = None
            edit_angles_bf16group_group["loaded"] = False
            edit_angles_bf16group_group["quant"] = None
            edit_angles_bf16group_group["lightning_merged"] = False
            edit_angles_bf16group_group["group_offload_config"] = None
            freed.append("edit_angles_bf16group")
        if target in ("layered", "all") and layered_group["loaded"]:
            layered_group["transformer"] = None
            layered_group["vae"] = None
            layered_group["processor"] = None
            layered_group["pipe"] = None
            layered_group["scheduler"] = None
            layered_group["loaded"] = False
            layered_group["quant"] = None
            layered_group["lightning_merged"] = False
            freed.append("layered")
        if (
            target == "all"
            and shared["loaded"]
            and not t2i_group["loaded"]
            and not edit_group["loaded"]
            and not edit_angles_group["loaded"]
            and not edit_angles_bf16_group["loaded"]
            and not edit_angles_bf16group_group["loaded"]
            and not layered_group["loaded"]
        ):
            shared["vae"] = None
            shared["text_encoder"] = None
            shared["tokenizer"] = None
            shared["loaded"] = False
            freed.append("shared")

        gc.collect()
        gpu.empty_cache()
        gpu.reset_peak_stats()
    print(f"[families.qwen_image] unloaded: {freed}")
    return {"freed": freed}


def unload_shared_if_unused() -> bool:
    """他のどのグループもロードしていなければ共有コンポーネント(vae/text_encoder/
    tokenizer、bf16で~15.7GB)を解放する。解放したら True を返す。

    追加理由(edit <-> edit_angles 切替時のOOM対策、実機検証で判明): fp8-lightning系の
    Edit変種(edit.py / edit_angles.py)は「bf16 transformerを直接cuda:0へロード ->
    LoRA fuse(bf16のまま一時的に約38-39GB)-> fp8 layerwise cast」という手順のため、
    fuse完了までの一時ピークだけで38GB超のVRAMを必要とする。旧グループを解放しても
    共有コンポーネント(~15.7GB)が常駐したままだと、48GB専有環境では
    (共有15.7GB + 新transformerのfuse時ピーク38GB以上)が総VRAM(実測空き~47GB)を
    超えてOOMする(実機確認済み: edit_angles常駐中に/api/editを叩くとCUDA OOM)。
    そのため edit <-> edit_angles 切替時は共有コンポーネントも一旦解放し、新グループの
    _load_*_group_locked() 内で(fuse完了後に)再ロードさせる(再ロードは実測 ~25秒)。
    edit_angles_bf16(2026-07-18追加、fuseしないbf16常駐変種)も transformer が
    ~38GB常駐という点で同種のリスクを持つため、edit/edit_angles/edit_angles_bf16 の
    3方向いずれの切替でもこの関数を呼ぶ(families/qwen_image/family.py
    _unload_other_edit_variants() 参照)。edit_angles_bf16group(CLAUDE.md 33番、
    bf16-group、旧charsheet完全再現)も同様に共有コンポーネントを参照するため対象に含める。
    """
    freed = False
    with state.lock:
        shared = state.shared
        if (
            shared["loaded"]
            and not state.t2i_group["loaded"]
            and not state.edit_group["loaded"]
            and not state.edit_angles_group["loaded"]
            and not state.edit_angles_bf16_group["loaded"]
            and not state.edit_angles_bf16group_group["loaded"]
            and not state.layered_group["loaded"]
        ):
            shared["vae"] = None
            shared["text_encoder"] = None
            shared["tokenizer"] = None
            shared["loaded"] = False
            freed = True
        gc.collect()
        gpu.empty_cache()
        gpu.reset_peak_stats()
    if freed:
        print("[families.qwen_image] unloaded: ['shared'] (Edit系グループ切替のため一旦解放)")
    return freed


def get_status() -> dict:
    """抽出元: pipeline_manager.py get_status()(行1624-1669)。"""
    from families.qwen_image.paths import is_2512_model

    runtime_config = state.get_runtime_config()
    t2i_group = state.t2i_group
    edit_group = state.edit_group
    edit_angles_group = state.edit_angles_group
    edit_angles_bf16_group = state.edit_angles_bf16_group
    edit_angles_bf16group_group = state.edit_angles_bf16group_group
    controlnet_union_group = state.controlnet_union_group
    controlnet_inpaint_group = state.controlnet_inpaint_group
    layered_group = state.layered_group
    shared = state.shared

    status = {
        "offload_mode": state._offload_mode,
        "runtime_config": repr(runtime_config),
        "shared_loaded": shared["loaded"],
        "shared_load_time_s": shared["load_time_s"],
        "t2i_model": "2512" if is_2512_model(runtime_config.t2i_model) else "qwen-image",
        "t2i_loaded": t2i_group["loaded"],
        "t2i_load_time_s": t2i_group["load_time_s"],
        "t2i_quant": t2i_group["quant"],
        "t2i_lora_available": t2i_group["lora_available"],
        "t2i_lora_unavailable_reason": t2i_group["lora_unavailable_reason"],
        "t2i_lightning_merged": t2i_group["lightning_merged"],
        "edit_loaded": edit_group["loaded"],
        "edit_load_time_s": edit_group["load_time_s"],
        "edit_transformer_fallback": edit_group["fallback"],
        "edit_quant": edit_group["quant"],
        "edit_lora_available": edit_group["lora_available"],
        "edit_lora_unavailable_reason": edit_group["lora_unavailable_reason"],
        "edit_lightning_merged": edit_group["lightning_merged"],
        "edit_angles_loaded": edit_angles_group["loaded"],
        "edit_angles_load_time_s": edit_angles_group["load_time_s"],
        "edit_angles_quant": edit_angles_group["quant"],
        "edit_angles_lightning_merged": edit_angles_group["lightning_merged"],
        "edit_angles_bf16_loaded": edit_angles_bf16_group["loaded"],
        "edit_angles_bf16_load_time_s": edit_angles_bf16_group["load_time_s"],
        "edit_angles_bf16_quant": edit_angles_bf16_group["quant"],
        "edit_angles_bf16_lightning_merged": edit_angles_bf16_group["lightning_merged"],
        "edit_angles_bf16group_loaded": edit_angles_bf16group_group["loaded"],
        "edit_angles_bf16group_load_time_s": edit_angles_bf16group_group["load_time_s"],
        "edit_angles_bf16group_quant": edit_angles_bf16group_group["quant"],
        "edit_angles_bf16group_lightning_merged": edit_angles_bf16group_group["lightning_merged"],
        "edit_angles_bf16group_offload_config": edit_angles_bf16group_group["group_offload_config"],
        "controlnet_union_loaded": controlnet_union_group["loaded"],
        "controlnet_union_load_time_s": controlnet_union_group["load_time_s"],
        "controlnet_inpaint_loaded": controlnet_inpaint_group["loaded"],
        "controlnet_inpaint_load_time_s": controlnet_inpaint_group["load_time_s"],
        "layered_loaded": layered_group["loaded"],
        "layered_load_time_s": layered_group["load_time_s"],
        "layered_quant": layered_group["quant"],
        "layered_lightning_merged": layered_group["lightning_merged"],
        "gpu_busy": gpu.generation_lock.locked(),
        "vram": gpu.vram_snapshot(),
    }
    return status


def is_any_loaded() -> bool:
    """FamilyRegistry の排他切替(Phase 2 の flux2 併用時)用: このファミリーが
    何かしらロード済みかどうか(qwen_imageは複数グループを持つため個別に確認する)。
    """
    return (
        state.shared["loaded"]
        or state.t2i_group["loaded"]
        or state.edit_group["loaded"]
        or state.edit_angles_group["loaded"]
        or state.edit_angles_bf16_group["loaded"]
        or state.edit_angles_bf16group_group["loaded"]
        or state.controlnet_union_group["loaded"]
        or state.controlnet_inpaint_group["loaded"]
        or state.layered_group["loaded"]
    )
