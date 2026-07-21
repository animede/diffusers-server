# -*- coding: utf-8 -*-
"""
ControlNet(Union: canny/depth/pose等 + Inpainting)のロード。

抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py
  - _load_controlnet_union_group_locked()(行1029-1064)
  - _load_controlnet_inpaint_group_locked()(行1067-1095)
  - get_controlnet_pipeline() / get_controlnet_inpaint_pipeline()(行1098-1111)

注意(旧 CLAUDE.md 13番の罠): これらの "_locked" 関数は呼び出し側(get_controlnet_pipeline()
等)が既に state.lock を保持している前提。内部で t2i._load_t2i_group_locked() を直接呼ぶ
(state.lock を再度取得しない)。
"""
import time

from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    QwenImageControlNetInpaintPipeline,
    QwenImageControlNetModel,
    QwenImageControlNetPipeline,
)

from families.qwen_image import state, t2i
from families.qwen_image.paths import BASE_REPO, CONTROLNET_INPAINT_REPO, CONTROLNET_UNION_REPO


def _load_controlnet_union_group_locked() -> None:
    """ControlNet Union(canny/depth/pose等)をロードする。base Qwen-Image の
    共有コンポーネント(vae/text_encoder/tokenizer)とT2Iグループのtransformerを再利用し、
    ControlNetモジュール(実測3.29GB)だけ追加ロードする。

    抽出元: pipeline_manager.py _load_controlnet_union_group_locked()(行1029-1064)。
    """
    controlnet_union_group = state.controlnet_union_group
    if controlnet_union_group["loaded"]:
        return
    # T2Iグループ(共有コンポーネント含む)を確実にロードする。ControlNet は無印
    # Qwen-Image ベースで学習されているため、常に無印の transformer を使う(呼び出し側
    # get_controlnet_pipeline() が事前に t2i_model を "qwen-image" へ切り替えている前提)。
    if not state.t2i_group["loaded"]:
        t2i._load_t2i_group_locked()

    import torch

    t0 = time.time()
    print(f"[families.qwen_image] loading ControlNet Union from {CONTROLNET_UNION_REPO}")
    controlnet = QwenImageControlNetModel.from_pretrained(CONTROLNET_UNION_REPO, torch_dtype=torch.bfloat16)
    controlnet.to("cuda" if torch.cuda.is_available() else "cpu")  # 小さいため常にフルGPU常駐

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(BASE_REPO, subfolder="scheduler")
    pipe = QwenImageControlNetPipeline(
        scheduler=scheduler,
        vae=state.shared["vae"],
        text_encoder=state.shared["text_encoder"],
        tokenizer=state.shared["tokenizer"],
        transformer=state.t2i_group["transformer"],
        controlnet=controlnet,
    )

    controlnet_union_group["controlnet"] = controlnet
    controlnet_union_group["pipe"] = pipe
    controlnet_union_group["scheduler"] = scheduler
    controlnet_union_group["loaded"] = True
    controlnet_union_group["load_time_s"] = time.time() - t0
    print(f"[families.qwen_image] ControlNet Union loaded in {controlnet_union_group['load_time_s']:.1f}s")


def _load_controlnet_inpaint_group_locked() -> None:
    """ControlNet Inpainting をロードする(構造はUnionと同様、専用リポジトリ)。

    抽出元: pipeline_manager.py _load_controlnet_inpaint_group_locked()(行1067-1095)。
    """
    controlnet_inpaint_group = state.controlnet_inpaint_group
    if controlnet_inpaint_group["loaded"]:
        return
    if not state.t2i_group["loaded"]:
        t2i._load_t2i_group_locked()

    import torch

    t0 = time.time()
    print(f"[families.qwen_image] loading ControlNet Inpainting from {CONTROLNET_INPAINT_REPO}")
    controlnet = QwenImageControlNetModel.from_pretrained(CONTROLNET_INPAINT_REPO, torch_dtype=torch.bfloat16)
    controlnet.to("cuda" if torch.cuda.is_available() else "cpu")

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(BASE_REPO, subfolder="scheduler")
    pipe = QwenImageControlNetInpaintPipeline(
        scheduler=scheduler,
        vae=state.shared["vae"],
        text_encoder=state.shared["text_encoder"],
        tokenizer=state.shared["tokenizer"],
        transformer=state.t2i_group["transformer"],
        controlnet=controlnet,
    )

    controlnet_inpaint_group["controlnet"] = controlnet
    controlnet_inpaint_group["pipe"] = pipe
    controlnet_inpaint_group["scheduler"] = scheduler
    controlnet_inpaint_group["loaded"] = True
    controlnet_inpaint_group["load_time_s"] = time.time() - t0
    print(f"[families.qwen_image] ControlNet Inpainting loaded in {controlnet_inpaint_group['load_time_s']:.1f}s")


def get_controlnet_pipeline() -> QwenImageControlNetPipeline:
    """抽出元: pipeline_manager.py get_controlnet_pipeline()(行1098-1103)。

    2026-07-18追加: t2i_group は 2512 / 無印のどちらかを排他的に保持するようになった
    (旧 2512-4bit 独立グループの廃止)。ControlNet(InstantX)は無印 Qwen-Image ベースで
    学習されているため、2512 がアクティブ/ロード済みなら先に無印へ切り替える
    (_switch_t2i_model_if_needed() は内部で state.lock を取得するため、ロック取得前に呼ぶ。
    切替時は t2i_group と ControlNet グループが解放され、無印で再ロードされる)。
    """
    t2i._switch_t2i_model_if_needed("qwen-image")
    if not state.controlnet_union_group["loaded"]:
        with state.lock:
            if not state.controlnet_union_group["loaded"]:
                _load_controlnet_union_group_locked()
    return state.controlnet_union_group["pipe"]


def get_controlnet_inpaint_pipeline() -> QwenImageControlNetInpaintPipeline:
    """抽出元: pipeline_manager.py get_controlnet_inpaint_pipeline()(行1106-1111)。
    無印への事前切替については get_controlnet_pipeline() のdocstring参照。
    """
    t2i._switch_t2i_model_if_needed("qwen-image")
    if not state.controlnet_inpaint_group["loaded"]:
        with state.lock:
            if not state.controlnet_inpaint_group["loaded"]:
                _load_controlnet_inpaint_group_locked()
    return state.controlnet_inpaint_group["pipe"]
