# -*- coding: utf-8 -*-
"""
Flux2Pipeline のロード(bnb-4bit ビルダー + offload/attention/compile 適用)。

抽出元: flux2_diffusers/pipeline_manager.py
  - MODEL_ID_BF16 / MODEL_ID_BNB_4BIT(行35-36)
  - _build_bnb_4bit() / _build_bf16() / PRECISION_BUILDERS(行119-135)
  - load_pipeline()(行292-334)

変更点:
  - offload 適用は core.optimize.apply_flux2_offload() に委譲(Qwen系と共通化、
    ロジックは flux2_diffusers _apply_offload() から無変更で移設済み)。
  - attention backend / compile 適用は core.optimize.apply_attention_backend() /
    apply_compile() に委譲(Qwen系と完全共通のロジック。flux2_diffusers 側は
    pipe.transformer に対して同じ呼び出しをしていたため、関数を差し替えるだけで
    挙動は同一)。
  - MODEL_ID_BF16 は「HFキャッシュから削除済み・再DL禁止」(タスク前提)のため、
    bf16 ビルダーはエラーメッセージ付きで残すが呼び出しは想定しない
    (DS_FLUX2_PRECISION=bf16 を明示指定した場合のみ到達し、gated repo + 106GB
    再ダウンロードが必要な旨を案内する)。

2026-07-19: ecocoro(alfredplpl/ecocoro-preview-1、Flux2KleinPipeline)を廃止(ユーザー決定)。
get_pipeline() は model 引数を受け取るが、"dev" 以外(または未指定)は "dev" として扱う。
"ecocoro" が明示指定された場合は ValueError を送出し、app.py 側で 400 に変換する。
"""
import time

import torch
from diffusers import Flux2Pipeline

from core.optimize import apply_attention_backend, apply_compile, apply_flux2_offload
from core import progress as progress_mod

from families.flux2 import state
from families.flux2.runtime import Flux2RuntimeConfig

MODEL_ID_BF16 = "black-forest-labs/FLUX.2-dev"
MODEL_ID_BNB_4BIT = "diffusers/FLUX.2-dev-bnb-4bit"


def _build_bnb_4bit(config: Flux2RuntimeConfig) -> Flux2Pipeline:
    # 重みはHub上で既に4bit量子化済み(bitsandbytesがオンザフライで逆量子化)。
    # torch_dtype は非量子化モジュール(text_encoderの一部/VAE等)の計算dtypeを決める。
    return Flux2Pipeline.from_pretrained(MODEL_ID_BNB_4BIT, torch_dtype=torch.bfloat16)


def _build_bf16(config: Flux2RuntimeConfig) -> Flux2Pipeline:
    print(
        "[families.flux2] warning: bf16フル精度版(black-forest-labs/FLUX.2-dev)は "
        "このマシンのHFキャッシュから削除済みで、再ダウンロードも禁止されています "
        "(タスク前提)。DS_FLUX2_PRECISION=bnb-4bit を使ってください。"
    )
    return Flux2Pipeline.from_pretrained(MODEL_ID_BF16, torch_dtype=torch.bfloat16)


PRECISION_BUILDERS = {
    "bnb-4bit": _build_bnb_4bit,
    "bf16": _build_bf16,
}


def _load_pipeline_locked() -> None:
    """state.lock を呼び出し側が保持している前提でロードする(冪等)。"""
    ps = state.pipeline_state
    if ps["loaded"]:
        return

    config = state.get_runtime_config()
    if config.precision not in PRECISION_BUILDERS:
        raise ValueError(
            f"Unknown DS_FLUX2_PRECISION {config.precision!r}. Available: {list(PRECISION_BUILDERS)}"
        )

    print(f"[families.flux2] loading Flux2Pipeline: {config}")
    t0 = time.time()

    builder = PRECISION_BUILDERS[config.precision]
    pipe = builder(config)

    apply_flux2_offload(pipe, config.offload, device=config.device, group_stream=config.group_stream)
    apply_attention_backend(pipe.transformer, config.attention_backend, config.device)
    apply_compile(pipe.transformer, config.compile)

    ps["pipe"] = pipe
    ps["loaded"] = True
    ps["precision"] = config.precision
    ps["offload_mode"] = config.offload
    ps["load_time_s"] = time.time() - t0
    print(f"[families.flux2] loaded in {ps['load_time_s']:.1f}s (precision={config.precision}, offload={config.offload})")


def _validate_model(model: "str | None") -> None:
    """2026-07-19: ecocoro廃止に伴うバリデーション。"dev" と未指定のみ許可し、
    "ecocoro" が明示指定された場合は明確なエラーメッセージで拒否する
    (app.py 側で ValueError を 400 に変換する規約)。
    """
    if model is None:
        return
    model = model.strip().lower()
    if model == "ecocoro":
        raise ValueError("ecocoroは廃止されました。flux2 の model には 'dev' を指定してください。")
    if model != "dev":
        raise ValueError(f"flux2 model は 'dev' である必要があります(指定値: {model!r})")


def get_pipeline(model: "str | None" = None):
    """ロード済みなら即返す。未ロードならロックを取ってロードする(冪等)。

    model は "dev"(既定)| None のみ有効。"ecocoro" 指定時は ValueError を送出する
    (2026-07-19、ecocoro廃止)。
    """
    _validate_model(model)
    if not state.pipeline_state["loaded"]:
        with state.lock:
            if not state.pipeline_state["loaded"]:
                _load_pipeline_locked()
    pipe = state.pipeline_state["pipe"]
    progress_mod.disable_diffusers_tqdm(pipe)
    return pipe
