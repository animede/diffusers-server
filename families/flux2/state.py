# -*- coding: utf-8 -*-
"""
FLUX.2 ファミリーのシングルトン状態(プロセス内ロック + ロード済みパイプライン)。

抽出元: flux2_diffusers/pipeline_manager.py
  - _lock, _pipeline, _pipeline_precision(行72-74)
  - load_pipeline() / get_loaded_precision()(行292-339)

FLUX.2 は T2I/I2I が単一の Flux2Pipeline インスタンスを共有する単純な構造のため、
グループ辞書を分割せず1つの dict にまとめる。

2026-07-19: 第2モデル "ecocoro"(alfredplpl/ecocoro-preview-1、Flux2KleinPipeline)を
廃止した(ユーザー決定)。dev(FLUX.2-dev bnb4bit)単一モデル構成に戻す。
モデルキャッシュ(~/.cache/huggingface/hub/models--alfredplpl--ecocoro-preview-1、~15GB)は
ユーザー承認待ちのため削除していない(コードから参照されなくなるだけ)。
"""
import threading

from families.flux2.runtime import Flux2RuntimeConfig

lock = threading.Lock()

_runtime_config: "Flux2RuntimeConfig | None" = None

# T2I/I2I で共有する単一の Flux2Pipeline インスタンス(FLUX.2-dev、bnb-4bit)。
# image=None なら T2I、image=[...] なら I2I として同じパイプラインを呼び分ける
# (flux2_diffusers の設計をそのまま踏襲)。
pipeline_state = {
    "pipe": None,
    "loaded": False,
    "load_time_s": None,
    "precision": None,
    "offload_mode": None,
}


def get_runtime_config() -> Flux2RuntimeConfig:
    global _runtime_config
    if _runtime_config is None:
        _runtime_config = Flux2RuntimeConfig()
    return _runtime_config


def reset_runtime_config_cache() -> None:
    """テスト用: 環境変数を変えて再読込したい場合に使う(通常の unload では呼ばない)。"""
    global _runtime_config
    _runtime_config = None
