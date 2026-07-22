# -*- coding: utf-8 -*-
"""
Z-Image ファミリーのシングルトン状態(プロセス内ロック + ロード済みパイプライン)。

抽出元: /home/animede/Z-image-diffusers/pipeline_manager.py
  - _lock, _base_pipeline, _i2i_pipeline, _inpaint_pipeline, _pipeline_config_repr(行90-94)
  - load_pipeline() / get_i2i_pipeline() / get_inpaint_pipeline()(行343-420)

旧実装と同じ設計: 単一の ZImagePipeline(base、T2I)をロードし、I2I / Inpaint は
`base_pipe.components`(参照共有、コピーではない)から遅延構築する。そのため VRAM 使用量は
1モデル分のみ(3つのパイプラインラッパーが同じ nn.Module インスタンス群を共有する)。
"""
import threading

from families.z_image.runtime import ZImageRuntimeConfig

lock = threading.Lock()

_runtime_config: "ZImageRuntimeConfig | None" = None

# base(T2I)。ZImagePipeline。I2I/Inpaint はこのパイプラインの components から
# 遅延構築される(base が未ロードなら base_pipeline_state 経由で先にロードする)。
base_pipeline_state = {
    "pipe": None,
    "loaded": False,
    "load_time_s": None,
    "precision": None,
    "offload_mode": None,
}

# I2I(ZImageImg2ImgPipeline)。base_pipeline_state["pipe"].components から派生。
i2i_pipeline_state = {
    "pipe": None,
    "loaded": False,
}

# Inpaint(ZImageInpaintPipeline)。base_pipeline_state["pipe"].components から派生。
inpaint_pipeline_state = {
    "pipe": None,
    "loaded": False,
}


def get_runtime_config() -> ZImageRuntimeConfig:
    global _runtime_config
    if _runtime_config is None:
        _runtime_config = ZImageRuntimeConfig()
    return _runtime_config


def reset_runtime_config_cache() -> None:
    """テスト用: 環境変数を変えて再読込したい場合に使う(通常の unload では呼ばない)。"""
    global _runtime_config
    _runtime_config = None
