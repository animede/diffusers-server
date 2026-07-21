# -*- coding: utf-8 -*-
"""
FLUX.2 ファミリー(T2I / I2I)。

flux2_diffusers/pipeline_manager.py(340行)を分割移植したパッケージ。
公開インターフェースは core.registry.ModelFamily 経由(load/generate/unload/status)に加え、
app.py が直接使う補助関数(画像前処理・パス定数など)をこのモジュールから re-export する。

サブモジュール構成:
  runtime.py    - Flux2RuntimeConfig(環境変数、元 Flux2Config 行81-116)
  state.py      - シングルトン状態(ロック・pipeline_state)
  pipeline.py   - Flux2Pipeline ロード(bnb-4bit ビルダー + offload/attention/compile 適用、
                  元 _build_bnb_4bit/_build_bf16/load_pipeline 行119-334)
  generate.py   - 生成本体(T2I/I2I、元 flux2_diffusers/app.py の Gradio ハンドラ中身)
  lifecycle.py  - unload() / get_status()(新規実装、Qwen系のパターンを踏襲)
  family.py     - Flux2Family(core.registry.ModelFamily の実装)

このパッケージを import した時点で core.registry.registry にシングルトンとして自動登録する
(app.py 起動時に import families.flux2 するだけでよい)。

REBUILD_PLAN §3.1: Qwen 系と FLUX.2 は VRAM 同時常駐不可とみなし、register() の
exclusive_with=["qwen_image"] でファミリー切替時の自動 unload を有効化する。
"""
from core.registry import registry

from families.flux2.family import Flux2Family
from families.flux2.generate import MAX_REF_IMAGES, OUTPUTS_DIR, make_generator, run_i2i, run_t2i
from families.flux2.lifecycle import get_status, unload
from families.flux2.state import get_runtime_config, lock

__all__ = [
    "Flux2Family",
    "family",
    "get_status",
    "unload",
    "run_t2i",
    "run_i2i",
    "make_generator",
    "MAX_REF_IMAGES",
    "OUTPUTS_DIR",
    "get_runtime_config",
    "lock",
]

# プロセス内シングルトン(core.registry.registry に登録)。
# exclusive_with=["qwen_image"]: どちらかをロードすると、もう一方が常駐していれば
# registry.load() が自動 unload してから load() する(REBUILD_PLAN §3.1)。
family = Flux2Family()
registry.register(family, exclusive_with=["qwen_image"])
