# -*- coding: utf-8 -*-
"""
Qwen-Image 系ファミリー(T2I / I2I / Edit / ControlNet / Inpaint / Layered)。

Ｑwenimage-edit-diffusers/pipeline_manager.py(1669行)を分割移植したパッケージ。
公開インターフェースは core.registry.ModelFamily 経由(load/generate/unload/status)に加え、
app.py が直接使う補助関数(画像前処理・canny・パス定数など)をこのモジュールから re-export する。

サブモジュール構成:
  paths.py      - モデルパス・リポジトリ定数(元 pipeline_manager.py 行46-236)
  runtime.py    - RuntimeConfig・quant判定(元 行291-352)
  state.py      - シングルトン状態(グループ辞書・ロック・オフロードモード決定、元 行660-807)
  shared.py     - 共有コンポーネント(vae/text_encoder/tokenizer)ロード(元 行774-941)
  t2i.py        - T2I/I2I(qwen-image / 2512)ロード・Lightning制御(元 行810-1146, 1508-1529)
  edit.py       - Edit ロード・Lightning制御(元 行1148-1376, 1532-1537)
  controlnet.py - ControlNet Union/Inpaint ロード(元 行1029-1111)
  layered.py    - Layered ロード(元 行1381-1502)
  lifecycle.py  - unload() / get_status()(元 行1540-1669)
  generate.py   - 生成本体(元 app.py の各エンドポイント関数の中身)
  family.py     - QwenImageFamily(core.registry.ModelFamily の実装)

このパッケージを import した時点で core.registry.registry にシングルトンとして自動登録する
(app.py 起動時に import families.qwen_image するだけでよい)。
"""
from core.registry import registry

from families.qwen_image.family import QwenImageFamily
from families.qwen_image.generate import (
    MAX_EDIT_IMAGES,
    OUTPUTS_DIR,
    make_generator,
    preprocess_image,
    run_canny,
    run_controlnet,
    run_edit,
    run_edit_angles,
    run_edit_angles_bf16,
    run_edit_angles_bf16group,
    run_i2i,
    run_inpaint,
    run_layered,
    run_t2i,
    t2i_model_meta,
)
from families.qwen_image.lifecycle import get_status, unload
from families.qwen_image.paths import (
    LAYERED_DEFAULT_LAYERS,
    LAYERED_DEFAULT_RESOLUTION,
    LAYERED_VALID_RESOLUTIONS,
    NEGATIVE_PROMPT_DEFAULT,
    is_2512_model,
)
from families.qwen_image.state import get_runtime_config, lock

__all__ = [
    "QwenImageFamily",
    "family",
    "get_status",
    "unload",
    "run_t2i",
    "run_i2i",
    "run_edit",
    "run_edit_angles",
    "run_edit_angles_bf16",
    "run_edit_angles_bf16group",
    "run_controlnet",
    "run_inpaint",
    "run_canny",
    "run_layered",
    "preprocess_image",
    "make_generator",
    "t2i_model_meta",
    "MAX_EDIT_IMAGES",
    "OUTPUTS_DIR",
    "NEGATIVE_PROMPT_DEFAULT",
    "LAYERED_DEFAULT_LAYERS",
    "LAYERED_DEFAULT_RESOLUTION",
    "LAYERED_VALID_RESOLUTIONS",
    "is_2512_model",
    "get_runtime_config",
    "lock",
]

# プロセス内シングルトン(core.registry.registry に登録)。
family = QwenImageFamily()
registry.register(family)
