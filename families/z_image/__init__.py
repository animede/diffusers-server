# -*- coding: utf-8 -*-
"""
Z-Image ファミリー(T2I / I2I / Inpaint)。

~/Z-image-diffusers/pipeline_manager.py(429行)+ api_server.py(276行)を
分割移植したパッケージ。公開インターフェースは core.registry.ModelFamily 経由
(load/generate/unload/status)に加え、app.py が直接使う補助関数をこのモジュールから
re-export する。

サブモジュール構成:
  runtime.py    - ZImageRuntimeConfig(環境変数、元 ZImageConfig 行101-151)
  state.py      - シングルトン状態(ロック・pipeline_state群)
  pipeline.py   - ZImagePipeline ロード + I2I/Inpaint 遅延構築
                  (元 _build_bf16/_build_bnb_4bit/load_pipeline/get_i2i_pipeline/
                  get_inpaint_pipeline 行154-420)
  generate.py   - 生成本体(T2I/I2I/Inpaint、元 api_server.py の各エンドポイント中身)
  lifecycle.py  - unload() / get_status()(新規実装、他ファミリーのパターンを踏襲)
  family.py     - ZImageFamily(core.registry.ModelFamily の実装)

このパッケージを import した時点で core.registry.registry にシングルトンとして自動登録する
(app.py 起動時に import families.z_image するだけでよい)。

排他構成(2026-07-19、ユーザー指示によりスコープ追加): qwen_image / flux2 / z_image は
いずれも VRAM 同時常駐不可とみなし、3方向すべてを相互排他にする。
FamilyRegistry.register(family, exclusive_with=[...]) は呼び出しごとに
{family.name, *exclusive_with} の集合を _exclusive_groups に追加し、
_families_exclusive_with() はその集合群のうち name を含むものをすべて合成して
「相手」を求める(register 呼び出し1回で両方向が有効になる、families/flux2/__init__.py
の "qwen_image" <-> "flux2" 登録と同じ仕組み)。z_image 登録時に
exclusive_with=["qwen_image", "flux2"] を渡せば、この1回の登録だけで
qwen_image<->z_image・flux2<->z_image の両方向が有効になる(qwen_image側の register()
呼び出しを変更する必要はない)。
"""
from core.registry import registry

from families.z_image.family import ZImageFamily
from families.z_image.generate import make_generator, run_i2i, run_inpaint, run_t2i
from families.z_image.lifecycle import get_status, unload
from families.z_image.pipeline import DEFAULT_GUIDANCE_SCALE, DEFAULT_STEPS
from families.z_image.state import get_runtime_config, lock

__all__ = [
    "ZImageFamily",
    "family",
    "get_status",
    "unload",
    "run_t2i",
    "run_i2i",
    "run_inpaint",
    "make_generator",
    "DEFAULT_STEPS",
    "DEFAULT_GUIDANCE_SCALE",
    "get_runtime_config",
    "lock",
]

# プロセス内シングルトン(core.registry.registry に登録)。
# exclusive_with=["qwen_image", "flux2"]: このファミリーをロードすると、qwen_image /
# flux2 のいずれかが常駐していれば registry.load() が自動 unload してからロードする
# (逆方向: qwen_image または flux2 をロードする際も z_image が常駐していれば自動
# unload される。register() 1回でこの両方向が有効になる、上記モジュールdocstring参照)。
family = ZImageFamily()
registry.register(family, exclusive_with=["qwen_image", "flux2"])
