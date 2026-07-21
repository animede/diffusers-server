# -*- coding: utf-8 -*-
"""
JoyAI-Image-Edit-Plus ファミリー(複数参照画像編集、Edit のみ)。

モデル: jdopensource/JoyAI-Image-Edit-Plus-Diffusers(Apache 2.0)。16B transformer
(bf16 ~32.5GB)+ Qwen3-VL-8B text_encoder(bf16 ~17.5GB)+ AutoencoderKLWan(~0.25GB)。
`JoyImageEditPlusPipeline` は venv の diffusers 0.40.0.dev0 に実装済み。

ステップ①(tools/joyai_regress.py・tools/joyai_speed_probe.py)の回帰確認・統合GO判定を
経て正式統合したもの(CLAUDE.md 46番: sm_120 パッチ化Conv3d病的低速の特定と対処)。

サブモジュール構成:
  runtime.py    - JoyAIRuntimeConfig(環境変数 DS_JOYAI_TE_OFFLOAD)
  state.py      - シングルトン状態(ロック・pipeline_state)
  pipeline.py   - JoyImageEditPlusPipeline のロード(RAM安全: transformer/text_encoder を
                  device_map="cuda" で個別ロード)+ PatchifyLinear ランタイムパッチ +
                  text_encoder CPU退避(48GB対応)
  generate.py   - 生成本体(Edit、1〜3枚の参照画像)
  lifecycle.py  - unload() / get_status()
  family.py     - JoyAIFamily(core.registry.ModelFamily の実装)

このパッケージを import した時点で core.registry.registry にシングルトンとして自動登録する
(app.py 起動時に import families.joyai するだけでよい)。

排他構成: qwen_image / flux2 / z_image / ltx2 / joyai の5ファミリーはいずれも VRAM
同時常駐不可とみなし、exclusive_with=["qwen_image", "flux2", "z_image", "ltx2"] で
5方向すべてを相互排他にする(FamilyRegistry.register() 呼び出し1回で全方向が有効になる、
families/ltx2/__init__.py と同じ仕組み。他ファミリー側の register() 呼び出しを変更する
必要はない)。

48GB対応(CLAUDE.md 47番参照): DS_JOYAI_TE_OFFLOAD(既定 "auto"、実質常時有効)で
Qwen3-VL text_encoder(~17.5GB)をプロンプトエンコード後にCPUへ退避する。常駐は
transformer(~32.5GB)+ vae等の小物のみとなり、48GB専有に収まる設計
(families/joyai/pipeline.py 参照)。
"""
from core.registry import registry

from families.joyai.family import JoyAIFamily
from families.joyai.generate import make_generator, run_edit
from families.joyai.lifecycle import get_status, unload
from families.joyai.pipeline import DEFAULT_GUIDANCE_SCALE, DEFAULT_STEPS, MAX_REF_IMAGES
from families.joyai.state import get_runtime_config, lock

__all__ = [
    "JoyAIFamily",
    "family",
    "get_status",
    "unload",
    "run_edit",
    "make_generator",
    "DEFAULT_STEPS",
    "DEFAULT_GUIDANCE_SCALE",
    "MAX_REF_IMAGES",
    "get_runtime_config",
    "lock",
]

# プロセス内シングルトン(core.registry.registry に登録)。
family = JoyAIFamily()
registry.register(family, exclusive_with=["qwen_image", "flux2", "z_image", "ltx2"])
