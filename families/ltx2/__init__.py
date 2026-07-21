# -*- coding: utf-8 -*-
"""
LTX-2.3(動画+音声生成)ファミリー(T2V / I2V / FLF)。

FLF(First-Last-Frame、2026-07-19追加): 最初と最後のフレーム画像を指定して間の動画を生成する
モード。LTX2ConditionPipeline を base の components から遅延構築する(I2V と同じ参照共有
パターン)。詳細は pipeline.py の get_flf_pipeline() / generate.py の run_flf() を参照。

keyframes(任意キーフレーム条件付け、2026-07-20追加): FLF の一般化。任意枚数の画像を
任意フレーム位置(indexは負値で末尾からの指定も可)・任意strengthで条件付ける。
ロードは flf と同一の get_flf_pipeline() を流用する(追加モデルDL不要)。
詳細は generate.py の run_keyframes() を参照。

V2A(Video to Audio、IA2Vの鏡像、2026-07-20追加): 既存動画に合った音声を生成して付与する。
IA2V(run_ia2v())が「音声latentを入力wav由来のclean latentへx0固定し、映像側だけを
denoiseする」のに対し、V2Aは「映像latentを入力動画由来のclean latentへx0固定し、
音声latentだけをdenoiseする」逆構成。ロードは ia2v/flf と同一の get_flf_pipeline() を
流用する(追加モデルDL不要)。詳細は generate.py の run_v2a() を参照。

IC-LoRA(In-Context LoRA、MergeGreen 動画編集、2026-07-20追加): 参照動画の中央部を
緑 RGB(0,191,0) でマスクし、プロンプトで指示した変化に沿って動画を生成し直す
動画編集モード(HF repo siraxe/MergeGreen_IC-lora_ltx2.3、Apache-2.0)。
LTX2InContextPipeline(diffusers本体、reference_conditions機構)+ MergeGreen LoRA
アダプタ(adapter方式、fuseせず、strength既定0.9)。詳細は pipeline.py の
get_iclora_pipeline()・generate.py の run_iclora() を参照。

tools/ltx2_regress.py(932行、コミット済みのスタンドアロン回帰確認スクリプト)で
実証済みの ComfyUI形式 scaled-fp8(28GB単一ファイル)streaming ロード + diffusers 2.3
非対応キー変換パッチ 4件を、他ファミリー(z_image 等)と同じ6モジュール構成へ整理して
統合したもの。

サブモジュール構成:
  runtime.py    - LTX2RuntimeConfig(環境変数 DS_LTX2_*)
  state.py      - シングルトン状態(ロック・pipeline_state群)
  convert.py    - ComfyUI scaled-fp8 -> diffusers コンポーネント別 state_dict 変換
                  (dequant、VAE 4段デコーダパッチ、prompt_adaln リネーム、
                  connector 振り分け、vocoder リネーム、Gemma キー変換)
  pipeline.py   - LTX2Pipeline ロード(none/group offload 適用)+ I2V 遅延構築
  generate.py   - 生成本体(T2V/I2V、音声wav保存+ffmpeg mux)
  lifecycle.py  - unload() / get_status()
  family.py     - LTX2Family(core.registry.ModelFamily の実装)

このパッケージを import した時点で core.registry.registry にシングルトンとして自動登録する
(app.py 起動時に import families.ltx2 するだけでよい)。

排他構成: qwen_image / flux2 / z_image / ltx2 の4ファミリーはいずれも VRAM 同時常駐
不可とみなし、exclusive_with=["qwen_image", "flux2", "z_image"] で4方向すべてを
相互排他にする(FamilyRegistry.register() 呼び出し1回で全方向が有効になる、
families/z_image/__init__.py と同じ仕組み。qwen_image/flux2/z_image 側の register()
呼び出しを変更する必要はない)。

48GB VRAM 対応(タスク指示): DS_LTX2_OFFLOAD="group" で transformer(bf16デクオンタイズ後
~35.4GB)のみ block-level group offload し、他コンポーネント(text_encoder/connectors/
vae類)は GPU 常駐のままにする(families/ltx2/pipeline.py・runtime.py 参照)。
"""
from core.registry import registry

from families.ltx2.family import LTX2Family
from families.ltx2.generate import (
    OUTPUTS_DIR,
    make_generator,
    run_flf,
    run_i2v,
    run_ia2v,
    run_iclora,
    run_keyframes,
    run_t2v,
    run_v2a,
)
from families.ltx2.lifecycle import get_status, unload
from families.ltx2.runtime import (
    DEFAULT_AUDIO_OUT,
    DEFAULT_FLF_FRAME_RATE,
    DEFAULT_FLF_STRENGTH,
    DEFAULT_FPS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_IA2V_AUDIO_OUT,
    DEFAULT_IA2V_FRAME_RATE,
    DEFAULT_IA2V_NEGATIVE_PROMPT,
    DEFAULT_IA2V_PROMPT,
    DEFAULT_ICLORA_MASK_COLOR,
    DEFAULT_ICLORA_NEGATIVE_PROMPT,
    DEFAULT_ICLORA_REF_STRENGTH,
    DEFAULT_ICLORA_STRENGTH,
    DEFAULT_NUM_FRAMES,
    DEFAULT_STEPS,
    DEFAULT_V2A_MAX_NUM_FRAMES,
    DEFAULT_V2A_NEGATIVE_PROMPT,
    DEFAULT_V2A_PROMPT,
    DEFAULT_WIDTH,
)
from families.ltx2.state import get_runtime_config, lock

__all__ = [
    "LTX2Family",
    "family",
    "get_status",
    "unload",
    "run_t2v",
    "run_i2v",
    "run_flf",
    "run_ia2v",
    "run_keyframes",
    "run_v2a",
    "run_iclora",
    "make_generator",
    "OUTPUTS_DIR",
    "DEFAULT_STEPS",
    "DEFAULT_GUIDANCE_SCALE",
    "DEFAULT_WIDTH",
    "DEFAULT_HEIGHT",
    "DEFAULT_NUM_FRAMES",
    "DEFAULT_FPS",
    "DEFAULT_FLF_STRENGTH",
    "DEFAULT_FLF_FRAME_RATE",
    "DEFAULT_IA2V_FRAME_RATE",
    "DEFAULT_IA2V_PROMPT",
    "DEFAULT_IA2V_NEGATIVE_PROMPT",
    "DEFAULT_AUDIO_OUT",
    "DEFAULT_IA2V_AUDIO_OUT",
    "DEFAULT_V2A_PROMPT",
    "DEFAULT_V2A_NEGATIVE_PROMPT",
    "DEFAULT_V2A_MAX_NUM_FRAMES",
    "DEFAULT_ICLORA_STRENGTH",
    "DEFAULT_ICLORA_REF_STRENGTH",
    "DEFAULT_ICLORA_MASK_COLOR",
    "DEFAULT_ICLORA_NEGATIVE_PROMPT",
    "get_runtime_config",
    "lock",
]

# プロセス内シングルトン(core.registry.registry に登録)。
family = LTX2Family()
registry.register(family, exclusive_with=["qwen_image", "flux2", "z_image"])
