# -*- coding: utf-8 -*-
"""
charsheet の生成呼び出し(旧 charsheet/pipeline.py の生成部分を置き換え)。

方式切替(DS_CHARSHEET_METHOD): charsheet(8方向キャラシート生成)はこれまで複数回の
目視比較を経て方式を変遷させてきた:
  1. プロンプトのみ方式(Multiple-angles LoRA 廃止) -> 品質が旧LoRA版に劣ると判定
  2. fp8-lightning-angles 変種(bf16 -> Lightning fuse -> Multiple-angles fuse ->
     fp8 layerwise cast)を復活 -> 「back のみ衣装が変わる」等、旧 charsheet に対する
     微妙な劣化が残る
  3. bf16-adapter 変種(旧 /home/animede/charsheet 方式への回帰、旧既定): fp8ストレージ化
     は fuse済みLoRA差分を量子化グリッドで丸めるため(コミット 9dbb0ab で判明した機構)、
     角度LoRAの効きが劣化している可能性がある。bf16のまま Lightning + Multiple-angles を
     adapter として適用するが、ベースは fp8_e4m3fn へストレージ圧縮する
     (families/qwen_image/edit_angles_bf16.py の「fp8-base-adapters」実装)。
  4. bf16-group 変種(本追加、新既定、CLAUDE.md 33番): 3.のfp8ストレージ化さえも
     わずかな精度劣化要因になりうるとの判断から、ベースも一切量子化せず純粋な bf16 の
     まま常駐させ、VRAM は block-level group offloading(transformerをブロック単位で
     GPU/CPU入れ替え)で賄う。旧 /home/animede/charsheet/pipeline.py の "group" モード
     (_apply_group_offload_to_transformer()、行282-334)と数値的に完全同一の構成
     (LoRA適用: load_lora_weights ×2 -> set_adapters(["lightning","angles"],
     adapter_weights=[1.0, 1.0])、fuse・fp8化なし)。

環境変数 DS_CHARSHEET_METHOD で4方式を切替可能(既定 "bf16-group"):
  - "bf16-group"(既定、新規): families/qwen_image の Edit「bf16-group」変種
    (families/qwen_image/edit_angles_bf16group.py。mode="edit_angles_bf16group"、
    旧 /home/animede/charsheet 完全再現)。
  - "bf16-adapters"(旧既定): families/qwen_image の Edit「fp8-base-adapters」変種
    (families/qwen_image/edit_angles_bf16.py。mode="edit_angles_bf16")。
  - "fp8-fuse": families/qwen_image の Edit「fp8-lightning-angles」変種
    (families/qwen_image/edit_angles.py。mode="edit_angles")。
  - "prompt-only": Multiple-angles LoRA なしのプロンプトのみ方式
    (families/qwen_image の通常 Edit、fp8-lightning。mode="edit")。退行時の
    フォールバック用に維持。

旧環境変数 DS_CHARSHEET_USE_ANGLES_LORA との整合: DS_CHARSHEET_METHOD が明示されていない
場合のみ、DS_CHARSHEET_USE_ANGLES_LORA=0 を "prompt-only" として後方互換で解釈する
(DS_CHARSHEET_USE_ANGLES_LORA=1/未設定は新既定の "bf16-group" を使う。他方式へ
戻したい場合は DS_CHARSHEET_METHOD を明示すること)。

  - 前処理(preprocess_image: ComfyUI の ImageScaleToTotalPixels 1MP 相当)は
    families.qwen_image.preprocess_image をそのまま再利用する(旧実装と同一ロジック)。
  - 生成前の height/width は明示指定必須(Phase 1/2 申し送り事項): Edit は
    height/width 未指定だと参照画像から自動推定される1024²相当の解像度になり、
    48GB専有GPUでは VAE decode の一時ピークで必ず OOM する。640x640 は実測 39.0GB で
    成功済み。本実装では CHARSHEET_EDIT_SIZE 環境変数で上書き可能にしつつ、既定を
    1024 にする(TE CPU退避により1024²でもOOMなく収まることを実機確認済み)。
  - ジョブ内で8方向を連続生成する間は Edit(どの方式でも)をロードしたまま使う
    (こまめに unload しない)。
  - edit / edit_angles / edit_angles_bf16 / edit_angles_bf16group の排他切替は
    families/qwen_image/family.py 側で行われる(registry.load() 経由でロードする限り、
    いずれかをロードすると他は自動解放される。通常の /api/edit の挙動・重みには
    一切影響しない)。

生成の排他制御(GPU 同時1件)は呼び出し側(apps.charsheet.jobs)が
core.gpu.generation_lock を直接扱う(旧 charsheet の「同時1ジョブ」排他は
apps.charsheet.jobs 側の current_job_lock が担う。生成そのものは
registry.load()->family.generate() を直接呼ぶ)。
"""
import os
import time
from typing import Optional

from PIL import Image

from core import gpu
from core.registry import registry

import families.qwen_image as qwen_image

# 生成解像度。環境変数 CHARSHEET_EDIT_SIZE で上書き可能。
#
# 経緯: Phase 1/2 では共有 text_encoder(~16GB)常駐のまま1024²を生成するとVAE decodeの
# 一時ピークでOOMするため、既定を640に制限していた。その後の目視比較で「旧charsheet
# (1024²生成)よりまだ品質が劣る」と判定され、調査の結果640²への解像度制限が主因と
# 判断された。families/qwen_image 側にTE CPU退避(DS_EDIT_TE_OFFLOAD、既定auto)を
# 実装したことで、プロンプトエンコード後にTEをCPUへ退避 -> denoise/VAE decodeの間は
# VRAMを空けておける -> 1024²でもOOMなく収まることを実機確認したため、既定を1024に戻す
# (旧 charsheet と同じ解像度)。VRAMが逼迫する環境では引き続き CHARSHEET_EDIT_SIZE=640 等の
# 明示的な縮小が可能。
_DEFAULT_EDIT_SIZE = 1024

# DS_CHARSHEET_METHOD の有効値 <-> families/qwen_image の mode。
_METHOD_TO_MODE = {
    "bf16-group": "edit_angles_bf16group",
    "bf16-adapters": "edit_angles_bf16",
    "fp8-fuse": "edit_angles",
    "prompt-only": "edit",
}
_DEFAULT_METHOD = "bf16-group"


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def charsheet_method() -> str:
    """DS_CHARSHEET_METHOD(既定 "bf16-group"): "bf16-group" | "bf16-adapters" |
    "fp8-fuse" | "prompt-only" のいずれか。未知の値が指定された場合は既定にフォールバックする。

    後方互換: DS_CHARSHEET_METHOD が未設定で、旧 DS_CHARSHEET_USE_ANGLES_LORA=0 が
    明示されている場合のみ "prompt-only" として解釈する(それ以外は新既定を使う)。
    """
    raw = os.environ.get("DS_CHARSHEET_METHOD")
    if raw is None:
        if not _env_bool("DS_CHARSHEET_USE_ANGLES_LORA", True):
            return "prompt-only"
        return _DEFAULT_METHOD
    method = raw.strip().lower()
    if method not in _METHOD_TO_MODE:
        print(
            f"[apps.charsheet] warning: 未知の DS_CHARSHEET_METHOD={raw!r} を無視し、"
            f"既定の {_DEFAULT_METHOD!r} を使います。"
        )
        return _DEFAULT_METHOD
    return method


def use_angles_lora() -> bool:
    """後方互換用(charsheet_method() が "prompt-only" 以外なら True、
    つまり Multiple-angles LoRA を何らかの形で併用しているかどうか)。
    apps/charsheet/prompts.py がプロンプト文言選択に使う。
    """
    return charsheet_method() != "prompt-only"


def _edit_mode() -> str:
    return _METHOD_TO_MODE[charsheet_method()]


def _edit_size() -> int:
    try:
        return int(os.environ.get("CHARSHEET_EDIT_SIZE", str(_DEFAULT_EDIT_SIZE)))
    except ValueError:
        return _DEFAULT_EDIT_SIZE


def preprocess_image(image: Image.Image) -> Image.Image:
    """旧 charsheet/pipeline.py preprocess_image() 相当(families.qwen_image に委譲)。"""
    return qwen_image.preprocess_image(image)


def generate_view(
    image: Image.Image,
    prompt: str,
    seed: int = 0,
    negative_prompt: str = "",
    width: Optional[int] = None,
    height: Optional[int] = None,
    progress_extra: Optional[dict] = None,
) -> dict:
    """1方向分の画像を生成する。DS_CHARSHEET_METHOD(既定 "bf16-group")に応じて
    families/qwen_image の Edit「bf16-group」("bf16-group") / 「fp8-base-adapters」
    ("bf16-adapters") / 「fp8-lightning-angles」("fp8-fuse") /
    通常Edit・プロンプトのみ("prompt-only")のいずれかを使用する。

    呼び出し前に ensure_edit_loaded() 済みであること(ジョブ側で8方向まとめて
    1回だけロードする設計。ここでは load() を呼ばない)。

    戻り値は run_edit()/run_edit_angles()/run_edit_angles_bf16()/
    run_edit_angles_bf16group() のメタデータ dict(image_url, elapsed_s, peak_vram_gb 等)。
    画像自体は meta["image_url"](/outputs/配下)経由でファイルとして保存されるため、
    呼び出し側でそのファイルを読み込んでジョブディレクトリへコピーする。
    """
    size = _edit_size() if (width is None and height is None) else None
    mode = _edit_mode()
    req = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": 4,
        "cfg": 1.0,
        "seed": seed if seed and seed > 0 else -1,
        "lightning": True,
        "shift": None,
        "width": width if width is not None else size,
        "height": height if height is not None else size,
        "_images": [image],
        "_progress_extra": progress_extra,
    }
    family = registry.get("qwen_image")
    return family.generate({"mode": mode, **req})


def get_edit_pipeline_family_name() -> str:
    return "qwen_image"


def ensure_edit_loaded() -> None:
    """Edit グループ(DS_CHARSHEET_METHOD に応じて edit_angles_bf16group /
    edit_angles_bf16 / edit_angles / edit)をロードする(未ロードなら)。
    ジョブ開始前に1回だけ呼ぶ。

    registry.load() 経由で呼ぶことで、families/qwen_image/family.py 側の
    edit / edit_angles / edit_angles_bf16 相互排他ロジック(いずれかロード時に他は
    自動解放)が効く。
    """
    registry.load("qwen_image", _edit_mode())


def acquire_generation_lock(blocking: bool = True) -> bool:
    return gpu.generation_lock.acquire(blocking=blocking)


def release_generation_lock() -> None:
    gpu.generation_lock.release()


def get_load_info() -> dict:
    """旧 pipeline.get_load_info() 互換(現在使用中のグループの状態を返す)。"""
    from families.qwen_image import state

    method = charsheet_method()
    group = {
        "bf16-group": state.edit_angles_bf16group_group,
        "bf16-adapters": state.edit_angles_bf16_group,
        "fp8-fuse": state.edit_angles_group,
        "prompt-only": state.edit_group,
    }[method]
    return {
        "loaded": group.get("loaded", False),
        "quant": group.get("quant"),
        "fallback": group.get("fallback", False),
        "lightning_merged": group.get("lightning_merged"),
        "load_time_s": group.get("load_time_s"),
        "angles_lora": use_angles_lora(),
        "method": method,
    }
