# -*- coding: utf-8 -*-
"""
環境変数の一元定義。

新名は `DS_*`(diffusers-server の略)に統一しつつ、旧実装(Ｑwenimage-edit-diffusers /
flux2_diffusers)で使われていた環境変数名も後方互換で読めるようにする。

抽出元:
  - Ｑwenimage-edit-diffusers/pipeline_manager.py
    - COMFYUI_DIR (行50), RuntimeConfig とその環境変数名一覧 (行255-259, 291-332)
    - QWENIMG_VRAM_FREE_THRESHOLD_GB / QWENIMG_VRAM_LOW_THRESHOLD_GB /
      QWENIMG_GGUF_VRAM_FREE_THRESHOLD_GB (行262-269)
  - flux2_diffusers/pipeline_manager.py
    - Flux2Config とその環境変数名一覧 (行81-116)

方針:
  - 「新名を優先し、新名が無ければ旧名にフォールバックする」ヘルパー `env_str` / `env_bool` /
    `env_float` を提供する。ファミリー別の RuntimeConfig(Phase 1/2 で core を使う側に実装)は、
    このヘルパー経由で値を取得すれば、新名・旧名どちらの環境変数でも動く。
  - PYTORCH_CUDA_ALLOC_CONF は、Ｑwenimage-edit-diffusers の CLAUDE.md 知見(8番)どおり
    "expandable_segments:True" を既定にする。ただし起動スクリプト等で既に設定済みの場合は
    上書きしない(ユーザー/運用側の明示指定を尊重する)。
  - 元実装の「勝手な改良はしない」方針に従い、しきい値のデフォルト数値は
    Ｑwenimage-edit-diffusers の実測値をそのまま踏襲する(新環境 RTX PRO 5000 48GB 専有でも、
    §4.5 の対照表に基づき Phase 1/2 で個別に見直す前提であり、Phase 0 では変更しない)。
"""
import os

# ============================================================================
# CUDA アロケータ設定(モジュール import 時に一度だけ適用。CUDA 初期化は伴わない)
# ============================================================================
def _ensure_expandable_segments() -> None:
    """PYTORCH_CUDA_ALLOC_CONF に expandable_segments:True が含まれるようにする。

    既に環境変数が設定されている場合は上書きしない(運用側の明示設定を尊重する)。
    torch.cuda の初期化は行わない(os.environ の設定のみなので副作用は軽微)。

    注: torch 2.9(このプロジェクトの venv で使用)では PYTORCH_CUDA_ALLOC_CONF は
    非推奨になっており、代わりに PYTORCH_ALLOC_CONF を使うよう警告が出る(動作はする)。
    元実装(Ｑwenimage-edit-diffusers CLAUDE.md 8番)が明示的に PYTORCH_CUDA_ALLOC_CONF を
    指定しているため後方互換のためこちらも設定しつつ、新しい変数名 PYTORCH_ALLOC_CONF
    も同じ値で設定して警告を避ける(両方とも既存の値があれば上書きしない)。
    """
    existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if not existing:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if not os.environ.get("PYTORCH_ALLOC_CONF"):
        os.environ["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"]


_ensure_expandable_segments()


# ============================================================================
# 新旧環境変数名の対応表
# ============================================================================
# 新名(DS_*) -> 旧名候補のタプル(優先順)。旧名は Qwen 系(QWENIMG_*)/ FLUX2 系(FLUX2_*)
# の両方を含みうる。DS_* が設定されていればそれを最優先し、無ければ旧名を先勝ちで探す。
_LEGACY_ALIASES = {
    "DS_COMFYUI_DIR": ("COMFYUI_DIR",),
    "DS_OFFLOAD": ("QWENIMG_OFFLOAD", "FLUX2_OFFLOAD", "ZIMAGE_OFFLOAD"),
    "DS_ATTN": ("QWENIMG_ATTN", "FLUX2_ATTN", "FLUX2_ATTN_BACKEND", "ZIMAGE_ATTN"),
    "DS_COMPILE": ("QWENIMG_COMPILE", "FLUX2_COMPILE", "ZIMAGE_COMPILE"),
    "DS_DEVICE": ("QWENIMG_DEVICE", "FLUX2_DEVICE", "ZIMAGE_DEVICE"),
    "DS_QUANT": ("QWENIMG_QUANT",),
    "DS_T2I_MODEL": ("QWENIMG_T2I_MODEL",),
    "DS_LAYERED_QUANT": ("QWENIMG_LAYERED_QUANT",),
    "DS_VRAM_FREE_THRESHOLD_GB": ("QWENIMG_VRAM_FREE_THRESHOLD_GB",),
    "DS_VRAM_LOW_THRESHOLD_GB": ("QWENIMG_VRAM_LOW_THRESHOLD_GB",),
    "DS_GGUF_VRAM_FREE_THRESHOLD_GB": ("QWENIMG_GGUF_VRAM_FREE_THRESHOLD_GB",),
    "DS_FLUX2_PRECISION": ("FLUX2_PRECISION",),
    "DS_FLUX2_GROUP_STREAM": ("FLUX2_GROUP_STREAM",),
    "DS_EDIT_TE_OFFLOAD": (),
    "DS_ZIMAGE_PRECISION": ("ZIMAGE_PRECISION",),
    "DS_ZIMAGE_VAE_TILING": ("ZIMAGE_VAE_TILING",),
    "DS_ZIMAGE_GROUP_STREAM": ("ZIMAGE_GROUP_STREAM",),
}


def env_str(new_name: str, default: str = "") -> str:
    """新名(DS_*)を優先し、無ければ登録済みの旧名を先勝ちで探し、それも無ければ default。"""
    value = os.environ.get(new_name)
    if value is not None:
        return value
    for legacy_name in _LEGACY_ALIASES.get(new_name, ()):
        value = os.environ.get(legacy_name)
        if value is not None:
            return value
    return default


def env_bool(new_name: str, default: bool = False) -> bool:
    """"1"/"true"/"yes"/"on"(大文字小文字を問わない)を True とみなす。

    抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py の _env_bool() (行284-288)。
    """
    value = env_str(new_name, "")
    if value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float(new_name: str, default: float) -> float:
    value = env_str(new_name, "")
    if value == "":
        return default
    return float(value)


# ============================================================================
# ComfyUI モデルディレクトリ(resolve.py が参照)
# ============================================================================
# 特定ユーザー名を含む絶対パスをハードコードしないため、ホームディレクトリ相対で解決する
# (Ｑwenimage-edit-diffusers/pipeline_manager.py 行49-51 を踏襲)。
COMFYUI_DIR = env_str("DS_COMFYUI_DIR", os.path.expanduser("~/ComfyUI"))
COMFYUI_MODELS_DIR = os.path.join(COMFYUI_DIR, "models")


# ============================================================================
# VRAM しきい値(gpu.py / optimize.py が参照)
# ============================================================================
# 元実装(Ｑwenimage-edit-diffusers)の既定値 65.0/28.0/35.0 は RTX PRO 6000 Blackwell 96GB
# (共有、実測空き ~45GB)基準だった。この環境ではその空き ~45GB が「まだ厳しい」水準として
# 扱われ、bf16 transformer 系は "group" オフロード判定になっていた。
#
# Phase 1(diffusers-server, RTX PRO 5000 Blackwell 48GB 専有、実測空き ~47GB)向けに既定値を
# 見直す(REBUILD_PLAN §4 Phase 1 申し送り事項、§4.5 の48GB適合表: T2I bnb4bit 25.8GB /
# Layered fp8-lightning 42.2GB / Edit fp8-lightning 43.2GB / ControlNet+fp8-lightning 43.6GB /
# GGUF Q4_K_M系 37-39GB は全て「専有45GB前後」に収まり、オフロードなし("none")で GPU 常駐
# 可能と実測されている)。旧値のままだと空き~47GBは新しい VRAM_FREE_THRESHOLD_GB 未満のため
# "group" 判定になり、48GB専有では不要なはずのオフロードが誤って有効化されてしまう
# (fp8-lightning/GGUF経路は transformer 自体は明示的に "none" 固定で読むため直接の
# 影響は無いが、共有コンポーネント配置・非量子化bf16経路の判定には影響するため実測に基づき
# 調整する)。
#
# 新既定値の根拠: 「none」判定は実測ピークが空きVRAMに収まる下限(~44GB付近、Edit
# fp8-lightning 43.2GB / ControlNet 43.6GB に余裕を持たせた値)に、「group」との境界
# (LOW_THRESHOLD)はそれより十分小さいVRAM(24GB級カード相当)でのみ group_lowvram に
# 落とすよう、旧実装の相対関係(free >= low の間は"group")を維持しつつ引き下げる。
# GGUF は元々 bf16 の約1/3(12-13GB)で GGUF_VRAM_FREE_THRESHOLD_GB を旧実装より下げても
# 実測(37-39GB)には影響しない安全域を確保する。
#
# 環境変数 DS_VRAM_FREE_THRESHOLD_GB 等(旧名 QWENIMG_VRAM_FREE_THRESHOLD_GB 等も後方互換で
# 読める)で明示指定すれば、旧環境向けの値(65.0/28.0/35.0)にいつでも戻せる。
VRAM_FREE_THRESHOLD_GB = env_float("DS_VRAM_FREE_THRESHOLD_GB", 44.0)
VRAM_LOW_THRESHOLD_GB = env_float("DS_VRAM_LOW_THRESHOLD_GB", 20.0)
GGUF_VRAM_FREE_THRESHOLD_GB = env_float("DS_GGUF_VRAM_FREE_THRESHOLD_GB", 30.0)


# ============================================================================
# sm_120(Blackwell)attention backend 知見(optimize.py が参照)
# ============================================================================
# Ｑwenimage-edit-diffusers/pipeline_manager.py 行272-281 と
# flux2_diffusers/pipeline_manager.py 行54-70 で共通して確認済みの知見:
# sage 系カーネルは sm_120 未対応(実行時に "no kernel image is available" で失敗)。
KNOWN_GOOD_ATTN_BACKENDS = (
    "default",
    "native",
    "_native_flash",
    "_native_cudnn",
    "_native_efficient",
    "flex",
    "xformers",
)
KNOWN_BAD_ATTN_BACKEND_PREFIXES = ("sage", "_sage")


def get_offload_mode_raw(default: str = "auto") -> str:
    """DS_OFFLOAD(旧 QWENIMG_OFFLOAD / FLUX2_OFFLOAD)の生値を返す。正規化は optimize.py 側。"""
    return env_str("DS_OFFLOAD", default).strip().lower()


def get_attention_backend(default: str = "default") -> str:
    """DS_ATTN(旧 QWENIMG_ATTN / FLUX2_ATTN / FLUX2_ATTN_BACKEND)の生値を返す。"""
    return env_str("DS_ATTN", default).strip().lower()


def get_compile_enabled(default: bool = False) -> bool:
    """DS_COMPILE(旧 QWENIMG_COMPILE / FLUX2_COMPILE)。"1"/"true"/"yes"/"on" を True とみなす。"""
    return env_bool("DS_COMPILE", default)


def get_device(default: str = "cuda") -> str:
    """DS_DEVICE(旧 QWENIMG_DEVICE / FLUX2_DEVICE)。"""
    return env_str("DS_DEVICE", default)


# ============================================================================
# ターミナル進捗バー(core/progress.py が参照、CLAUDE.md 55番)
# ============================================================================
# DS_TERMINAL_PROGRESS: "0"(既定、OFF=従来挙動どおり何も出力しない)/ "1"(ON)。
# ON時、core/progress.py の状態更新(start_loading/start_generating/update_step/
# set_phase/finish)に同期して、サーバ起動ターミナルの stderr へ 0.1〜0.2秒間隔で
# 1行プログレスバー(\r で上書き、finish時のみ確定行として改行)を描画する。
# 既定 OFF のままなら core/progress.py・各ファミリーの生成経路は一切変化しない
# (このフラグを読む分岐が追加されるのみで、フラグが False の間は早期returnする)。
TERMINAL_PROGRESS = env_bool("DS_TERMINAL_PROGRESS", False)


# ============================================================================
# Edit系: text_encoder CPU退避(families/qwen_image/edit.py 等が参照)
# ============================================================================
# DS_EDIT_TE_OFFLOAD: "auto"(既定)/ "on" / "off"。
#   auto: 主判定は要求解像度(height*width、未指定なら1024²相当とみなす)。
#     EDIT_TE_OFFLOAD_AREA_THRESHOLD 以上なら退避を有効にする。空きVRAMは
#     「現在の空き」であって「この生成が必要とするVRAM」の代理指標にはならないため、
#     判定を反転させる用途では使わない。空きVRAMが
#     EDIT_TE_OFFLOAD_FREE_VRAM_GB_THRESHOLD+16GB を大きく超える大容量VRAM環境
#     (72GB版等)でのみ「退避不要」のエスケープハッチとして働く
#     (core/optimize.py の should_offload_edit_text_encoder() 参照)。
#   on: 常に有効(charsheet 経路は実質これに相当する解像度で運用)。
#   off: 常に無効(旧挙動、共有コンポーネント常駐のまま)。
#
# 実測(RTX PRO 5000 Blackwell 48GB専有): Edit fp8-lightning 640² でピーク39GB
# (TE常駐込み)、edit_angles 8方向でピーク43.8GB。1024²は自動推定でVAE decodeの
# 一時ピークが5GBほど乗り、TE(~16GB)常駐のままだとOOMする実績があるため、
# 640²を境界の目安として EDIT_TE_OFFLOAD_AREA_THRESHOLD を設定する
# (640*640=409600px。これ以上の要求解像度ではTE退避を既定で有効にする)。
EDIT_TE_OFFLOAD_AREA_THRESHOLD = env_float("DS_EDIT_TE_OFFLOAD_AREA_THRESHOLD", float(640 * 640))
EDIT_TE_OFFLOAD_FREE_VRAM_GB_THRESHOLD = env_float("DS_EDIT_TE_OFFLOAD_FREE_VRAM_GB_THRESHOLD", 44.0)


def get_edit_te_offload_mode(default: str = "auto") -> str:
    """DS_EDIT_TE_OFFLOAD の生値を返す("auto" / "on" / "off")。正規化は呼び出し側で行う。"""
    return env_str("DS_EDIT_TE_OFFLOAD", default).strip().lower()


# ============================================================================
# charsheet "bf16-group" 方式(旧 /home/animede/charsheet 完全再現、CLAUDE.md 33番)
# ============================================================================
# 旧 charsheet/pipeline.py の CHARSHEET_GROUP_OFFLOAD_* 環境変数をそのまま DS_* 名で
# 引き継ぐ(旧名は後方互換で読める。_LEGACY_ALIASES には未登録のため、
# CHARSHEET_GROUP_OFFLOAD_* を直接使いたい場合は DS_CHARSHEET_GROUP_OFFLOAD_* を
# 明示的に設定すること。値の意味・既定値は旧実装の
# _apply_group_offload_to_transformer()(pipeline.py 行282-334)と同一)。
DS_CHARSHEET_GROUP_OFFLOAD_BLOCKS = env_str("DS_CHARSHEET_GROUP_OFFLOAD_BLOCKS", "1")
DS_CHARSHEET_GROUP_OFFLOAD_USE_STREAM = env_bool("DS_CHARSHEET_GROUP_OFFLOAD_USE_STREAM", True)
DS_CHARSHEET_GROUP_OFFLOAD_NON_BLOCKING = env_str("DS_CHARSHEET_GROUP_OFFLOAD_NON_BLOCKING", "")
DS_CHARSHEET_GROUP_OFFLOAD_RECORD_STREAM = env_bool("DS_CHARSHEET_GROUP_OFFLOAD_RECORD_STREAM", False)
DS_CHARSHEET_GROUP_OFFLOAD_LOW_CPU_MEM = env_bool("DS_CHARSHEET_GROUP_OFFLOAD_LOW_CPU_MEM", True)
DS_CHARSHEET_GROUP_OFFLOAD_DISK_PATH = env_str("DS_CHARSHEET_GROUP_OFFLOAD_DISK_PATH", "")

# ロード前のホスト RAM 空き(GB)チェック閾値。transformer bf16(~38GB)を CPU にピン留め
# する設計のため、これを下回る場合は明確なエラーで中止する(黙ってスワップに突入して
# システムフリーズを起こした過去の事故(CLAUDE.md 17番)を防ぐガード)。
DS_CHARSHEET_GROUP_OFFLOAD_MIN_RAM_GB = env_float("DS_CHARSHEET_GROUP_OFFLOAD_MIN_RAM_GB", 42.0)
