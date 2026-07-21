# -*- coding: utf-8 -*-
"""
Qwen-Image 系ファミリーのランタイム構成(環境変数)。

抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py
  - RuntimeConfig とその環境変数名一覧(行291-332)
  - quant_suffix() / is_fp8_lightning_quant()(行335-352)

変更点: 環境変数の読み取りは core.config.env_str/env_bool 経由にした(DS_* 新名を優先し、
旧名 QWENIMG_* にフォールバック)。ロジック・既定値は無変更。

新環境(RTX PRO 5000 48GB 専有)向けの既定値調整(REBUILD_PLAN 申し送り事項):
  Edit/Layered の既定量子化を fp8-lightning にする(REBUILD_PLAN §2.4)。
  T2I の既定は "2512"(2026-07-18変更: 公式 Qwen/Qwen-Image-2512 bf16 + Lightning fuse +
  fp8 layerwise cast。旧既定 "2512-4bit"(コミュニティ製NF4)は品質劣位のため廃止、
  paths.py の T2I_2512_REPO コメント参照)。環境変数で明示指定すれば上書き可能。
"""
from core.config import env_bool, env_str

from families.qwen_image.paths import FP8_LIGHTNING_QUANT_VALUES, LAYERED_DEFAULT_QUANT


# 申し送り事項(REBUILD_PLAN §2.4, タスク1)+ 2026-07-18変更: Edit/Layered = fp8-lightning、
# T2I の既定は "2512"(公式 Qwen/Qwen-Image-2512 bf16 + Lightning fuse + fp8 cast。
# 旧既定 "2512-4bit"(コミュニティ製NF4)は廃止。旧値は後方互換で "2512" として扱う)。
DEFAULT_QUANT = "fp8-lightning"
DEFAULT_T2I_MODEL = "2512"


class RuntimeConfig:
    """環境変数から読み取るランタイム構成。

    DS_OFFLOAD(旧 QWENIMG_OFFLOAD) : "auto"(既定)/ "none" / "group" / "group_lowvram" / "model_cpu"
    DS_ATTN(旧 QWENIMG_ATTN)       : "default"(既定)/ "native" / "_native_flash" / "_native_cudnn" /
                       "_native_efficient" / "flex" / "xformers"
    DS_COMPILE(旧 QWENIMG_COMPILE) : "0"(既定)/ "1"
    DS_DEVICE(旧 QWENIMG_DEVICE)   : "cuda"(既定)
    DS_QUANT(旧 QWENIMG_QUANT)     : "fp8-lightning"(このファミリーの既定。旧実装は"none")/
                       "gguf-q4_k_m" 等 / "none"(bf16のまま)
    DS_T2I_MODEL(旧 QWENIMG_T2I_MODEL) : "2512"(このファミリーの既定。公式2512 fp8-lightning fuse)/
                       "qwen-image"(旧値 "2512-4bit" は後方互換で "2512" 扱い)
    DS_LAYERED_QUANT(旧 QWENIMG_LAYERED_QUANT) : "fp8-lightning"(既定)
    """

    def __init__(self):
        self.offload = env_str("DS_OFFLOAD", "auto").strip().lower()
        self.attention_backend = env_str("DS_ATTN", "default").strip().lower()
        self.compile = env_bool("DS_COMPILE", False)
        self.device = env_str("DS_DEVICE", "cuda")
        self.quant = env_str("DS_QUANT", DEFAULT_QUANT).strip().lower()
        self.t2i_model = env_str("DS_T2I_MODEL", DEFAULT_T2I_MODEL).strip().lower()
        self.layered_quant = env_str("DS_LAYERED_QUANT", LAYERED_DEFAULT_QUANT).strip().lower()

    def __repr__(self):
        return (
            f"RuntimeConfig(offload={self.offload!r}, attention_backend={self.attention_backend!r}, "
            f"compile={self.compile}, device={self.device!r}, quant={self.quant!r}, "
            f"t2i_model={self.t2i_model!r}, layered_quant={self.layered_quant!r})"
        )


def quant_suffix(quant: str) -> "str | None":
    """DS_QUANT の値からGGUFファイル名サフィックス(例: 'gguf-q4_k_m' -> 'Q4_K_M')を得る。
    'none'/空/'bf16'/'fp8-lightning' なら None(GGUF経路ではない)を返す。

    抽出元: pipeline_manager.py quant_suffix()(行335-346)。ロジック無変更。
    """
    quant = (quant or "none").strip().lower()
    if quant in ("", "none", "off", "bf16", "fp8") or quant in FP8_LIGHTNING_QUANT_VALUES:
        return None
    if quant.startswith("gguf-"):
        quant = quant[len("gguf-"):]
    elif quant.startswith("gguf_"):
        quant = quant[len("gguf_"):]
    return quant.upper()


def is_fp8_lightning_quant(quant: str) -> bool:
    """DS_QUANT が fp8-lightning(マージ済み高速化オプション)かどうか。"""
    return (quant or "none").strip().lower() in FP8_LIGHTNING_QUANT_VALUES
