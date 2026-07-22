# -*- coding: utf-8 -*-
"""
Z-Image ファミリーのランタイム構成(環境変数)。

抽出元: /home/animede/Z-image-diffusers/pipeline_manager.py の ZImageConfig クラス
(行101-151)。

環境変数名は DS_* に統一しつつ、旧実装(Z-image-diffusers)で使われていた ZIMAGE_* も
後方互換で読めるようにする(core.config._LEGACY_ALIASES に追記済み)。

新環境(RTX PRO 5000 48GB 専有)向けの既定値の考え方: 旧実装(RTX PRO 6000 96GB)は
offload="none"(全常駐、bf16で~20GB)を既定にしていた。この環境でも Z-Image-Turbo
bf16(~20GB)は48GB専有に十分収まる見込みのため、既定は "none" のまま維持する
(タスク指示33番: 「transformerを丸ごとCPUへスワップする実装は禁止。48GBに収まるので
全GPU常駐でよい」)。
"""
from core.config import env_bool, env_str

DEFAULT_PRECISION = "bf16"
DEFAULT_OFFLOAD = "none"


class ZImageRuntimeConfig:
    """環境変数から読み取るランタイム構成(Z-image-diffusers ZImageConfig を踏襲)。

    DS_ZIMAGE_PRECISION(旧 ZIMAGE_PRECISION) : "bf16"(既定、フル精度・~20GB) |
                       "bnb-4bit"(bitsandbytes 4bit量子化、低VRAM・品質/速度トレードオフ)
    DS_OFFLOAD(旧 ZIMAGE_OFFLOAD)       : "none"(既定、全常駐。48GB専有では不要な限り
                       これを維持すること、タスク指示33番) | "model"(alias
                       "model_cpu_offload") | "group"(alias "group_offload")
    DS_ATTN(旧 ZIMAGE_ATTN)             : attention backend 名
    DS_COMPILE(旧 ZIMAGE_COMPILE)       : "0"(既定) | "1"
    DS_ZIMAGE_VAE_TILING(旧 ZIMAGE_VAE_TILING) : "0"(既定) | "1"
    DS_DEVICE(旧 ZIMAGE_DEVICE)         : "cuda"(既定)
    DS_ZIMAGE_GROUP_STREAM(旧 ZIMAGE_GROUP_STREAM) : "1"(既定、CUDA-stream prefetch)| "0"
    """

    def __init__(self):
        self.precision = env_str("DS_ZIMAGE_PRECISION", DEFAULT_PRECISION).strip().lower()
        self.offload = env_str("DS_OFFLOAD", DEFAULT_OFFLOAD).strip().lower()
        self.attention_backend = env_str("DS_ATTN", "default").strip().lower()
        self.compile = env_bool("DS_COMPILE", False)
        self.vae_tiling = env_bool("DS_ZIMAGE_VAE_TILING", False)
        self.device = env_str("DS_DEVICE", "cuda")
        self.group_stream = env_bool("DS_ZIMAGE_GROUP_STREAM", True)

    def __repr__(self):
        return (
            f"ZImageRuntimeConfig(precision={self.precision!r}, offload={self.offload!r}, "
            f"attention_backend={self.attention_backend!r}, compile={self.compile}, "
            f"vae_tiling={self.vae_tiling}, device={self.device!r}, group_stream={self.group_stream})"
        )
