# -*- coding: utf-8 -*-
"""
FLUX.2 ファミリーのランタイム構成(環境変数)。

抽出元: flux2_diffusers/pipeline_manager.py の Flux2Config クラス(行81-116)。

新環境(RTX PRO 5000 48GB 専有)向けの既定値の考え方(REBUILD_PLAN タスク2申し送り):
  bnb-4bit transformer(~18GB)+ Mistral3 系 text_encoder は合計しても 48GB 専有下で
  GPU 常駐("none")が可能な見込み(タスク1の回帰確認で実測: model_cpu_offload 使用時で
  ピーク18.31GB。GPU占有なので offload="none" にすれば更に高速化できる可能性が高い)。
  ただし旧実装は "model_cpu_offload" を安全側の既定にしていたため、このファイルでも
  既定値は "model_cpu_offload" のまま維持し、実測後に Phase 2 検証タスクで offload="none"
  へ既定変更するかどうかを判断する(§4.5 の「推奨構成はすべて検証可能」に基づき、
  実測結果は検証セクションで報告する)。DS_OFFLOAD=none で明示上書き可能。

環境変数名は DS_* に統一しつつ、core.config._LEGACY_ALIASES 経由で旧名
(FLUX2_PRECISION / FLUX2_OFFLOAD / FLUX2_ATTN / FLUX2_ATTN_BACKEND / FLUX2_COMPILE /
FLUX2_DEVICE / FLUX2_GROUP_STREAM)も後方互換で読める。
"""
from core.config import env_bool, env_str

# flux2 固有の精度切替は DS_FLUX2_PRECISION(旧 FLUX2_PRECISION)専用キーを使う
# (DS_QUANT は Qwen 系の GGUF/fp8-lightning 切替と意味が異なるため共用しない)。
DEFAULT_PRECISION = "bnb-4bit"
DEFAULT_OFFLOAD = "model_cpu_offload"


class Flux2RuntimeConfig:
    """環境変数から読み取るランタイム構成(flux2_diffusers Flux2Config を踏襲)。

    DS_FLUX2_PRECISION(旧 FLUX2_PRECISION) : "bnb-4bit"(既定)| "bf16"
                       (bf16 は gated repo アクセスが必要。かつこの環境では bf16 フル精度版
                       は削除済み・再DL禁止のため事実上使用不可)
    DS_OFFLOAD(旧 FLUX2_OFFLOAD)       : "model_cpu_offload"(既定、alias "model") |
                       "none"(全常駐、最速)| "group"(alias "group_offload"、最省VRAM)
    DS_ATTN(旧 FLUX2_ATTN / FLUX2_ATTN_BACKEND) : attention backend 名
    DS_COMPILE(旧 FLUX2_COMPILE)       : "0"(既定)| "1"
    DS_DEVICE(旧 FLUX2_DEVICE)         : "cuda"(既定)
    DS_FLUX2_GROUP_STREAM(旧 FLUX2_GROUP_STREAM) : "1"(既定、CUDA-stream prefetch)| "0"
    """

    def __init__(self):
        self.precision = env_str("DS_FLUX2_PRECISION", DEFAULT_PRECISION).strip().lower()
        self.offload = env_str("DS_OFFLOAD", DEFAULT_OFFLOAD).strip().lower()
        self.attention_backend = env_str("DS_ATTN", "default").strip().lower()
        self.compile = env_bool("DS_COMPILE", False)
        self.device = env_str("DS_DEVICE", "cuda")
        self.group_stream = env_bool("DS_FLUX2_GROUP_STREAM", True)

    def __repr__(self):
        return (
            f"Flux2RuntimeConfig(precision={self.precision!r}, offload={self.offload!r}, "
            f"attention_backend={self.attention_backend!r}, compile={self.compile}, "
            f"device={self.device!r}, group_stream={self.group_stream})"
        )
