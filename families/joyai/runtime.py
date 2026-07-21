# -*- coding: utf-8 -*-
"""
JoyAI-Image-Edit-Plus ファミリーのランタイム構成(環境変数)。

抽出元: tools/joyai_regress.py(ステップ①の回帰確認スクリプト)。families/z_image・
families/ltx2 と同じ「runtime.py に環境変数を集約する」規約に合わせて新設した。

DS_JOYAI_TE_OFFLOAD(既定 "auto"): Qwen3-VL text_encoder(bf16 ~17.5GB)を
プロンプトエンコード後に CPU へ退避するかどうか。
  - "auto"(既定): 48GB専有を想定し常に有効にする(常駐すると transformer(~32.5GB)+
    text_encoder(~17.5GB)で ~50GB となり48GB専有では収まらないため、"auto" は
    実質「常に on」。96GB機など大容量VRAM環境でも退避コスト自体は小さい
    (エンコード時だけの短い窓、CLAUDE.md 39番のIA2V事故のような「丸ごと片道スワップ」
    ではなく毎回のGPU復帰を伴う往復のため、常時オンでも安全側)。
  - "on": 常に有効。
  - "off": 無効(常駐のまま。96GB機で最速を狙う場合のみ推奨)。
"""
from core.config import env_str

DEFAULT_TE_OFFLOAD = "auto"


class JoyAIRuntimeConfig:
    """環境変数から読み取るランタイム構成。

    DS_JOYAI_TE_OFFLOAD : "auto"(既定) | "on" | "off"
    DS_DEVICE           : "cuda"(既定)
    """

    def __init__(self):
        self.te_offload = env_str("DS_JOYAI_TE_OFFLOAD", DEFAULT_TE_OFFLOAD).strip().lower()
        self.device = env_str("DS_DEVICE", "cuda")

    def __repr__(self):
        return f"JoyAIRuntimeConfig(te_offload={self.te_offload!r}, device={self.device!r})"


def should_offload_te(config: "JoyAIRuntimeConfig | None" = None) -> bool:
    """DS_JOYAI_TE_OFFLOAD から、この生成で text_encoder CPU退避を使うか判定する。

    48GB専有では常駐(transformer ~32.5GB + text_encoder ~17.5GB = ~50GB)がそもそも
    収まらないため、"auto" は常に True を返す(qwen_image の should_offload_edit_text_encoder()
    と異なり解像度依存の分岐は設けない。JoyAIは常にtransformerが大きい16Bモデルのため)。
    """
    config = config or JoyAIRuntimeConfig()
    if config.te_offload == "off":
        return False
    return True  # "auto" / "on" とも常に有効
