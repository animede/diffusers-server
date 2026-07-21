# -*- coding: utf-8 -*-
"""
Flux2Family: core.registry.ModelFamily の実装(FLUX.2 T2I/I2I)。

load(mode) / generate(request) / unload() / status() の共通インターフェースで
T2I / I2I をまとめる(Qwen系 families/qwen_image/family.py と対になる実装)。

mode:
  "t2i" / "i2i" -> どちらも pipeline.get_pipeline() で同一の Flux2Pipeline を返す
                   (image有無で挙動が変わるだけなのでロード自体は共通)。

generate(request) の request は {"mode": <str>, ...パラメータ, "_images": [...] (i2iのみ)}
形式(app.py 側であらかじめ画像を読み込み・前処理してから渡す)。
"""
from core.registry import ModelFamily

from families.flux2 import generate as generate_mod
from families.flux2 import lifecycle
from families.flux2.pipeline import get_pipeline


class Flux2Family(ModelFamily):
    name = "flux2"

    def load(self, mode: str, model: "str | None" = None) -> None:
        """model: "dev"(既定)| None。"ecocoro" 指定時は ValueError
        (2026-07-19、ecocoro廃止。get_pipeline() 参照)。
        """
        if mode not in ("t2i", "i2i"):
            raise ValueError(f"unknown flux2 mode: {mode!r}")
        get_pipeline(model)

    def generate(self, request: dict) -> dict:
        mode = request["mode"]
        if mode == "t2i":
            return generate_mod.run_t2i(request)
        if mode == "i2i":
            return generate_mod.run_i2i(request, request["_images"])
        raise ValueError(f"unknown flux2 mode: {mode!r}")

    def unload(self) -> dict:
        return lifecycle.unload("all")

    def status(self) -> dict:
        return lifecycle.get_status()

    def is_loaded(self) -> bool:
        return lifecycle.is_loaded()
