# -*- coding: utf-8 -*-
"""
ZImageFamily: core.registry.ModelFamily の実装(Z-Image T2I / I2I / Inpaint)。

load(mode) / generate(request) / unload() / status() の共通インターフェースで
T2I / I2I / Inpaint をまとめる(families/flux2/family.py と対になる実装)。

mode:
  "t2i"     -> pipeline.get_base_pipeline()
  "i2i"     -> pipeline.get_i2i_pipeline()(base の components から遅延構築)
  "inpaint" -> pipeline.get_inpaint_pipeline()(base の components から遅延構築)

いずれのモードも実体は同じ base ZImagePipeline のコンポーネント(transformer/vae/
text_encoder等)を共有するため、どれか1つをロードすれば他モードのロードは
「ラッパーオブジェクトの構築」のみで VRAM を追加消費しない。
"""
from core.registry import ModelFamily

from families.z_image import generate as generate_mod
from families.z_image import lifecycle
from families.z_image import pipeline as pipeline_mod


class ZImageFamily(ModelFamily):
    name = "z_image"

    def load(self, mode: str, **kwargs) -> None:
        if mode == "t2i":
            pipeline_mod.get_base_pipeline()
        elif mode == "i2i":
            pipeline_mod.get_i2i_pipeline()
        elif mode == "inpaint":
            pipeline_mod.get_inpaint_pipeline()
        else:
            raise ValueError(f"unknown z_image mode: {mode!r}")

    def generate(self, request: dict) -> dict:
        mode = request["mode"]
        if mode == "t2i":
            return generate_mod.run_t2i(request)
        if mode == "i2i":
            return generate_mod.run_i2i(request, request["_image"])
        if mode == "inpaint":
            return generate_mod.run_inpaint(request, request["_image"], request["_mask_image"])
        raise ValueError(f"unknown z_image mode: {mode!r}")

    def unload(self) -> dict:
        return lifecycle.unload("all")

    def status(self) -> dict:
        return lifecycle.get_status()

    def is_loaded(self) -> bool:
        return lifecycle.is_loaded()
