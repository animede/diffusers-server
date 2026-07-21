# -*- coding: utf-8 -*-
"""
JoyAIFamily: core.registry.ModelFamily の実装(JoyAI-Image-Edit-Plus、Edit のみ)。
"""
from core.registry import ModelFamily

from families.joyai import generate as generate_mod
from families.joyai import lifecycle
from families.joyai import pipeline as pipeline_mod


class JoyAIFamily(ModelFamily):
    name = "joyai"

    def load(self, mode: str, **kwargs) -> None:
        if mode == "edit":
            pipeline_mod.get_pipeline()
        else:
            raise ValueError(f"unknown joyai mode: {mode!r}")

    def generate(self, request: dict) -> dict:
        mode = request["mode"]
        if mode == "edit":
            return generate_mod.run_edit(request, request["_images"])
        raise ValueError(f"unknown joyai mode: {mode!r}")

    def unload(self) -> dict:
        return lifecycle.unload("all")

    def status(self) -> dict:
        return lifecycle.get_status()

    def is_loaded(self) -> bool:
        return lifecycle.is_loaded()
