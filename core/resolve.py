# -*- coding: utf-8 -*-
"""
モデルパス解決(ComfyUI ローカル優先 + HF Hub 自動DLフォールバック)。

抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py
  - _resolve_model_path() (行239-249)
  - COMFYUI_DIR / COMFYUI_MODELS_DIR の定義 (行49-51) は core/config.py に移設し、
    ここでは core.config から import して使う。

変更点: ロジックは無変更。print() によるログはそのまま踏襲(呼び出し側から見える
挙動を変えないため)。torch の import は行わない(このモジュールは CUDA を一切使わない)。
"""
import os

from huggingface_hub import hf_hub_download

from core.config import COMFYUI_MODELS_DIR

__all__ = ["COMFYUI_MODELS_DIR", "resolve_model_path", "resolve_comfyui_path"]


def resolve_model_path(local_path: str, repo_id: str, repo_filename: str) -> str:
    """ローカル(ComfyUI)にファイルがあればそれを使い、無ければ HF Hub から自動ダウンロードして
    そのキャッシュパスを返す(2回目以降はキャッシュヒットで再ダウンロードしない)。

    抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py の _resolve_model_path()(行239-249)。
    """
    if os.path.exists(local_path):
        return local_path
    print(
        f"[core.resolve] {local_path} が見つからないため HF Hub からダウンロードします: "
        f"{repo_id}/{repo_filename}"
    )
    downloaded = hf_hub_download(repo_id=repo_id, filename=repo_filename)
    print(f"[core.resolve] ダウンロード完了: {downloaded}")
    return downloaded


def resolve_comfyui_path(*parts: str) -> str:
    """COMFYUI_MODELS_DIR 配下のパスを組み立てるショートハンド。

    例: resolve_comfyui_path("diffusion_models", "qwen_image_fp8_e4m3fn.safetensors")
    元実装では各ファミリーが os.path.join(COMFYUI_MODELS_DIR, ...) を個別に書いていた
    (例: pipeline_manager.py 行54-56 の T2I_TRANSFORMER_PATH)。Phase 1 の移植で
    families/qwen_image.py がこの関数を使って同じパスを組み立てられるようにする。
    """
    return os.path.join(COMFYUI_MODELS_DIR, *parts)
