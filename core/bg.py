# -*- coding: utf-8 -*-
"""背景除去(方式を選択できる共通モジュール、2026-07-28追加)。

方式:
  - "rembg" (既定): rembg / isnet-general-use(汎用。従来の実装。写真・実物向け)
  - "anime": SkyTNT/anime-segmentation の ISNet(`skytnt/anime-seg` の `isnetis.onnx`、
    Apache-2.0)。**アニメ・イラスト・ぬいぐるみ的なキャラクター画像で汎用モデルより
    切り抜きが良い**ためユーザー要望で追加した。

依存は追加していない: onnxruntime は rembg が既に使っており、cv2 も他機能
(ControlNet の canny、LTX-2.3 の動画読み込み)で使用済み。モデルは
`hf_hub_download("skytnt/anime-seg", "isnetis.onnx")`(176MB)で通常のHFキャッシュへ
取得する(`DS_ANIME_SEG_ONNX` にローカルパスを指定すればそれを使う)。

実行プロバイダは既定CPU(生成用GPUと競合させない。rembgもCPU実行)。
`DS_ANIME_SEG_PROVIDER=cuda` でGPU実行にできる。
"""
import os
import threading

import numpy as np
from PIL import Image

# 方式名(APIの method パラメータで指定する値)
BG_METHODS = ("rembg", "anime")
DEFAULT_BG_METHOD = "rembg"

_ANIME_REPO = "skytnt/anime-seg"
_ANIME_FILE = "isnetis.onnx"
_ANIME_SIZE = 1024  # ONNX の入力は [1,3,1024,1024] 固定(実機で確認済み)

_anime_session = None
_anime_lock = threading.Lock()


def _anime_model_path() -> str:
    override = os.environ.get("DS_ANIME_SEG_ONNX")
    if override:
        return override
    from huggingface_hub import hf_hub_download

    return hf_hub_download(_ANIME_REPO, _ANIME_FILE)


def _get_anime_session():
    """ISNet(anime-segmentation)の ONNX セッション(遅延ロードのシングルトン)。"""
    global _anime_session
    if _anime_session is None:
        with _anime_lock:
            if _anime_session is None:
                import onnxruntime as ort

                want = (os.environ.get("DS_ANIME_SEG_PROVIDER") or "cpu").strip().lower()
                available = ort.get_available_providers()
                if want == "cuda" and "CUDAExecutionProvider" in available:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    providers = ["CPUExecutionProvider"]
                path = _anime_model_path()
                print(f"[core.bg] anime-segmentation をロードします: {path} ({providers[0]})")
                _anime_session = ort.InferenceSession(path, providers=providers)
    return _anime_session


def _remove_background_anime(img: Image.Image) -> Image.Image:
    """anime-segmentation(ISNet)で背景を除去して RGBA を返す。

    前処理・後処理は本家 inference.py と同じ:
    アスペクト比を保って長辺を1024へ縮小 -> 1024x1024 の中央へゼロパディング ->
    0-1正規化・CHW -> 推論 -> パディングを取り除いて元サイズへ戻す。
    """
    import cv2

    rgb = np.asarray(img.convert("RGB"))
    h0, w0 = rgb.shape[:2]
    s = _ANIME_SIZE
    if h0 > w0:
        h, w = s, max(1, int(s * w0 / h0))
    else:
        h, w = max(1, int(s * h0 / w0)), s
    ph, pw = s - h, s - w
    buf = np.zeros((s, s, 3), dtype=np.float32)
    buf[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(
        rgb.astype(np.float32) / 255.0, (w, h)
    )
    inp = np.transpose(buf, (2, 0, 1))[None]

    session = _get_anime_session()
    mask = session.run(None, {session.get_inputs()[0].name: inp})[0][0][0]
    mask = mask[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
    mask = cv2.resize(mask, (w0, h0))
    alpha = np.clip(mask * 255.0, 0, 255).astype(np.uint8)

    out = img.convert("RGBA")
    out.putalpha(Image.fromarray(alpha, mode="L"))
    return out


def resolve_method(method: str) -> str:
    """方式名を正規化する(未知の値は既定へフォールバックし警告を出す)。"""
    value = (method or DEFAULT_BG_METHOD).strip().lower()
    if value in BG_METHODS:
        return value
    print(f"[core.bg] warning: 未知の背景除去方式 {method!r} を無視して "
          f"{DEFAULT_BG_METHOD!r} を使います。")
    return DEFAULT_BG_METHOD


def remove_background(img: Image.Image, method: str = DEFAULT_BG_METHOD) -> Image.Image:
    """背景を除去して RGBA(背景透過)画像を返す。

    method: "rembg"(既定、汎用)| "anime"(アニメ・イラスト・キャラクター向け)
    """
    if resolve_method(method) == "anime":
        return _remove_background_anime(img)
    # rembg / isnet-general-use(従来実装。apps/charsheet/bg.py に置いていたもの)
    from apps.charsheet.bg import remove_background_rembg

    return remove_background_rembg(img)
