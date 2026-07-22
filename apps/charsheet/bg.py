# -*- coding: utf-8 -*-
"""
背景除去(rembg / ISNet)。旧 charsheet/bg.py の移植(ロジック無変更)。

briaai/RMBG-1.4 の transformers remote code は transformers 5.x と非互換のため、
rembg パッケージ(isnet-general-use, ONNX)を使用する。
モデルは遅延ロードのシングルトン(初回のみ ~179MB を ~/.u2net にダウンロード)。
rembg は comfy-env(venv の親環境)に既存で、venv からも import 可能なことを確認済み
(pip install はしない、REBUILD_PLAN §3.4)。
"""
import threading

from PIL import Image

MODEL_NAME = "isnet-general-use"

_session = None
_session_lock = threading.Lock()


def _get_session():
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                from rembg import new_session

                _session = new_session(MODEL_NAME)
    return _session


def remove_background(img: Image.Image) -> Image.Image:
    """背景を除去して RGBA(背景透過)画像を返す。"""
    from rembg import remove

    result = remove(img.convert("RGB"), session=_get_session())
    return result.convert("RGBA")
