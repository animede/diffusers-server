# -*- coding: utf-8 -*-
"""アップスケール(Real-ESRGAN x2、spandrel 経由)。

用途: Tポーズ4ビューの出力(既定 1024²)を **2048²** にしてから 3D化・リグへ渡す。

方式の選定(2026-07-28、ユーザー要望「2048へアップスケールする機能(ボタン)」):
  - **GAN系のアップスケーラ(Real-ESRGAN)を採用**。決定論的で**内容を書き換えない**ため、
    このプロジェクトが繰り返し戦ってきた同一性ドリフト(髪型・衣装・体型)が起きない。
  - 却下: 拡散モデル(Qwen-Image-Edit)で2048を再生成する案。解像度は上がるが
    **再生成なので細部が変わる**(実測でも編集のたびに髪型・衣装が動いた)うえ、
    2048²は計算量・VRAMとも重い。3D入力用の「同じ絵をきれいに拡大したい」という
    要件には合わない。

依存は増やしていない: `spandrel`(0.4.1)は comfy-env に既存で、venv から import できる。
重みは `hf_hub_download("ai-forever/Real-ESRGAN", "RealESRGAN_x2.pth")`(64MB、
通常のHFキャッシュ)。ローカルに置きたい場合は `DS_UPSCALE_MODEL` にパスを指定する
(ComfyUI の `models/upscale_models/` に置いたファイルもそのまま渡せる)。

GPU は生成と同じ1本のロック(core.gpu.generation_lock)で排他すること(呼び出し側の責務)。
"""
import os
import threading

import torch
from PIL import Image

__all__ = ["DEFAULT_TARGET", "upscale_image", "unload"]

DEFAULT_TARGET = 2048

_HF_REPO = "ai-forever/Real-ESRGAN"
_HF_FILE = "RealESRGAN_x2.pth"

_model = None
_lock = threading.Lock()


def _model_path() -> str:
    override = os.environ.get("DS_UPSCALE_MODEL")
    if override:
        return override
    from huggingface_hub import hf_hub_download

    return hf_hub_download(_HF_REPO, _HF_FILE)


def _get_model():
    """spandrel でモデルをロードする(遅延ロードのシングルトン、GPU常駐)。"""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                import spandrel

                path = _model_path()
                print(f"[core.upscale] アップスケーラをロードします: {path}")
                descriptor = spandrel.ModelLoader().load_from_file(path)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                descriptor.model.eval().to(device)
                print(
                    f"[core.upscale] {type(descriptor.model).__name__} "
                    f"(x{descriptor.scale}) を {device} にロードしました"
                )
                _model = descriptor
    return _model


def _to_tensor(img: Image.Image, device) -> torch.Tensor:
    import numpy as np

    a = np.asarray(img.convert("RGB"), dtype="float32") / 255.0
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)


def _to_image(t: torch.Tensor) -> Image.Image:
    import numpy as np

    a = t.squeeze(0).clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((a * 255.0 + 0.5).astype(np.uint8), "RGB")


def upscale_image(img: Image.Image, target: int = DEFAULT_TARGET) -> Image.Image:
    """画像を拡大して長辺が `target` px になるようにする。

    モデルは x2 固定なので、まず x2 で拡大し、`target` と一致しない場合だけ
    Lanczos で合わせる(既定運用の 1024 -> 2048 はちょうど x2 なので再サンプルなし)。
    アルファは扱わない(呼び出し側は白背景版を渡し、透過版は拡大後に作り直すこと。
    そのほうが切り抜きの精度も上がる)。
    """
    descriptor = _get_model()
    device = next(descriptor.model.parameters()).device
    with torch.no_grad():
        out = descriptor.model(_to_tensor(img, device))
    result = _to_image(out)
    del out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    w, h = result.size
    long_side = max(w, h)
    if long_side != target:
        scale = target / float(long_side)
        result = result.resize(
            (max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS
        )
    return result


def unload() -> bool:
    """アップスケーラを解放する(VRAM 約0.1GB)。解放したら True。"""
    global _model
    with _lock:
        if _model is None:
            return False
        _model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[core.upscale] アップスケーラを解放しました")
    return True
