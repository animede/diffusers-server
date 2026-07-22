# -*- coding: utf-8 -*-
"""
Mage-Flow ラッパーサービス(専用venv `venv-mageflow/` で動かす独立FastAPIプロセス)。

なぜ別プロセスか(CLAUDE.md 50番参照):
  Mage-Flow(microsoft/Mage の mage_flow パッケージ)は torch 2.13 / transformers 5.5 /
  diffusers 0.38 / flash-attn 2.8.3 を要求し、本体サーバ(venv/: torch 2.9 +
  diffusers 0.40.0.dev0)とはバージョンが衝突するため同一プロセスに載せられない。
  そこで完全隔離の専用venv(venv-mageflow/、--system-site-packages なし)で
  この小さなFastAPIを起動し、本体 app.py は HTTP プロキシ(/api/mageflow/*)経由で叩く。

起動方法(run_mageflow.sh 参照):
  DS_MAGEFLOW_PORT=8602 ./run_mageflow.sh
  または直接:
  venv-mageflow/bin/python -m uvicorn mageflow_service.app_mageflow:app \
      --host 127.0.0.1 --port 8602

設計:
  - モデルは遅延ロード+同時1バリアントのみ常駐(t2i/editの各3種、計6リポジトリ)。
    別バリアント要求時は先に unload + empty_cache してからロードする
    (4.1B bf16 で1モデル約9GB常駐+生成ピーク18-20GB。複数常駐はさせない)。
  - 生成は threading.Lock で同時1件(このプロセス内の排他。本体サーバの
    core.gpu.generation_lock とはプロセスが違うので、全体の排他は app.py 側の
    exclusive パラメータで制御する。CLAUDE.md 50番)。
  - 出力は本体と同じ outputs/ ディレクトリに mageflow_t2i_*.png / mageflow_edit_*.png
    として保存し、メタデータ(elapsed_s / peak_vram_gb / seed / image_url 等)を返す。
    本体 app.py が /outputs を静的配信しているため、image_url はそのままUIで表示できる。
  - Mage-Flow は全プロンプトをテキストエンコーダ内蔵のコンテンツゲートで検査し、
    拒否されたプロンプトは「拒否プレースホルダ画像」が返る仕様(無効化不可)。また
    正常出力には Gaussian-Shading 透かしが常に埋め込まれる(トグルなし)。
"""
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from io import BytesIO
from typing import List, Optional

# mage_flow を editable install していない環境でも動くよう third_party を sys.path に足す
# (通常は `venv-mageflow/bin/pip install -e third_party/Mage/mage_flow --no-deps` 済み)。
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_THIRD_PARTY_MAGE = os.path.join(_BASE_DIR, "third_party", "Mage")
if os.path.isdir(_THIRD_PARTY_MAGE) and _THIRD_PARTY_MAGE not in sys.path:
    sys.path.insert(0, _THIRD_PARTY_MAGE)

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, ImageOps
from pydantic import BaseModel

OUTPUTS_DIR = os.path.join(_BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = FastAPI(title="Mage-Flow Wrapper Service")

# ============================================================================
# バリアント定義(mage_flow README の公式表どおり)
#   T2I : base=30steps/cfg5.0, rl=20steps/cfg5.0, turbo=4steps/cfg1.0
#   Edit: base=30steps/cfg5.0, rl=30steps/cfg5.0, turbo=4steps/cfg1.0
# steps/cfg はリクエストで未指定(None)の場合にこの既定値を使う。
# ============================================================================
T2I_VARIANTS = {
    "base": {"repo": "microsoft/Mage-Flow-Base", "steps": 30, "cfg": 5.0},
    "rl": {"repo": "microsoft/Mage-Flow", "steps": 20, "cfg": 5.0},
    "turbo": {"repo": "microsoft/Mage-Flow-Turbo", "steps": 4, "cfg": 1.0},
}
EDIT_VARIANTS = {
    "base": {"repo": "microsoft/Mage-Flow-Edit-Base", "steps": 30, "cfg": 5.0},
    "rl": {"repo": "microsoft/Mage-Flow-Edit", "steps": 30, "cfg": 5.0},
    "turbo": {"repo": "microsoft/Mage-Flow-Edit-Turbo", "steps": 4, "cfg": 1.0},
}
DEFAULT_VARIANT = "rl"

MIN_SIZE = 512
MAX_SIZE = 2048
MAX_EDIT_IMAGES = 3  # 学習時の上限に合わせる(それ以上も受理されるが品質保証外)


# ============================================================================
# 状態管理(同時1モデルのみ常駐)
# ============================================================================
class _State:
    def __init__(self):
        self.lock = threading.Lock()  # 生成+ロードの排他(このプロセス内で同時1件)
        self.pipe = None
        self.kind: Optional[str] = None  # "t2i" | "edit"
        self.variant: Optional[str] = None  # "base" | "rl" | "turbo"
        self.repo: Optional[str] = None
        self.load_time_s: Optional[float] = None
        self.busy = False  # /status 用(lockを待たずに読める簡易フラグ)


state = _State()


def _unload_locked() -> "list[str]":
    """ロード済みパイプラインを解放する(state.lock 保持前提)。"""
    freed = []
    if state.pipe is not None:
        freed.append(f"{state.kind}:{state.variant} ({state.repo})")
        # MageFlowPipeline は .model(MageFlowModel)配下に transformer/vae/text_encoder を持つ。
        # 参照を切って gc + empty_cache で返却する(diffusers-server 本体の unload と同じ方針)。
        state.pipe = None
        state.kind = None
        state.variant = None
        state.repo = None
        state.load_time_s = None
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    return freed


def _ensure_pipeline_locked(kind: str, variant: str):
    """要求された (kind, variant) のパイプラインを返す(state.lock 保持前提)。

    別の (kind, variant) が常駐していれば先に解放してからロードする(同時1モデル)。
    """
    variants = T2I_VARIANTS if kind == "t2i" else EDIT_VARIANTS
    if variant not in variants:
        raise HTTPException(
            status_code=400,
            detail=f"model は base / rl / turbo のいずれかを指定してください(指定値: {variant})",
        )
    repo = variants[variant]["repo"]
    if state.pipe is not None and state.kind == kind and state.variant == variant:
        return state.pipe

    _unload_locked()
    from mage_flow import MageFlowPipeline

    t0 = time.time()
    pipe = MageFlowPipeline.from_pretrained(repo, device="cuda")
    state.pipe = pipe
    state.kind = kind
    state.variant = variant
    state.repo = repo
    state.load_time_s = time.time() - t0
    return pipe


def _round16_clamp(x: int) -> int:
    """16の倍数へ丸め、512〜2048にクランプする(Mage-Flowの対応解像度)。"""
    x = round(x / 16) * 16
    return max(MIN_SIZE, min(MAX_SIZE, x))


def _resolve_seed(seed: int) -> int:
    if seed is None or seed < 0:
        return int.from_bytes(os.urandom(4), "little")
    return seed


def _new_output_path(mode: str) -> "tuple[str, str]":
    """outputs/ 配下の新規ファイルパス(本体 generate.py の命名規則を踏襲)。"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{mode}_{ts}_{uuid.uuid4().hex[:8]}.png"
    return os.path.join(OUTPUTS_DIR, name), name


def _peak_vram_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024**3)


def _defaults_for(kind: str, variant: str, steps: Optional[int], cfg: Optional[float]):
    """steps/cfg が未指定ならバリアント既定値で埋める(turbo→4/1.0 等)。"""
    variants = T2I_VARIANTS if kind == "t2i" else EDIT_VARIANTS
    info = variants.get(variant, variants[DEFAULT_VARIANT])
    return (steps if steps is not None else info["steps"],
            cfg if cfg is not None else info["cfg"])


# ============================================================================
# エンドポイント
# ============================================================================
class T2IRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    steps: Optional[int] = None  # 未指定はバリアント既定(base30/rl20/turbo4)
    cfg: Optional[float] = None  # 未指定はバリアント既定(base/rl 5.0, turbo 1.0)
    seed: int = -1
    model: str = DEFAULT_VARIANT  # "base" | "rl" | "turbo"


@app.post("/t2i")
def t2i(req: T2IRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt を指定してください")
    width = _round16_clamp(req.width)
    height = _round16_clamp(req.height)
    steps, cfg = _defaults_for("t2i", req.model, req.steps, req.cfg)
    seed = _resolve_seed(req.seed)

    # ロック待ちで後続リクエストを積み上げない(初回のモデルダウンロード(~17GB)は
    # 数分〜数十分かかりうるため、待たせるとプロキシ側(300s)が全てタイムアウトする。
    # 本体サーバの409規約と同じ「実行中は即409」に統一)。
    if not state.lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Mage-Flowサービスが別の処理(モデルのダウンロード/ロードまたは生成)を"
            "実行中です。GET /status で busy を確認し、完了後に再試行してください。",
        )
    try:
        state.busy = True
        try:
            pipe = _ensure_pipeline_locked("t2i", req.model)
            load_time_s = state.load_time_s
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            images = pipe.generate(
                [req.prompt],
                neg_prompts=[req.negative_prompt] if req.negative_prompt else None,
                seeds=[seed],
                steps=steps,
                cfg=cfg,
                heights=[height],
                widths=[width],
            )
            elapsed = time.time() - t0
            peak = _peak_vram_gb()
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Mage-Flow T2I生成に失敗しました: {exc}")
        finally:
            state.busy = False
    finally:
        state.lock.release()

    out_path, out_name = _new_output_path("mageflow_t2i")
    images[0].save(out_path)
    return {
        "mode": "mageflow_t2i",
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": cfg,
        "seed": seed,
        "model": req.model,
        "repo": T2I_VARIANTS[req.model]["repo"],
        "elapsed_s": elapsed,
        "load_time_s": load_time_s,
        "peak_vram_gb": peak,
        "image_url": f"/outputs/{out_name}",
        "output_path": out_path,
    }


@app.post("/edit")
async def edit(
    image: List[UploadFile] = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    steps: Optional[int] = Form(None),
    cfg: Optional[float] = Form(None),
    seed: int = Form(-1),
    max_size: int = Form(1024),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    model: str = Form(DEFAULT_VARIANT),
):
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="prompt を指定してください")
    if not image:
        raise HTTPException(status_code=400, detail="参照画像を1枚以上アップロードしてください")
    if len(image) > MAX_EDIT_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"参照画像は最大{MAX_EDIT_IMAGES}枚までです(学習時の上限)",
        )

    refs = []
    for up in image:
        data = await up.read()
        try:
            img = Image.open(BytesIO(data))
            img = ImageOps.exif_transpose(img)  # スマホ撮影画像のEXIF回転を正規化(本体と同じ)
            refs.append(img.convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"画像を読み込めません({up.filename}): {exc}")

    steps_v, cfg_v = _defaults_for("edit", model, steps, cfg)
    seed_v = _resolve_seed(seed)
    # width/height 両方指定なら明示解像度(16の倍数へ丸め+クランプ)、
    # 未指定なら max_size(長辺)でアスペクト比維持のまま生成する(mage_flow の仕様)。
    heights = widths = None
    max_size_v = None
    if width is not None and height is not None:
        heights = [_round16_clamp(height)]
        widths = [_round16_clamp(width)]
    else:
        max_size_v = max(MIN_SIZE, min(MAX_SIZE, int(max_size)))

    # ロック待ちの積み上げ防止(t2i側と同じ理由・同じ409規約)。
    if not state.lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Mage-Flowサービスが別の処理(モデルのダウンロード/ロードまたは生成)を"
            "実行中です。GET /status で busy を確認し、完了後に再試行してください。",
        )
    try:
        state.busy = True
        try:
            pipe = _ensure_pipeline_locked("edit", model)
            load_time_s = state.load_time_s
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            images_out = pipe.edit(
                [prompt],
                [refs],  # 1サンプル = 参照画像のリスト(複数参照はシーケンス連結される)
                neg_prompts=[negative_prompt] if negative_prompt else None,
                seeds=[seed_v],
                steps=steps_v,
                cfg=cfg_v,
                max_size=max_size_v,
                heights=heights,
                widths=widths,
            )
            elapsed = time.time() - t0
            peak = _peak_vram_gb()
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Mage-Flow Edit生成に失敗しました: {exc}")
        finally:
            state.busy = False
    finally:
        state.lock.release()

    out_img = images_out[0]
    out_path, out_name = _new_output_path("mageflow_edit")
    out_img.save(out_path)
    return {
        "mode": "mageflow_edit",
        "prompt": prompt,
        "negative_prompt": negative_prompt or None,
        "num_ref_images": len(refs),
        "width": out_img.width,
        "height": out_img.height,
        "max_size": max_size_v,
        "steps": steps_v,
        "cfg": cfg_v,
        "seed": seed_v,
        "model": model,
        "repo": EDIT_VARIANTS[model]["repo"],
        "elapsed_s": elapsed,
        "load_time_s": load_time_s,
        "peak_vram_gb": peak,
        "image_url": f"/outputs/{out_name}",
        "output_path": out_path,
    }


@app.get("/status")
def status():
    vram = None
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        vram = {
            "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
            "free_gb": free_b / (1024**3),
            "total_gb": total_b / (1024**3),
        }
    return {
        "service": "mageflow",
        "loaded": state.pipe is not None,
        "kind": state.kind,
        "variant": state.variant,
        "repo": state.repo,
        "load_time_s": state.load_time_s,
        "busy": state.busy,
        "vram": vram,
    }


@app.post("/unload")
def unload():
    with state.lock:
        freed = _unload_locked()
    return {"freed": freed}
