# -*- coding: utf-8 -*-
"""
diffusers-server 本体(FastAPI、薄く保つ)。

Phase 1: Qwen-Image 系(T2I / I2I / Edit / ControlNet / Inpaint / Canny / Layered)を
core.registry 経由で提供する。旧 Ｑwenimage-edit-diffusers/app.py とパス・パラメータ互換。
Phase 2: FLUX.2 系(T2I / I2I)を追加。families/flux2 は families/qwen_image と
exclusive_with 関係で登録されており(REBUILD_PLAN §3.1)、どちらかの生成リクエスト時に
もう一方がロード済みなら registry.load() が自動 unload してからロードする。

GPU 処理は core.registry.registry.generate() が全ファミリー共通の1本のロック
(core.gpu.generation_lock)で排他制御する(busy 時は 409)。生成結果は outputs/ に保存し、
統一メタデータ(所要時間・ピークVRAM・モデル/量子化・全パラメータ・seed)を返す。

ポート 8601。static/ を "/" で配信(旧サーバと同じ)。
"""
import io
import os
import time
import traceback
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

import families.qwen_image as qwen_image  # noqa: F401  (import 副作用で registry に登録)
import families.flux2 as flux2  # noqa: F401  (import 副作用で registry に登録)
import families.z_image as z_image  # noqa: F401  (import 副作用で registry に登録)
import families.ltx2 as ltx2  # noqa: F401  (import 副作用で registry に登録)
import families.joyai as joyai  # noqa: F401  (import 副作用で registry に登録)
from apps.charsheet import router as charsheet_router
from apps.charsheet.bg import remove_background
from apps.scene_angles import router as scene_angles_router
from core import llm, progress as progress_mod
from core.registry import registry

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
STATIC_DIR = os.path.join(BASE_DIR, "static")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = FastAPI(title="DiffusersWEBUI")


def _open_upload_image(data: bytes, mode: "str | None" = "RGB") -> Image.Image:
    """アップロード画像を開く共通ヘルパー(EXIF回転の正規化込み)。

    スマホ撮影画像等は「ピクセルは横向きのまま + EXIF Orientation で回転を表現」して
    保存されている。ブラウザのプレビューはEXIFを解釈して正立表示するが、PILの
    `Image.open()` は生ピクセルをそのまま返すため、これを怠ると生成結果だけが90度
    回転する(実ユーザー報告のバグ、2026-07-20修正)。`ImageOps.exif_transpose()` で
    回転を実ピクセルに適用してから指定モードへ変換する。
    """
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    return img.convert(mode) if mode else img

_last_generation = {"mode": None, "elapsed_s": None, "peak_vram_gb": None, "at": None}

MAX_EDIT_IMAGES = qwen_image.MAX_EDIT_IMAGES
MAX_FLUX2_REF_IMAGES = flux2.MAX_REF_IMAGES

# request内の "mode" (t2i/i2i/edit/controlnet/...) を、どのファミリーが担当するかへの対応表。
# flux2_t2i / flux2_i2i / zimage_t2i / zimage_i2i / zimage_inpaint は families/flux2・
# families/z_image 側の mode 名と別名にしてある(qwen_image の t2i/i2i/inpaint と
# 名前が衝突しないよう、外部向けのモード識別子として区別する)。
_MODE_TO_FAMILY = {
    "t2i": "qwen_image",
    "i2i": "qwen_image",
    "edit": "qwen_image",
    "controlnet": "qwen_image",
    "inpaint": "qwen_image",
    "layered": "qwen_image",
    "flux2_t2i": "flux2",
    "flux2_i2i": "flux2",
    "zimage_t2i": "z_image",
    "zimage_i2i": "z_image",
    "zimage_inpaint": "z_image",
    "ltx2_t2v": "ltx2",
    "ltx2_i2v": "ltx2",
    "ltx2_flf": "ltx2",
    "ltx2_ia2v": "ltx2",
    "ltx2_keyframes": "ltx2",
    "ltx2_v2a": "ltx2",
    "ltx2_iclora": "ltx2",
    "joyai_edit": "joyai",
}
# families/flux2・families/z_image・families/ltx2 側の ModelFamily.load()/generate() は
# 内部モード名(t2i/i2i/inpaint/t2v/i2v/flf/ia2v)のみを受け付けるため、外部向けモード名から
# 内部モード名への変換表を持つ。
_FLUX2_MODE_ALIAS = {"flux2_t2i": "t2i", "flux2_i2i": "i2i"}
_ZIMAGE_MODE_ALIAS = {"zimage_t2i": "t2i", "zimage_i2i": "i2i", "zimage_inpaint": "inpaint"}
_LTX2_MODE_ALIAS = {
    "ltx2_t2v": "t2v",
    "ltx2_i2v": "i2v",
    "ltx2_flf": "flf",
    "ltx2_ia2v": "ia2v",
    "ltx2_keyframes": "keyframes",
    "ltx2_v2a": "v2a",
    "ltx2_iclora": "iclora",
}
_JOYAI_MODE_ALIAS = {"joyai_edit": "edit"}


def _record_last_generation(meta: dict) -> None:
    _last_generation.update(
        mode=meta.get("mode"),
        elapsed_s=meta.get("elapsed_s"),
        peak_vram_gb=meta.get("peak_vram_gb"),
        at=datetime.now().isoformat(),
    )


def _generate_or_409(mode: str, request: dict, load_kwargs: "dict | None" = None) -> dict:
    """全ファミリー共通ロック(core.gpu.generation_lock)を非ブロッキングで取得し、
    取得できた場合のみ対象ファミリーの load() -> generate() を直接呼ぶ。
    ロック取得に失敗した場合(既に他の生成が実行中)は 409 を返す(旧実装の
    _acquire_gpu_or_409() 相当)。

    注意1: core.registry.FamilyRegistry.generate() は内部で gpu.generation_scope()
    (ブロッキング取得)を使うため、409即時応答にはここで非ブロッキング取得してから
    ファミリーの generate() を直接呼ぶ必要がある(registry.generate() を経由すると
    ロック取得済みの状態から二重に with generation_lock: するとデッドロックする)。

    注意2(Phase 2 で追加): 生成の前に必ず registry.load(family_name, mode) を呼ぶ。
    これにより、排他グループ(qwen_image ⇔ flux2)の相手が常駐中なら registry が
    自動 unload してからロードする(REBUILD_PLAN §3.1)。各ファミリーの
    get_*_pipeline()/get_pipeline() 自体も遅延ロードで冪等なため、load() を先に
    呼んでも二重ロードにはならない。

    load_kwargs(タスク1/タスク2で追加): registry.load(family_name, internal_mode,
    **load_kwargs) にそのまま転送する(例: {"model": "2512"})。モデル切替時の
    バリデーションエラーは各ファミリーが ValueError を送出する規約とし、
    ここで 400 に変換する。
    """
    from core import gpu

    family_name = _MODE_TO_FAMILY[mode]
    internal_mode = (
        _FLUX2_MODE_ALIAS.get(mode)
        or _ZIMAGE_MODE_ALIAS.get(mode)
        or _LTX2_MODE_ALIAS.get(mode)
        or _JOYAI_MODE_ALIAS.get(mode)
        or mode
    )
    load_kwargs = load_kwargs or {}

    if not gpu.generation_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409, detail="別の生成が実行中です。しばらく待ってから再試行してください。"
        )
    try:
        gpu.empty_cache()
        # 進捗状態: registry.load() はモデル未ロード時・ファミリー切替時に実際の
        # ロード処理(数秒〜数十秒)を行う。ロード済みなら registry.load() は
        # 即座に返る冪等呼び出しのため、その場合はこの phase="loading" はごく短時間で
        # family.generate() 内の progress_mod.start_generating() に上書きされるだけで
        # 実害はない(core/progress.py の設計方針)。
        progress_mod.start_loading(mode)
        registry.load(family_name, internal_mode, **load_kwargs)
        request = {"mode": internal_mode, **request}
        family = registry.get(family_name)
        meta = family.generate(request)
        meta["mode"] = mode  # 外部向けモード名(flux2_t2i等)に戻す
        _record_last_generation(meta)
        return meta
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{mode}生成に失敗しました: {exc}")
    finally:
        progress_mod.finish()
        gpu.generation_lock.release()


# ============================================================================
# T2I
# ============================================================================
class T2IRequest(BaseModel):
    prompt: str
    negative_prompt: str = qwen_image.NEGATIVE_PROMPT_DEFAULT
    width: int = 1328
    height: int = 1328
    steps: int = 30
    cfg: float = 4.0
    seed: int = -1
    lightning: bool = False
    shift: Optional[float] = None
    # タスク1(2026-07-18更新): "2512"(既定。公式 Qwen/Qwen-Image-2512 fp8-lightning fuse)
    # | "qwen-image"(無印)。旧値 "2512-4bit" は後方互換で "2512" として扱う(README参照)。
    # 未指定 = 現行の DS_T2I_MODEL 既定のまま(切替なし)。指定時、現在ロード済みのモデルと
    # 異なればT2Iグループを自動 unload してから切り替える(families/qwen_image/t2i.py
    # の _switch_t2i_model_if_needed() 参照)。
    model: Optional[str] = None


@app.post("/api/t2i")
def generate_t2i(req: T2IRequest):
    payload = req.model_dump()
    return _generate_or_409("t2i", payload, load_kwargs={"model": payload.get("model")})


# ============================================================================
# I2I
# ============================================================================
@app.post("/api/i2i")
async def generate_i2i(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(qwen_image.NEGATIVE_PROMPT_DEFAULT),
    strength: float = Form(0.6),
    steps: int = Form(30),
    cfg: float = Form(4.0),
    seed: int = Form(-1),
    lightning: bool = Form(False),
    shift: Optional[float] = Form(None),
    # タスク1(2026-07-18更新): 任意。"2512"(既定) | "qwen-image"。2512 が無印と同じ
    # transformer 共有構造(fp8-lightning fuse)になったため、I2I もモデル切替に対応した
    # (旧NF4時代は 2512-4bit 指定で 400 を返していた。families/qwen_image/t2i.py の
    # get_i2i_pipeline() 参照)。
    model: Optional[str] = Form(None),
):
    contents = await image.read()
    try:
        src_image = _open_upload_image(contents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")
    src_image = qwen_image.preprocess_image(src_image)

    return _generate_or_409(
        "i2i",
        {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "strength": strength,
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
            "lightning": lightning,
            "shift": shift,
            "model": model,
            "_image": src_image,
        },
        load_kwargs={"model": model},
    )


# ============================================================================
# Edit
# ============================================================================
@app.post("/api/edit")
async def generate_edit(
    images: List[UploadFile] = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(qwen_image.NEGATIVE_PROMPT_DEFAULT),
    steps: int = Form(30),
    cfg: float = Form(4.0),
    seed: int = Form(-1),
    lightning: bool = Form(True),
    shift: Optional[float] = Form(None),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
):
    # width/height 任意指定(Phase 1 申し送り事項への対処): 48GB専有では自動推定の
    # 標準解像度(1024²相当)でVAE decodeの一時ピークがOOMする実績があるため、
    # height=640&width=640 等を明示できるようにする。未指定時は旧挙動(自動推定)のまま。
    if not images:
        raise HTTPException(status_code=400, detail="参照画像を1〜3枚アップロードしてください。")
    if len(images) > MAX_EDIT_IMAGES:
        raise HTTPException(status_code=400, detail=f"参照画像は最大{MAX_EDIT_IMAGES}枚までです。")

    src_images = []
    for f in images:
        contents = await f.read()
        try:
            img = _open_upload_image(contents)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")
        src_images.append(qwen_image.preprocess_image(img))

    return _generate_or_409(
        "edit",
        {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
            "lightning": lightning,
            "shift": shift,
            "width": width,
            "height": height,
            "_images": src_images,
        },
    )


# ============================================================================
# ControlNet
# ============================================================================
# width/height は任意指定(未指定時は制御画像のサイズに自動追従、旧挙動のまま)。
# Phase 1 申し送り事項: 48GB専有では標準解像度(1024²相当)でVAE decodeの一時ピークが
# ~5GBほど乗りOOMする実績があるため、逼迫時は height=640&width=640 等を明示すること
# (families/qwen_image/generate.py run_controlnet() 参照)。
@app.post("/api/controlnet")
async def generate_controlnet(
    control_image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(" "),
    control_type: str = Form("canny"),
    controlnet_conditioning_scale: float = Form(1.0),
    steps: int = Form(30),
    cfg: float = Form(4.0),
    seed: int = Form(-1),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
):
    contents = await control_image.read()
    try:
        ctrl_img = _open_upload_image(contents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"制御画像の読み込みに失敗しました: {exc}")

    return _generate_or_409(
        "controlnet",
        {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "control_type": control_type,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
            "_control_image": ctrl_img,
        },
    )


# ============================================================================
# Inpaint
# ============================================================================
@app.post("/api/inpaint")
async def generate_inpaint(
    image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(" "),
    controlnet_conditioning_scale: float = Form(1.0),
    steps: int = Form(30),
    cfg: float = Form(4.0),
    seed: int = Form(-1),
):
    try:
        src_contents = await image.read()
        src_image = _open_upload_image(src_contents)
        mask_contents = await mask_image.read()
        mask_img = _open_upload_image(mask_contents, "L")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")

    return _generate_or_409(
        "inpaint",
        {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "steps": steps,
            "cfg": cfg,
            "seed": seed,
            "_image": src_image,
            "_mask_image": mask_img,
        },
    )


# ============================================================================
# アウトペイント(既存インペイント機構の流用による画角拡張、2026-07-24追加、
# CLAUDE.md 54番。キャンバス配置→エッジ延長充填→フェザーマスク→既存インペイント→
# 中央再合成。ヘルパーは apps/outpaint.py)
# ============================================================================
@app.post("/api/outpaint")
async def generate_outpaint(
    image: UploadFile = File(...),
    width: int = Form(1280),
    height: int = Form(720),
    prompt: str = Form(""),
    negative_prompt: str = Form(" "),
    seed: int = Form(-1),
    feather: int = Form(64),
    engine: str = Form("qwen"),
    steps: int = Form(30),
    cfg: float = Form(4.0),
):
    """アウトペイント: 入力画像をアスペクト維持で中央配置し、余白帯を既存インペイントで
    生成して指定サイズへ拡張する。最後に元画像の中央部をピクセルそのまま再合成する
    (フェザー帯より内側は完全無傷)。

    - `engine`: "qwen"(既定、A/B検証で明確に優位)| "zimage"(文脈無視の傾向があり
      非推奨、選択肢としてのみ残す)
    - `prompt`: 空でも可(空のときは周囲の文脈のみで生成される。シーン記述を渡すと
      左右の内容を誘導できる)
    - 左右拡張(縦長→横長)・上下拡張(横長→縦長)の両方に対応(フィット配置で
      帯が出る軸が自動的に決まる)。
    """
    from apps import outpaint as outpaint_mod

    engine = (engine or "qwen").strip().lower()
    if engine not in ("qwen", "zimage"):
        raise HTTPException(status_code=400, detail="engine は qwen / zimage のいずれかです")

    try:
        contents = await image.read()
        src_image = _open_upload_image(contents)
        canvas, mask, paste_box = outpaint_mod.build_outpaint_canvas(
            src_image, width, height, feather
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")

    if engine == "qwen":
        meta = _generate_or_409(
            "inpaint",
            {
                "prompt": prompt.strip() or " ",  # インペイントはprompt必須のため空は半角スペース
                "negative_prompt": negative_prompt,
                "controlnet_conditioning_scale": 1.0,
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
                "_image": canvas,
                "_mask_image": mask,
            },
        )
    else:
        meta = _generate_or_409(
            "zimage_inpaint",
            {
                "prompt": prompt.strip() or " ",
                "negative_prompt": negative_prompt,
                "strength": 1.0,
                "steps": 8,
                "guidance_scale": 0.0,
                "width": width,
                "height": height,
                "seed": seed,
                "_image": canvas,
                "_mask_image": mask,
            },
        )

    # 中央再合成(フェザーより内側は元画像ピクセル100%を保証)して別ファイルに保存
    generated_name = str(meta.get("image_url") or "").rsplit("/", 1)[-1]
    generated_path = os.path.join(OUTPUTS_DIR, generated_name)
    try:
        generated = Image.open(generated_path)
        final = outpaint_mod.recomposite_center(
            generated, src_image, paste_box, feather, width, height
        )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"outpaint_{ts}_{uuid.uuid4().hex[:8]}.png"
        final.save(os.path.join(OUTPUTS_DIR, out_name))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"中央再合成に失敗しました: {exc}")

    meta.update(
        mode="outpaint",
        image_url=f"/outputs/{out_name}",
        raw_inpaint_url=f"/outputs/{generated_name}",  # 再合成前(デバッグ用)
        engine=engine,
        feather=int(feather),
        width=final.width,
        height=final.height,
        paste_box=list(paste_box),
    )
    return meta


# ============================================================================
# Canny(おまけ機能、GPU不使用。ロック不要)
# ============================================================================
@app.post("/api/canny")
async def make_canny(
    image: UploadFile = File(...),
    low_threshold: int = Form(100),
    high_threshold: int = Form(200),
):
    contents = await image.read()
    try:
        src = _open_upload_image(contents, None)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")
    try:
        return qwen_image.run_canny(src, low_threshold, high_threshold)
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=f"opencv(cv2)が利用できません: {exc}")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Canny生成に失敗しました: {exc}")


# ============================================================================
# 背景削除ユーティリティ(おまけ機能、GPU不使用・ロック不要)
# ============================================================================
# rembg(isnet-general-use, ONNX)は apps/charsheet/bg.py がキャラシートの
# remove_bg 機能向けに既に使っている実装をそのまま再利用する(セッションは
# プロセス内シングルトン、初回のみ ~179MB を ~/.u2net にダウンロード)。
# GPU の generation_lock は使わない(画像生成とは無関係の CPU/ONNX 処理のため、
# core.llm と同様に GPU 処理中でも並行して呼べる)。
@app.post("/api/remove_bg")
async def api_remove_bg(image: UploadFile = File(...)):
    contents = await image.read()
    try:
        src = _open_upload_image(contents, "RGBA")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")

    t0 = time.time()
    try:
        out_image = remove_background(src)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"背景削除に失敗しました: {exc}")
    elapsed = time.time() - t0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"removebg_{ts}_{uuid.uuid4().hex[:8]}.png"
    out_image.save(os.path.join(OUTPUTS_DIR, out_name))
    return {"mode": "remove_bg", "image_url": f"/outputs/{out_name}", "elapsed_s": elapsed}


# ============================================================================
# Layered
# ============================================================================
@app.post("/api/layered")
async def generate_layered(
    image: UploadFile = File(...),
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    layers: int = Form(qwen_image.LAYERED_DEFAULT_LAYERS),
    resolution: int = Form(qwen_image.LAYERED_DEFAULT_RESOLUTION),
    steps: Optional[int] = Form(None),
    cfg: Optional[float] = Form(None),
    shift: Optional[float] = Form(None),
    seed: int = Form(-1),
    cfg_normalize: bool = Form(False),
    use_en_prompt: bool = Form(True),
):
    if resolution not in qwen_image.LAYERED_VALID_RESOLUTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"resolution は {qwen_image.LAYERED_VALID_RESOLUTIONS} のいずれかである必要があります",
        )
    if layers < 1 or layers > 16:
        raise HTTPException(status_code=400, detail="layers は 1〜16 の範囲で指定してください")

    contents = await image.read()
    try:
        # RGBAで渡すこと(専用VAEはinput_channels=4でRGBA画素空間を直接エンコードする。
        # RGBで渡すとVAEのconv_in入力チャンネル数が合わずエラーになる)。
        src_image = _open_upload_image(contents, "RGBA")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")

    return _generate_or_409(
        "layered",
        {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "layers": layers,
            "resolution": resolution,
            "steps": steps,
            "cfg": cfg,
            "shift": shift,
            "seed": seed,
            "cfg_normalize": cfg_normalize,
            "use_en_prompt": use_en_prompt,
            "_image": src_image,
        },
    )


# ============================================================================
# FLUX.2 T2I / I2I(Phase 2 で新設。families/flux2 が qwen_image と exclusive_with
# 関係で登録されているため、生成リクエスト時に qwen_image が常駐していれば
# registry.load() が自動 unload する)
# ============================================================================
class Flux2T2IRequest(BaseModel):
    prompt: str
    steps: int = 28
    guidance_scale: float = 4.0
    width: int = 1024
    height: int = 1024
    seed: int = -1
    # "dev"(既定、FLUX.2-dev bnb4bit)| 未指定のみ有効。2026-07-19: ecocoro廃止に伴い、
    # "ecocoro" を明示指定した場合は families/flux2/pipeline.py が ValueError を送出し、
    # _generate_or_409() が 400(「ecocoroは廃止されました」)に変換する。
    model: Optional[str] = None


@app.post("/api/flux2/t2i")
def generate_flux2_t2i(req: Flux2T2IRequest):
    payload = req.model_dump()
    return _generate_or_409("flux2_t2i", payload, load_kwargs={"model": payload.get("model")})


@app.post("/api/flux2/i2i")
async def generate_flux2_i2i(
    images: List[UploadFile] = File(...),
    prompt: str = Form(...),
    steps: int = Form(28),
    guidance_scale: float = Form(4.0),
    width: int = Form(1024),
    height: int = Form(1024),
    seed: int = Form(-1),
    # "dev"(既定)| 未指定のみ有効。"ecocoro" は 400(2026-07-19、ecocoro廃止)。
    model: Optional[str] = Form(None),
):
    if not images:
        raise HTTPException(status_code=400, detail="参照画像を1〜10枚アップロードしてください。")
    if len(images) > MAX_FLUX2_REF_IMAGES:
        raise HTTPException(status_code=400, detail=f"参照画像は最大{MAX_FLUX2_REF_IMAGES}枚までです。")

    ref_images = []
    for f in images:
        contents = await f.read()
        try:
            img = _open_upload_image(contents)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")
        ref_images.append(img)

    return _generate_or_409(
        "flux2_i2i",
        {
            "prompt": prompt,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": seed,
            "model": model,
            "_images": ref_images,
        },
        load_kwargs={"model": model},
    )


# ============================================================================
# Z-Image(Tongyi-MAI/Z-Image-Turbo)T2I / I2I / Inpaint。families/z_image が
# qwen_image・flux2 と exclusive_with 関係で登録されているため(3方向相互排他)、
# 生成リクエスト時に他方が常駐していれば registry.load() が自動 unload する。
# ============================================================================
class ZImageT2IRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = z_image.DEFAULT_STEPS
    guidance_scale: float = z_image.DEFAULT_GUIDANCE_SCALE
    width: int = 1024
    height: int = 1024
    seed: int = -1


@app.post("/api/zimage/t2i")
def generate_zimage_t2i(req: ZImageT2IRequest):
    payload = req.model_dump()
    return _generate_or_409("zimage_t2i", payload)


@app.post("/api/zimage/i2i")
async def generate_zimage_i2i(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    strength: float = Form(0.6),
    steps: int = Form(z_image.DEFAULT_STEPS),
    guidance_scale: float = Form(z_image.DEFAULT_GUIDANCE_SCALE),
    width: int = Form(1024),
    height: int = Form(1024),
    seed: int = Form(-1),
):
    contents = await image.read()
    try:
        src_image = _open_upload_image(contents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")

    return _generate_or_409(
        "zimage_i2i",
        {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "strength": strength,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": seed,
            "_image": src_image,
        },
    )


@app.post("/api/zimage/inpaint")
async def generate_zimage_inpaint(
    image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    strength: float = Form(1.0),
    steps: int = Form(z_image.DEFAULT_STEPS),
    guidance_scale: float = Form(z_image.DEFAULT_GUIDANCE_SCALE),
    width: int = Form(1024),
    height: int = Form(1024),
    seed: int = Form(-1),
):
    try:
        src_contents = await image.read()
        src_image = _open_upload_image(src_contents)
        mask_contents = await mask_image.read()
        mask_img = _open_upload_image(mask_contents, "L")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")

    return _generate_or_409(
        "zimage_inpaint",
        {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "strength": strength,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "seed": seed,
            "_image": src_image,
            "_mask_image": mask_img,
        },
    )


# ============================================================================
# LTX-2.3(動画+音声生成)T2V / I2V。families/ltx2 が qwen_image・flux2・z_image と
# exclusive_with 関係で登録されているため(4方向相互排他)、生成リクエスト時に他方が
# 常駐していれば registry.load() が自動 unload する。
# ============================================================================
class LTX2T2VRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""  # 蒸留モデル(guidance_scale=1.0既定)では実質無効
    width: int = ltx2.DEFAULT_WIDTH
    height: int = ltx2.DEFAULT_HEIGHT
    num_frames: int = ltx2.DEFAULT_NUM_FRAMES
    fps: int = ltx2.DEFAULT_FPS
    steps: int = ltx2.DEFAULT_STEPS
    guidance_scale: float = ltx2.DEFAULT_GUIDANCE_SCALE
    seed: int = -1
    audio_out: str = ltx2.DEFAULT_AUDIO_OUT  # "on"(既定、生成音声mux) / "off"(無音mp4)
    upscale: int = 0  # 0(既定、無効)/ 1(latent upsamplerで2x spatialアップスケール)


@app.post("/api/ltx2/t2v")
def generate_ltx2_t2v(req: LTX2T2VRequest):
    payload = req.model_dump()
    return _generate_or_409("ltx2_t2v", payload)


@app.post("/api/ltx2/i2v")
async def generate_ltx2_i2v(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    width: int = Form(ltx2.DEFAULT_WIDTH),
    height: int = Form(ltx2.DEFAULT_HEIGHT),
    num_frames: int = Form(ltx2.DEFAULT_NUM_FRAMES),
    fps: int = Form(ltx2.DEFAULT_FPS),
    steps: int = Form(ltx2.DEFAULT_STEPS),
    guidance_scale: float = Form(ltx2.DEFAULT_GUIDANCE_SCALE),
    seed: int = Form(-1),
    audio_out: str = Form(ltx2.DEFAULT_AUDIO_OUT),
    upscale: int = Form(0),
):
    contents = await image.read()
    try:
        src_image = _open_upload_image(contents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")

    return _generate_or_409(
        "ltx2_i2v",
        {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "audio_out": audio_out,
            "upscale": upscale,
            "_image": src_image,
        },
    )


@app.post("/api/ltx2/flf")
async def generate_ltx2_flf(
    first_image: UploadFile = File(...),
    last_image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    width: int = Form(ltx2.DEFAULT_WIDTH),
    height: int = Form(ltx2.DEFAULT_HEIGHT),
    num_frames: int = Form(ltx2.DEFAULT_NUM_FRAMES),
    fps: float = Form(ltx2.DEFAULT_FLF_FRAME_RATE),
    steps: int = Form(ltx2.DEFAULT_STEPS),
    guidance_scale: float = Form(ltx2.DEFAULT_GUIDANCE_SCALE),
    strength: float = Form(ltx2.DEFAULT_FLF_STRENGTH),
    seed: int = Form(-1),
    audio_out: str = Form(ltx2.DEFAULT_AUDIO_OUT),
    upscale: int = Form(0),
):
    """FLF(First-Last-Frame): 最初と最後のフレーム画像を指定し、間の動画を生成する。

    ComfyUIワークフロー video_ltx2_3-22b-flf-bf8.json の LTXVAddGuide ×2
    (frame_idx=0/-1, strength既定0.7)に対応する。

    upscale=1 でT2V/I2Vと同じ latent upsampler 2段生成(低解像度denoise→2x spatial→
    高解像度decode)を有効化する(2026-07-23追加、CLAUDE.md 43番・51番参照)。
    """
    first_contents = await first_image.read()
    last_contents = await last_image.read()
    try:
        first_img = _open_upload_image(first_contents)
        last_img = _open_upload_image(last_contents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")

    return _generate_or_409(
        "ltx2_flf",
        {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "strength": strength,
            "seed": seed,
            "audio_out": audio_out,
            "upscale": upscale,
            "_first_image": first_img,
            "_last_image": last_img,
        },
    )


@app.post("/api/ltx2/keyframes")
async def generate_ltx2_keyframes(
    images: List[UploadFile] = File(...),
    indices: str = Form(...),
    strengths: str = Form(""),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    width: int = Form(ltx2.DEFAULT_WIDTH),
    height: int = Form(ltx2.DEFAULT_HEIGHT),
    num_frames: int = Form(ltx2.DEFAULT_NUM_FRAMES),
    fps: float = Form(ltx2.DEFAULT_FLF_FRAME_RATE),
    steps: int = Form(ltx2.DEFAULT_STEPS),
    guidance_scale: float = Form(ltx2.DEFAULT_GUIDANCE_SCALE),
    seed: int = Form(-1),
    audio_out: str = Form(ltx2.DEFAULT_AUDIO_OUT),
):
    """任意キーフレーム条件付け(FLFの一般化): N枚の画像を任意フレーム位置・任意strengthで
    条件付けて動画を生成する。

    indices はカンマ区切りの整数文字列(負値可、例 "0,45,-1")で images と同数必須。
    strengths はカンマ区切りのfloat文字列で、省略時は0.7を全画像に適用する(1個だけ
    指定した場合も全画像に適用、families/ltx2/generate.py の run_keyframes() 参照)。
    """
    try:
        parsed_indices = [int(s.strip()) for s in indices.split(",") if s.strip() != ""]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"indices の解析に失敗しました: {exc}")

    strengths = strengths.strip()
    if strengths:
        try:
            parsed_strengths = [float(s.strip()) for s in strengths.split(",") if s.strip() != ""]
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"strengths の解析に失敗しました: {exc}")
    else:
        parsed_strengths = [ltx2.DEFAULT_FLF_STRENGTH]

    loaded_images = []
    for f in images:
        contents = await f.read()
        try:
            loaded_images.append(_open_upload_image(contents))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")

    return _generate_or_409(
        "ltx2_keyframes",
        {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "audio_out": audio_out,
            "_images": loaded_images,
            "_indices": parsed_indices,
            "_strengths": parsed_strengths,
        },
    )


@app.post("/api/ltx2/ia2v")
async def generate_ltx2_ia2v(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    width: int = Form(ltx2.DEFAULT_WIDTH),
    height: int = Form(ltx2.DEFAULT_HEIGHT),
    num_frames: int = Form(0),  # 0(未指定) なら音声長×fpsから自動計算(8n+1丸め)
    fps: float = Form(ltx2.DEFAULT_IA2V_FRAME_RATE),
    steps: int = Form(ltx2.DEFAULT_STEPS),
    guidance_scale: float = Form(ltx2.DEFAULT_GUIDANCE_SCALE),
    seed: int = Form(-1),
    audio_out: str = Form(ltx2.DEFAULT_IA2V_AUDIO_OUT),
):
    """IA2V(Image + Audio to Video): 参照画像 + wav音声 からリップシンク動画を生成する。

    音声latentを入力wav由来のclean latentへ毎ステップx0固定し、映像側だけをdenoiseする
    独自ループ(families/ltx2/generate.py の run_ia2v()、diffusers本体無改変)。
    prompt 未指定時は DEFAULT_IA2V_PROMPT(リップシンク補助の既定文)を使う。
    audio_out: "original"(既定、入力wavの原音をそのままmux)/ "vocoder"(再合成音)/
    "none"(無音mp4)。
    """
    import tempfile

    image_contents = await image.read()
    try:
        src_image = _open_upload_image(image_contents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")

    audio_contents = await audio.read()
    suffix = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"
    tmp_audio = tempfile.NamedTemporaryFile(prefix="ltx2_ia2v_", suffix=suffix, delete=False)
    try:
        tmp_audio.write(audio_contents)
        tmp_audio.flush()
        tmp_audio.close()

        return _generate_or_409(
            "ltx2_ia2v",
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_frames": num_frames or None,
                "fps": fps,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "audio_out": audio_out,
                "_image": src_image,
                "_audio_path": tmp_audio.name,
            },
        )
    finally:
        try:
            os.unlink(tmp_audio.name)
        except OSError:
            pass


@app.post("/api/ltx2/v2a")
async def generate_ltx2_v2a(
    video: UploadFile = File(...),
    prompt: str = Form(""),
    negative_prompt: str = Form(""),
    width: int = Form(ltx2.DEFAULT_WIDTH),
    height: int = Form(ltx2.DEFAULT_HEIGHT),
    num_frames: int = Form(0),  # 0(未指定) なら動画全長を8n+1へ丸める(上限あり)
    steps: int = Form(ltx2.DEFAULT_STEPS),
    guidance_scale: float = Form(ltx2.DEFAULT_GUIDANCE_SCALE),
    seed: int = Form(-1),
):
    """V2A(Video to Audio): 既存動画に合った音声を生成して付与する(IA2Vの鏡像)。

    映像latentを入力動画由来のclean latentへ毎ステップx0固定し、音声latentだけを
    denoiseする独自ループ(families/ltx2/generate.py の run_v2a()、diffusers本体無改変)。
    入力動画の音声(あれば)は無視し、生成した音声で差し替える。
    prompt/negative_prompt 未指定時はそれぞれ DEFAULT_V2A_PROMPT / DEFAULT_V2A_NEGATIVE_PROMPT
    (環境音を想定した既定文)を使う。
    """
    import tempfile

    video_contents = await video.read()
    suffix = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    tmp_video = tempfile.NamedTemporaryFile(prefix="ltx2_v2a_", suffix=suffix, delete=False)
    try:
        tmp_video.write(video_contents)
        tmp_video.flush()
        tmp_video.close()

        return _generate_or_409(
            "ltx2_v2a",
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_frames": num_frames or None,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "_video_path": tmp_video.name,
            },
        )
    finally:
        try:
            os.unlink(tmp_video.name)
        except OSError:
            pass


@app.post("/api/ltx2/iclora")
async def generate_ltx2_iclora(
    video: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    mask_mode: str = Form("middle"),
    mask_start: Optional[int] = Form(None),
    mask_end: Optional[int] = Form(None),
    width: int = Form(ltx2.DEFAULT_WIDTH),
    height: int = Form(ltx2.DEFAULT_HEIGHT),
    num_frames: int = Form(0),  # 0(未指定) なら動画全長を8n+1へ丸める(上限あり)
    fps: float = Form(ltx2.DEFAULT_FLF_FRAME_RATE),
    steps: int = Form(ltx2.DEFAULT_STEPS),
    guidance_scale: float = Form(ltx2.DEFAULT_GUIDANCE_SCALE),
    ref_strength: float = Form(ltx2.DEFAULT_ICLORA_REF_STRENGTH),
    seed: int = Form(-1),
):
    """IC-LoRA(MergeGreen): 参照動画の中央部を緑マスクし、prompt で指示した変化に沿って
    動画を生成し直す動画編集モード(HF repo siraxe/MergeGreen_IC-lora_ltx2.3、Apache-2.0)。

    prompt には「シーン内で何が変わるか」を記述する(例: "the apple turns into an
    orange")。mask_mode="middle"(既定)ならサーバ側で中央フレーム(mask_start/mask_end
    未指定時は動画全体の中央1/3)を緑 RGB(0,191,0) で塗りつぶす。mask_mode="prefilled"
    なら、既に緑マスク済みの動画をそのまま reference_conditions へ渡す
    (families/ltx2/generate.py の run_iclora() 参照)。
    """
    import tempfile

    video_contents = await video.read()
    suffix = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    tmp_video = tempfile.NamedTemporaryFile(prefix="ltx2_iclora_", suffix=suffix, delete=False)
    try:
        tmp_video.write(video_contents)
        tmp_video.flush()
        tmp_video.close()

        return _generate_or_409(
            "ltx2_iclora",
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "mask_mode": mask_mode,
                "mask_start": mask_start,
                "mask_end": mask_end,
                "width": width,
                "height": height,
                "num_frames": num_frames or None,
                "fps": fps,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "ref_strength": ref_strength,
                "seed": seed,
                "_video_path": tmp_video.name,
            },
        )
    finally:
        try:
            os.unlink(tmp_video.name)
        except OSError:
            pass


# ============================================================================
# JoyAI-Image-Edit-Plus(複数参照画像編集)。families/joyai が qwen_image・flux2・
# z_image・ltx2 と exclusive_with 関係で登録されているため(5方向相互排他)、生成
# リクエスト時に他方が常駐していれば registry.load() が自動 unload する。
# ============================================================================
@app.post("/api/joyai/edit")
async def generate_joyai_edit(
    images: List[UploadFile] = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    steps: int = Form(joyai.DEFAULT_STEPS),
    guidance_scale: float = Form(joyai.DEFAULT_GUIDANCE_SCALE),
    seed: int = Form(-1),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
):
    """JoyAI-Image-Edit-Plus: 複数参照画像(最大3枚)を渡し、プロンプトで指示した編集・合成を
    行う(jdopensource/JoyAI-Image-Edit-Plus-Diffusers、Apache 2.0)。width/height 未指定時は
    最後の参照画像のサイズから自動推定される(JoyImageEditPlusPipeline 内部の既定挙動)。
    """
    if not images:
        raise HTTPException(status_code=400, detail="参照画像を1〜3枚アップロードしてください。")
    if len(images) > joyai.MAX_REF_IMAGES:
        raise HTTPException(status_code=400, detail=f"参照画像は最大{joyai.MAX_REF_IMAGES}枚までです。")

    src_images = []
    for f in images:
        contents = await f.read()
        try:
            img = _open_upload_image(contents)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")
        src_images.append(img)

    return _generate_or_409(
        "joyai_edit",
        {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "width": width,
            "height": height,
            "_images": src_images,
        },
    )


# ============================================================================
# unload / status
# ============================================================================
class UnloadRequest(BaseModel):
    # "t2i" | "edit" | "controlnet" | "layered" | "flux2" | "zimage" | "ltx2" | "joyai" |
    # "all"(旧互換: 無指定 = 全部)
    target: str = "all"


@app.post("/api/unload")
def unload(req: UnloadRequest):
    if req.target == "flux2":
        return {"flux2": flux2.family.unload()}
    if req.target == "zimage":
        return {"z_image": z_image.family.unload()}
    if req.target == "ltx2":
        return {"ltx2": ltx2.family.unload()}
    if req.target == "joyai":
        return {"joyai": joyai.family.unload()}
    if req.target not in ("t2i", "edit", "controlnet", "layered", "all"):
        raise HTTPException(
            status_code=400,
            detail=(
                "target は t2i / edit / controlnet / layered / flux2 / zimage / ltx2 / "
                "joyai / all のいずれかです"
            ),
        )
    if req.target == "all":
        # 旧互換: 無指定(既定値 "all")は登録済み全ファミリー
        # (qwen_image + flux2 + z_image + ltx2 + joyai)を解放する。
        return registry.unload(None)
    result = qwen_image.family.unload_target(req.target)
    return result


@app.get("/api/progress")
def get_progress():
    """生成中の進捗状態(グローバル1本、core/progress.py)を返す。

    非GPU・排他不要の読み取り専用エンドポイント(core.gpu.generation_lock を一切
    使わない)。生成リクエストと同時に、UIから500ms間隔等でポーリングしてよい。
    """
    return progress_mod.snapshot()


@app.get("/api/status")
def status():
    st = registry.status_all()
    st["last_generation"] = _last_generation
    # 旧クライアント互換: qwen_image 固有のキーをトップレベルにも展開する
    # (旧 static/app.js は data.t2i_loaded 等をトップレベルから読むため)。
    qwen_status = st.get("qwen_image", {})
    merged = dict(qwen_status)
    merged.update(st)
    merged.pop("qwen_image", None)
    # flux2 は独立した状態を持つファミリーなので、旧クライアント互換の展開はせず
    # st["flux2"] のネスト構造のまま残す(merged.update(st) で既に含まれている)。
    return merged


# ============================================================================
# LLMプロンプト支援(非GPU・排他不要。core.gpu.generation_lock を使わないため、
# 画像生成中でも呼べるし、画像生成側には一切影響しない)
# ============================================================================
class PromptLLMRequest(BaseModel):
    text: str


@app.post("/api/prompt/enhance")
def prompt_enhance(req: PromptLLMRequest):
    start = time.time()
    try:
        result = llm.enhance_prompt(req.text)
    except llm.LLMConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"result": result, "elapsed_s": time.time() - start}


@app.post("/api/prompt/translate")
def prompt_translate(req: PromptLLMRequest):
    start = time.time()
    try:
        result = llm.translate_prompt(req.text)
    except llm.LLMConnectionError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {"result": result, "elapsed_s": time.time() - start}


# ============================================================================
# Mage-Flow プロキシ(2026-07-22追加、CLAUDE.md 50番)
#
# Mage-Flow はバージョン衝突(torch 2.13 / transformers 5.5 / flash-attn 必須)のため
# 本体プロセスに載せられず、専用venv(venv-mageflow/)の別プロセス
# (mageflow_service/app_mageflow.py、DS_MAGEFLOW_URL 既定 http://127.0.0.1:8602)で動く。
# ここは core/mageflow.py 経由の純粋なHTTP転送のみ(registry.load() は呼ばない —
# Mage-Flow のモデルはプロセス外にあり、FamilyRegistry の排他unload機構の対象外)。
#
# 排他制御はリクエストごとに選択式(exclusive パラメータ、既定 true):
#   - exclusive=true : core.gpu.generation_lock を非ブロッキング取得してから転送
#     (取得失敗は409。_generate_or_409() と同じ流儀)。同一GPUを共有する他ファミリーの
#     生成と VRAM ピークが重ならないようにする既定動作。逆方向(他ファミリー生成中は
#     Mage-Flow が409になる/Mage-Flow 生成中は他ファミリーが409になる)も同じロック
#     1本で担保される。
#   - exclusive=false: ロックを取らず即転送(並行実行を許可)。Mage-Flow は4.1B・
#     ピーク18-20GB程度なので、96GB機で空きVRAMに余裕がある場合はユーザーの明示
#     選択で他の生成と並行できる。VRAM合算がオーバーする組み合わせを選ぶと
#     どちらかがCUDA OOMで失敗するリスクはユーザー持ち。
# ============================================================================
from core import mageflow as mageflow_client  # noqa: E402


def _mageflow_errors_to_http(exc: Exception) -> HTTPException:
    """core/mageflow.py の例外をHTTPレスポンスへ変換する(LLM連携の502変換と同じ流儀)。"""
    if isinstance(exc, mageflow_client.MageFlowConnectionError):
        return HTTPException(status_code=502, detail=str(exc))
    if isinstance(exc, mageflow_client.MageFlowServiceError):
        # ラッパー側のバリデーション(400)や生成失敗(500)はステータスごと透過する
        return HTTPException(status_code=exc.status_code, detail=exc.detail)
    return HTTPException(status_code=500, detail=f"Mage-Flowプロキシで予期しないエラー: {exc}")


def _mageflow_forward_with_exclusivity(exclusive: bool, forward_fn):
    """exclusive=true なら generation_lock を非ブロッキング取得してから転送する。

    _generate_or_409() と同じ409応答パターンだが、registry.load() は呼ばない
    (Mage-Flow はプロセス外のため FamilyRegistry のロード/排他unload対象ではない)。
    ロック保持中は他ファミリーの生成が409になり、逆に他ファミリー生成中は
    こちらが409になる(全ファミリー横断ロック1本の効果をそのまま利用)。
    """
    from core import gpu

    if not exclusive:
        try:
            return forward_fn()
        except (mageflow_client.MageFlowConnectionError, mageflow_client.MageFlowServiceError) as exc:
            raise _mageflow_errors_to_http(exc)
    if not gpu.generation_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409, detail="別の生成が実行中です。しばらく待ってから再試行してください。"
        )
    try:
        return forward_fn()
    except (mageflow_client.MageFlowConnectionError, mageflow_client.MageFlowServiceError) as exc:
        raise _mageflow_errors_to_http(exc)
    finally:
        gpu.generation_lock.release()


class MageFlowT2IRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    steps: Optional[int] = None  # 未指定はバリアント既定(base30/rl20/turbo4)
    cfg: Optional[float] = None  # 未指定はバリアント既定(base/rl 5.0, turbo 1.0)
    seed: int = -1
    model: str = "rl"  # "base" | "rl" | "turbo"
    exclusive: bool = True  # 既定で他の生成と排他(上のコメント参照)


@app.post("/api/mageflow/t2i")
def mageflow_t2i(req: MageFlowT2IRequest):
    payload = req.model_dump()
    exclusive = payload.pop("exclusive")

    def _forward():
        meta = mageflow_client.forward_t2i(payload)
        meta["exclusive"] = exclusive
        _record_last_generation(meta)
        return meta

    return _mageflow_forward_with_exclusivity(exclusive, _forward)


@app.post("/api/mageflow/edit")
async def mageflow_edit(
    image: List[UploadFile] = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    steps: Optional[int] = Form(None),
    cfg: Optional[float] = Form(None),
    seed: int = Form(-1),
    max_size: int = Form(1024),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    model: str = Form("rl"),
    exclusive: bool = Form(True),
):
    # アップロードをメモリへ読み込んでからラッパーへ multipart 転送する
    # (EXIF正規化・RGB変換等の前処理はラッパー側 mageflow_service/app_mageflow.py が行う)。
    files = []
    for up in image:
        data = await up.read()
        files.append(("image", (up.filename or "image.png", data, up.content_type or "image/png")))
    form_data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": str(seed),
        "max_size": str(max_size),
        "model": model,
    }
    if steps is not None:
        form_data["steps"] = str(steps)
    if cfg is not None:
        form_data["cfg"] = str(cfg)
    if width is not None:
        form_data["width"] = str(width)
    if height is not None:
        form_data["height"] = str(height)

    def _forward():
        meta = mageflow_client.forward_edit(form_data, files)
        meta["exclusive"] = exclusive
        _record_last_generation(meta)
        return meta

    return _mageflow_forward_with_exclusivity(exclusive, _forward)


@app.get("/api/mageflow/status")
def mageflow_status():
    try:
        return mageflow_client.forward_status()
    except mageflow_client.MageFlowConnectionError as exc:
        # ステータスは「未起動」を200で返す(UIのステータスバーが8秒ごとにポーリング
        # するため、未起動状態を毎回502エラー扱いにしない)。
        return {"service": "mageflow", "available": False, "detail": str(exc)}
    except mageflow_client.MageFlowServiceError as exc:
        raise _mageflow_errors_to_http(exc)


@app.post("/api/mageflow/unload")
def mageflow_unload():
    try:
        return mageflow_client.forward_unload()
    except (mageflow_client.MageFlowConnectionError, mageflow_client.MageFlowServiceError) as exc:
        raise _mageflow_errors_to_http(exc)


# ============================================================================
# charsheet(Phase 3。apps/charsheet/ 側のルーターを /api/charsheet 接頭辞で統合)
# ============================================================================
app.include_router(charsheet_router, prefix="/api/charsheet")

# ============================================================================
# scene_angles(2026-07-24追加、CLAUDE.md 53番。1枚のシーン画像から
# カメラ指示プロンプト8種で同一シーンの8アングル画像を生成するジョブ式API。
# パイプラインは charsheet と同じ edit_angles 系グループを流用)
# ============================================================================
app.include_router(scene_angles_router, prefix="/api/scene_angles")


# ============================================================================
# 生成画像・静的ファイル配信
# ============================================================================
class NoCacheStaticFiles(StaticFiles):
    """静的ファイルにブラウザキャッシュを効かせない(開発中の反映漏れ防止)。"""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
