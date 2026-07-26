# -*- coding: utf-8 -*-
"""Tポーズ4ビュー生成ジョブ管理 + API(image-3d / rig-service 向け)。

apps/scene_angles/jobs.py のジョブ機構(投入→バックグラウンドスレッド→ポーリング)を
踏襲する。相違点:

  - 生成呼び出しは apps.tpose.generate(通常 Edit、angles LoRA なし)。
  - **2段生成**: front を最初に生成し、back / 45度2枚はその front 出力を参照画像に
    連鎖生成する(元画像から直接背面化すると帽子・尻尾等の造形が前面と食い違う)。
  - しっぽ参照画像(tail_ref、任意)を渡すと front 以外のビューで2枚目の参照として
    使う。**同一性が参照画像側へ引きずられる副作用**(実機で頭部が黒髪化)を確認して
    いるため既定は使わず、テキスト指定(tail)を推奨する。
  - 4ビューそれぞれを個別ダウンロードできるエンドポイント(?download=1 または
    /download/{key}.png)と、4枚まとめてのZIPを提供する。

同時実行制御:
  - tposeジョブ同士は current_job_id で同時1件(charsheet/scene_anglesと同じ流儀)。
  - 他アプリ(charsheet/scene_angles)や通常APIとの排他は core.gpu.generation_lock の
    blocking取得で自然に直列化される。
"""
import io
import os
import shutil
import threading
import traceback
import uuid
import zipfile
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps

from apps.tpose import generate as generate_mod
from apps.tpose.prompts import (
    BODY_PRESETS,
    NEGATIVE_PROMPT,
    PALMS_MODES,
    SUBJECT_MODES,
    TAIL_PRESETS,
    VIEW_BY_KEY,
    VIEWS,
    build_prompt,
)
from core import gpu
from core import progress as progress_mod

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "tpose")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

router = APIRouter()

jobs = {}  # job_id -> dict
jobs_lock = threading.Lock()
current_job_id = None
current_job_lock = threading.Lock()


def _job_dir(job_id: str) -> str:
    d = os.path.join(OUTPUTS_DIR, job_id)
    os.makedirs(d, exist_ok=True)
    return d


def _check_job_id(job_id: str) -> None:
    if not job_id or "/" in job_id or "\\" in job_id or ".." in job_id:
        raise HTTPException(status_code=404, detail="不正な job_id です")


def _parse_views(raw: str) -> list:
    """views パラメータ(カンマ区切り)を VIEWS のサブセットへ解決する。

    空・未指定なら4ビュー全部。未知のIDは400。順序は VIEWS 定義順に正規化する
    (front を必ず最初に生成する必要があるため、指定順に依存させない)。
    front を含まない指定も許可する(front 相当の画像を入力として渡す運用)。
    """
    if not raw or not raw.strip():
        return list(VIEWS)
    requested = {token.strip() for token in raw.split(",") if token.strip()}
    unknown = requested - set(VIEW_BY_KEY.keys())
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=(
                f"不正なビューIDです: {sorted(unknown)}。"
                f"有効なID: {[v['key'] for v in VIEWS]}"
            ),
        )
    return [v for v in VIEWS if v["key"] in requested]


def _init_job(job_id: str, seed: int, views: list, params: dict):
    entries = [
        {
            "key": v["key"],
            "label_ja": v["label_ja"],
            "label_en": v["label_en"],
            "for_3d": v["for_3d"],
            "status": "queued",
            "url": None,
            "download_url": None,
            "nobg_url": None,
            "nobg_download_url": None,
            "prompt": None,
        }
        for v in views
    ]
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "total": len(views),
            "seed": seed,
            "params": params,
            "views": entries,
            "zip_url": None,
            "error": None,
            "created_at": datetime.now().isoformat(),
            "load_info": None,
        }


def _update_job(job_id: str, **kwargs):
    with jobs_lock:
        jobs[job_id].update(kwargs)


def _update_view(job_id: str, key: str, **kwargs):
    with jobs_lock:
        for v in jobs[job_id]["views"]:
            if v["key"] == key:
                v.update(kwargs)
                break


def _copy_generated_image(meta: dict, dest_path: str) -> None:
    image_url = meta.get("image_url", "")
    name = image_url.rsplit("/", 1)[-1]
    src_path = os.path.join(BASE_DIR, "outputs", name)
    shutil.copy2(src_path, dest_path)


def _build_zip(job_dir: str, keys: list) -> Optional[str]:
    """生成済みビューをZIPにまとめる(一括ダウンロード用)。

    背景透過版(`<key>_nobg.png`)があれば併せて含める。
    """
    paths = [(k, os.path.join(job_dir, f"{k}.png")) for k in keys]
    paths = [(k, p) for k, p in paths if os.path.exists(p)]
    if not paths:
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for key, path in paths:
            zf.write(path, arcname=f"{key}.png")
            nobg = os.path.join(job_dir, f"{key}_nobg.png")
            if os.path.exists(nobg):
                zf.write(nobg, arcname=f"{key}_nobg.png")
        input_path = os.path.join(job_dir, "input.png")
        if os.path.exists(input_path):
            zf.write(input_path, arcname="input.png")
    zip_path = os.path.join(job_dir, "download.zip")
    with open(zip_path, "wb") as f:
        f.write(buf.getvalue())
    return zip_path


def _cutout_rgba(src_rgb: Image.Image, rembg_rgba: Image.Image) -> Image.Image:
    """rembg の出力から背景透過画像(RGBA)を組み立てる。

    実機で出た2つの不具合への対処(momo.png のTポーズ正面で確認):
      - **アルファに内部の穴が空く**: オーバーオールの帯の間が半透明(alpha≈160)になり、
        暗い色が透けて見えた。`scipy.ndimage.binary_fill_holes` で「境界から到達できない
        透明領域」を不透明に埋める(scipy は comfy-env に既存。import できない環境では
        穴埋めをスキップして従来どおりの出力にフォールバックする)。
      - **RGBが濁る**: rembg 出力のRGBは穴の周辺で暗く濁っていたため、RGBは
        **元の白背景画像から取り直す**(アルファだけ rembg から使う)。
    さらに、ほぼ不透明な画素(alpha>=250、isnet の出力は被写体でも 254 になる)は 255 へ
    丸める(下流ツールの `alpha == 255` 判定が効くようにするため)。
    """
    import numpy as np

    alpha = np.array(rembg_rgba.getchannel("A"))
    try:
        from scipy import ndimage

        solid = alpha > 128
        filled = ndimage.binary_fill_holes(solid)
        alpha = np.where(filled & ~solid, 255, alpha)
    except ImportError:  # pragma: no cover - scipy が無い環境向けのフォールバック
        print("[apps.tpose] warning: scipy が無いためアルファの穴埋めをスキップします")
    alpha = np.where(alpha >= 250, 255, alpha).astype(np.uint8)
    out = src_rgb.convert("RGBA")
    out.putalpha(Image.fromarray(alpha, mode="L"))
    return out


def _remove_bg_pass(job_id: str, job_dir: str, keys: list) -> None:
    """生成済み各ビューの背景を除去し `<key>_nobg.png`(RGBA)として保存する。

    白背景版(`<key>.png`)は残す(3Dツール側が白背景を前提にする場合や、
    切り抜き失敗時の退避のため)。rembg はCPU(ONNX)処理なので **GPUロックを
    解放した後に**呼ぶこと(呼び出し側 `_run_job` がそうしている)。
    1枚失敗しても他のビューの処理は続ける(部分的な成功を許容する)。
    """
    from apps.charsheet.bg import remove_background

    for key in keys:
        src = os.path.join(job_dir, f"{key}.png")
        if not os.path.exists(src):
            continue
        try:
            with Image.open(src) as im:
                src_rgb = im.convert("RGB")
                rgba = _cutout_rgba(src_rgb, remove_background(src_rgb))
            rgba.save(os.path.join(job_dir, f"{key}_nobg.png"))
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            _update_view(job_id, key, nobg_error=str(exc))
            continue
        _update_view(
            job_id, key,
            nobg_url=f"/api/tpose/jobs/{job_id}/images/{key}_nobg.png",
            nobg_download_url=f"/api/tpose/jobs/{job_id}/download/{key}_nobg.png",
        )


def _run_job(job_id: str, input_path: str, tail_ref_path: Optional[str], seed: int,
             views: list, params: dict):
    global current_job_id
    job_dir = _job_dir(job_id)
    got_lock = False
    try:
        _update_job(job_id, status="running")

        input_image = Image.open(input_path).convert("RGB")
        processed = generate_mod.preprocess_image(input_image)
        processed.save(os.path.join(job_dir, "input.png"))

        tail_ref = None
        if tail_ref_path:
            tail_ref = generate_mod.preprocess_image(
                Image.open(tail_ref_path).convert("RGB")
            )

        got_lock = generate_mod.acquire_generation_lock(blocking=True)
        gpu.empty_cache()
        generate_mod.ensure_edit_loaded()

        # front を最初に生成し、以降のビューはその出力を参照画像にする(2段生成)。
        front_image = None
        total = len(views)
        for idx, view in enumerate(views):
            key = view["key"]
            # 1段目(元画像からポーズを変える)判定: front、または front 未選択時の最初のビュー
            first_stage = (key == "front" or front_image is None)
            prompt = build_prompt(
                key,
                subject=params.get("subject", "auto"),
                palms=params["palms"],
                paw_pads=params["paw_pads"],
                tail=params["tail"],
                body=params["body"],
                extra=params["extra_prompt"],
                first_stage=first_stage,
            )
            _update_view(job_id, key, status="running", prompt=prompt)

            if first_stage:
                # front(または front 未選択時の最初のビュー)は元画像から生成する
                refs = [processed]
            else:
                # front 以外は「生成した正面Tポーズ」を主参照にする(ポーズ・構図を
                # 揃えるため)。加えて **元画像も2枚目の参照として渡す**: 正面画像
                # だけを参照にすると、背面ビューで後頭部が黒髪の人型へ変質する
                # ドリフトが実機で発生した(CLAUDE.md 34番と同種)。元画像を併せて
                # 渡すと毛並み・耳・色の手がかりが増え、この変質が起きにくくなる。
                # MAX_EDIT_IMAGES=3 のため最大3枚(正面 + 元画像 + しっぽ参照)。
                refs = [front_image, processed]
                if tail_ref is not None:
                    refs.append(tail_ref)

            try:
                meta = generate_mod.generate_view(
                    refs,
                    prompt=prompt,
                    seed=seed,
                    negative_prompt=NEGATIVE_PROMPT,
                    progress_extra={
                        "job_id": job_id,
                        "direction": key,
                        "direction_label": view.get("label_ja"),
                        "direction_index": idx + 1,
                        "direction_total": total,
                        "app": "tpose",
                    },
                )
                out_path = os.path.join(job_dir, f"{key}.png")
                _copy_generated_image(meta, out_path)
                if key == "front":
                    front_image = Image.open(out_path).convert("RGB")
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                _update_view(job_id, key, status="error")
                _update_job(job_id, status="error", error=f"{key}: {exc}")
                return

            with jobs_lock:
                jobs[job_id]["progress"] += 1
                progress = jobs[job_id]["progress"]
            _update_view(
                job_id, key, status="done",
                url=f"/api/tpose/jobs/{job_id}/images/{key}.png",
                download_url=f"/api/tpose/jobs/{job_id}/download/{key}.png",
            )
            _update_job(job_id, progress=progress)

        keys = [v["key"] for v in views]

        # 背景除去(任意)は rembg(CPU/ONNX)なので、**先にGPUロックを解放**してから
        # 実行する(他のGPUジョブを不要に待たせない。1枚あたり1秒前後)。
        if params.get("remove_bg"):
            if got_lock:
                generate_mod.release_generation_lock()
                got_lock = False
            _update_job(job_id, status="removing_bg")
            _remove_bg_pass(job_id, job_dir, keys)

        zip_path = _build_zip(job_dir, keys)
        _update_job(
            job_id,
            status="done",
            zip_url=f"/api/tpose/jobs/{job_id}/download.zip" if zip_path else None,
            load_info=generate_mod.get_load_info(),
        )
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        _update_job(job_id, status="error", error=str(exc))
    finally:
        progress_mod.finish()
        if got_lock:
            generate_mod.release_generation_lock()
        with current_job_lock:
            current_job_id = None


@router.post("/generate")
async def generate(
    image: UploadFile = File(...),
    tail_ref: Optional[UploadFile] = File(None),
    seed: int = Form(0),
    views: str = Form(""),
    subject: str = Form("auto"),
    palms: str = Form("forward"),
    paw_pads: str = Form("auto"),
    tail: str = Form(""),
    body: str = Form(""),
    extra_prompt: str = Form(""),
    remove_bg: bool = Form(False),
):
    """Tポーズ4ビュー生成ジョブを開始する。

    - image: 入力キャラクター画像(必須、multipart)。全身でも胸像でもよい
      (胸像からでも全身Tポーズを生成できることを実機確認済み。ただし写っていない
      脚部の衣装はモデルが創作する)
    - tail_ref: しっぽ形状の参照画像(任意)。front 以外のビューで2枚目の参照として
      使う。**同一性が参照画像側へ引きずられる副作用があるため非推奨**
      (テキスト指定 tail を推奨)
    - seed: 乱数シード(任意、既定0=ビューごとにランダム)
    - views: カンマ区切りのビューID(任意。省略=4ビュー全部)。
      有効ID: front / back / front_left_45 / front_right_45
    - subject: 被写体タイプ。"auto"(既定、動物/人間どちらの語彙も使わない中立)|
      "animal"(毛皮・肉球のある動物やぬいぐるみ)| "human"(人物・リアルな人形)。
      **リアルな人形で背面が動物化する/手に肉球が付く場合は "human" を指定する**
    - palms: "forward"(既定、手のひらをカメラへ向ける=リグ用Tポーズの標準)|
      "natural"(指示しない)
    - paw_pads: "auto"(既定、参照画像の肉球色を踏襲)| "none"(肉球に言及しない)|
      自由記述(例 "pink", "dark brown")
    - body: 体型の自由記述(例 "short stubby legs and a large head")。**脚が伸びる
      劣化への主要な対処**(apps/tpose/prompts.py の「脚が伸びる問題」参照)。
      ぬいぐるみ・デフォルメ体型では指定を推奨。空なら何も足さない
    - tail: しっぽ形状の自由記述(例 "a long fluffy tail with a black tip")。
      "none" でしっぽなし、空/"auto" で指定なし(未指定だとビューごとに形状が
      揺れるため、入力画像にしっぽが写っていない場合は指定推奨)
    - extra_prompt: プロンプト末尾へ追記する自由記述(任意)
    - remove_bg: 背景除去(rembg / isnet-general-use)。true なら各ビューの背景透過版
      `<key>_nobg.png`(RGBA)を併せて生成する(白背景版はそのまま残す)。
      生成完了後・GPUロック解放後にCPUで処理するため、GPU待ちは発生しない
    """
    global current_job_id

    selected = _parse_views(views)
    if palms not in PALMS_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"palms は {list(PALMS_MODES)} のいずれかです。",
        )
    if subject not in SUBJECT_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"subject は {list(SUBJECT_MODES)} のいずれかです。",
        )

    with current_job_lock:
        if current_job_id is not None:
            raise HTTPException(
                status_code=409,
                detail="別のTポーズ生成ジョブが実行中です。しばらく待ってから再試行してください。",
            )
        job_id = uuid.uuid4().hex[:12]
        current_job_id = job_id

    job_dir = _job_dir(job_id)
    input_path = os.path.join(job_dir, "upload_raw.png")
    tail_ref_path = None

    try:
        contents = await image.read()
        # EXIF回転を実ピクセルへ適用してから保存(charsheetと同じ理由)
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(contents)))
        img.save(input_path)
        with Image.open(input_path) as im:
            im.verify()
        if tail_ref is not None:
            tail_contents = await tail_ref.read()
            if tail_contents:
                tail_ref_path = os.path.join(job_dir, "tail_ref.png")
                timg = ImageOps.exif_transpose(Image.open(io.BytesIO(tail_contents)))
                timg.save(tail_ref_path)
                with Image.open(tail_ref_path) as im:
                    im.verify()
    except HTTPException:
        raise
    except Exception as exc:
        with current_job_lock:
            current_job_id = None
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")

    params = {
        "remove_bg": bool(remove_bg),
        "subject": subject,
        "palms": palms,
        "paw_pads": paw_pads,
        "tail": tail,
        "body": body,
        "extra_prompt": extra_prompt,
        "tail_ref": bool(tail_ref_path),
        "size": generate_mod.edit_size(),
    }
    _init_job(job_id, seed, selected, params)

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, input_path, tail_ref_path, seed, selected, params),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    _check_job_id(job_id)
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="ジョブが見つかりません")
        return JSONResponse(dict(job))


def _resolve_view_file(key: str) -> str:
    """URLの `{key}` を許可リストで検証してファイル名へ解決する。

    `<view>` は白背景版、`<view>_nobg` は背景透過版(remove_bg=true時のみ存在)。
    """
    base = key[:-5] if key.endswith("_nobg") else key
    if base not in VIEW_BY_KEY:
        raise HTTPException(status_code=404, detail="不正なビューIDです")
    return f"{key}.png"


@router.get("/jobs/{job_id}/images/{key}.png")
async def get_view_image(job_id: str, key: str):
    """ビュー画像の表示用(inline)。`{key}` に `_nobg` 付きを指定すると背景透過版。"""
    _check_job_id(job_id)
    _resolve_view_file(key)
    path = os.path.join(OUTPUTS_DIR, job_id, f"{key}.png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="画像が見つかりません")
    return FileResponse(path, media_type="image/png")


@router.get("/jobs/{job_id}/download/{key}.png")
async def download_view_image(job_id: str, key: str):
    """ビュー画像の個別ダウンロード(Content-Disposition: attachment)。

    ファイル名は tpose_<key>_<job_id>.png(image-3d のビュー割り当てで
    front/back が判別しやすいように key を先頭近くに入れる)。
    `{key}` に `_nobg` 付きを指定すると背景透過版をダウンロードする。
    """
    _check_job_id(job_id)
    _resolve_view_file(key)
    path = os.path.join(OUTPUTS_DIR, job_id, f"{key}.png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="画像が見つかりません")
    return FileResponse(
        path, media_type="image/png", filename=f"tpose_{key}_{job_id}.png"
    )


@router.get("/jobs/{job_id}/download.zip")
async def download_zip(job_id: str):
    """生成済み全ビュー + 入力画像のZIP一括ダウンロード。"""
    _check_job_id(job_id)
    path = os.path.join(OUTPUTS_DIR, job_id, "download.zip")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="ZIP が見つかりません")
    return FileResponse(
        path, media_type="application/zip", filename=f"tpose_{job_id}.zip"
    )


@router.get("/jobs/{job_id}/input.png")
async def get_input_image(job_id: str):
    _check_job_id(job_id)
    path = os.path.join(OUTPUTS_DIR, job_id, "input.png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="画像が見つかりません")
    return FileResponse(path, media_type="image/png")


@router.get("/views")
async def list_views():
    """利用可能なビューID・しっぽプリセットの一覧(UI用)。GPU不使用。

    for_3d は Hunyuan3D-2mv(image-3d)のビュースロット(front/left/back/right)へ
    そのまま渡してよいかどうか。45度ビューは False(参考出力であり、left/right
    スロットへ入れるとカメラ事前分布を誤らせる)。
    """
    return {
        "views": [
            {
                "key": v["key"],
                "label_ja": v["label_ja"],
                "label_en": v["label_en"],
                "for_3d": v["for_3d"],
            }
            for v in VIEWS
        ],
        "tail_presets": TAIL_PRESETS,
        "body_presets": BODY_PRESETS,
        "palms_modes": list(PALMS_MODES),
        "subject_modes": list(SUBJECT_MODES),
    }
