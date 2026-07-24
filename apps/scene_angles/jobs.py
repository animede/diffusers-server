# -*- coding: utf-8 -*-
"""
シーンアングル生成ジョブ管理 + API。

apps/charsheet/jobs.py のジョブ機構(投入→バックグラウンドスレッド→ポーリング)を
踏襲した簡易版。charsheet が持つ refine / undo / remove_bg / split / シート合成は
シーンアングルでは不要なため実装しない(必要になったら charsheet の該当実装を参照)。

生成呼び出しは apps.charsheet.generate をそのまま流用する
(edit_angles系グループ・DS_CHARSHEET_METHOD・CHARSHEET_EDIT_SIZE・GPUロックの
挙動はすべて charsheet と共通。charsheet 側コードは無変更)。

同時実行制御:
  - scene_angles ジョブ同士は current_job_id で同時1件に制限(charsheetと同じ流儀)。
  - charsheet ジョブとの相互排他は行わない(ジョブスレッド内で
    core.gpu.generation_lock を blocking 取得するため、先行ジョブの完了を
    自然に待つ。1方向ごとにロックを持ち続ける設計も charsheet と同一)。
"""
import io
import os
import shutil
import threading
import traceback
import uuid
from datetime import datetime

from fastapi import APIRouter, File, Form, HTTPException
from fastapi import UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps

from apps.charsheet import generate as generate_mod
from apps.scene_angles.prompts import ANGLE_BY_KEY, ANGLES, NEGATIVE_PROMPT
from core import gpu
from core import progress as progress_mod

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "scene_angles")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

router = APIRouter()

# --- ジョブ管理(メモリ内、charsheetと同じ流儀) ---
jobs = {}  # job_id -> dict
jobs_lock = threading.Lock()
current_job_id = None  # 実行中ジョブ(scene_anglesジョブ同士の同時1件制御)
current_job_lock = threading.Lock()


def _job_dir(job_id: str) -> str:
    d = os.path.join(OUTPUTS_DIR, job_id)
    os.makedirs(d, exist_ok=True)
    return d


def _parse_angles(raw: str) -> list:
    """angles パラメータ(カンマ区切りID)を ANGLES のサブセットへ解決する。

    空・未指定なら8種全部。未知のIDは 400。順序は ANGLES 定義順に正規化する
    (呼び出し側の指定順に依存させない)。
    """
    if not raw or not raw.strip():
        return list(ANGLES)
    requested = {token.strip() for token in raw.split(",") if token.strip()}
    unknown = requested - set(ANGLE_BY_KEY.keys())
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=(
                f"不正なアングルIDです: {sorted(unknown)}。"
                f"有効なID: {[a['key'] for a in ANGLES]}"
            ),
        )
    return [a for a in ANGLES if a["key"] in requested]


def _init_job(job_id: str, seed: int, angles: list):
    views = [
        {
            "key": a["key"],
            "label_ja": a["label_ja"],
            "label_en": a["label_en"],
            "status": "queued",
            "url": None,
        }
        for a in angles
    ]
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "total": len(angles),
            "seed": seed,
            "views": views,
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
    """generate_view() の meta["image_url"](/outputs/xxx.png)をジョブディレクトリへ
    コピーする(charsheetの同名ヘルパと同じ)。"""
    image_url = meta.get("image_url", "")
    name = image_url.rsplit("/", 1)[-1]
    src_path = os.path.join(BASE_DIR, "outputs", name)
    shutil.copy2(src_path, dest_path)


def _run_job(job_id: str, input_path: str, seed: int, angles: list):
    global current_job_id
    job_dir = _job_dir(job_id)
    got_lock = False
    try:
        _update_job(job_id, status="running")

        input_image = Image.open(input_path).convert("RGB")
        # ComfyUIワークフローの ImageScaleToTotalPixels(~1MP)相当の前処理
        # (charsheetと同一実装を流用)
        processed = generate_mod.preprocess_image(input_image)
        processed.save(os.path.join(job_dir, "input.png"))

        got_lock = generate_mod.acquire_generation_lock(blocking=True)
        gpu.empty_cache()
        generate_mod.ensure_edit_loaded()

        total = len(angles)
        for idx, angle in enumerate(angles):
            key = angle["key"]
            _update_view(job_id, key, status="running")
            try:
                meta = generate_mod.generate_view(
                    processed,
                    prompt=angle["prompt"],
                    seed=seed,
                    negative_prompt=NEGATIVE_PROMPT,
                    progress_extra={
                        "job_id": job_id,
                        "direction": key,
                        "direction_label": angle.get("label_ja"),
                        "direction_index": idx + 1,
                        "direction_total": total,
                        "app": "scene_angles",
                    },
                )
                out_path = os.path.join(job_dir, f"{key}.png")
                _copy_generated_image(meta, out_path)
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
                url=f"/api/scene_angles/jobs/{job_id}/images/{key}.png",
            )
            _update_job(job_id, progress=progress)

        _update_job(job_id, status="done", load_info=generate_mod.get_load_info())
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
    seed: int = Form(0),
    angles: str = Form(""),
):
    """シーンアングル生成ジョブを開始する。

    - image: 入力シーン画像(必須、multipart)
    - seed: 乱数シード(任意、既定0=角度ごとにランダム)
    - angles: カンマ区切りのアングルID(任意。省略=8種全部)。
      有効ID: close_up / wide_angle / right_90 / right_45 / aerial /
      low_angle / left_90 / left_45
    """
    global current_job_id

    selected = _parse_angles(angles)

    with current_job_lock:
        if current_job_id is not None:
            raise HTTPException(
                status_code=409,
                detail="別のシーンアングルジョブが実行中です。しばらく待ってから再試行してください。",
            )
        job_id = uuid.uuid4().hex[:12]
        current_job_id = job_id

    job_dir = _job_dir(job_id)
    input_path = os.path.join(job_dir, "upload_raw.png")

    try:
        contents = await image.read()
        # EXIF回転を実ピクセルへ適用してから保存(charsheetと同じ理由)
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(contents)))
        img.save(input_path)
        with Image.open(input_path) as im:
            im.verify()
    except Exception as exc:
        with current_job_lock:
            current_job_id = None
        raise HTTPException(status_code=400, detail=f"画像の読み込みに失敗しました: {exc}")

    _init_job(job_id, seed, selected)

    thread = threading.Thread(
        target=_run_job, args=(job_id, input_path, seed, selected), daemon=True
    )
    thread.start()

    return {"job_id": job_id}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="ジョブが見つかりません")
    with jobs_lock:
        return JSONResponse(dict(jobs[job_id]))


@router.get("/jobs/{job_id}/images/{key}.png")
async def get_angle_image(job_id: str, key: str):
    # job_id / key とも許可リスト・文字種で検証(パストラバーサル対策)
    if not job_id or "/" in job_id or "\\" in job_id or ".." in job_id:
        raise HTTPException(status_code=404, detail="不正な job_id です")
    if key not in ANGLE_BY_KEY:
        raise HTTPException(status_code=404, detail="不正なアングルIDです")
    path = os.path.join(OUTPUTS_DIR, job_id, f"{key}.png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="画像が見つかりません")
    return FileResponse(path, media_type="image/png")


@router.get("/jobs/{job_id}/input.png")
async def get_input_image(job_id: str):
    if not job_id or "/" in job_id or "\\" in job_id or ".." in job_id:
        raise HTTPException(status_code=404, detail="不正な job_id です")
    path = os.path.join(OUTPUTS_DIR, job_id, "input.png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="画像が見つかりません")
    return FileResponse(path, media_type="image/png")


@router.get("/angles")
async def list_angles():
    """利用可能なアングルIDの一覧(UI用)。GPU不使用。"""
    return {
        "angles": [
            {"key": a["key"], "label_ja": a["label_ja"], "label_en": a["label_en"]}
            for a in ANGLES
        ]
    }
