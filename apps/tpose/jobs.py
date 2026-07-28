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
    CLAWS_MODES,
    NEGATIVE_PROMPT,
    PALMS_MODES,
    SUBJECT_MODES,
    TAIL_PRESETS,
    VIEW_BY_KEY,
    VIEWS,
    build_edit_prompt,
    build_prompt,
    build_recolor_prompt,
    resolve_subject,
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
            # 2048アップスケール版(POST /jobs/{id}/upscale で作る)
            "up_url": None,
            "up_download_url": None,
            "up_nobg_url": None,
            "up_nobg_download_url": None,
            "up_size": None,
            "has_prev": False,
            # rev: この画像が書き換わった回数。UIは rev が変わったときだけ画像を
            # 再読み込みする(毎回キャッシュバスターを付けるとポーリングのたびに
            # 再取得されて表示がフラッシュするため。ユーザー報告 2026-07-28)
            "rev": 0,
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


def _bump_view_rev(job_id: str, key: str) -> None:
    """ビュー画像が書き換わったことを UI へ知らせる(rev をインクリメント)。"""
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        for v in job["views"]:
            if v["key"] == key:
                v["rev"] = int(v.get("rev", 0)) + 1
                break


def _copy_generated_image(meta: dict, dest_path: str) -> None:
    image_url = meta.get("image_url", "")
    name = image_url.rsplit("/", 1)[-1]
    src_path = os.path.join(BASE_DIR, "outputs", name)
    shutil.copy2(src_path, dest_path)


def _build_zip(job_dir: str, keys: list) -> Optional[str]:
    """生成済みビューをZIPにまとめる(一括ダウンロード用)。

    背景透過版(`<key>_nobg.png`)とアップスケール版(`<key>_2048*.png`)があれば
    併せて含める。
    """
    paths = [(k, os.path.join(job_dir, f"{k}.png")) for k in keys]
    paths = [(k, p) for k, p in paths if os.path.exists(p)]
    if not paths:
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for key, path in paths:
            zf.write(path, arcname=f"{key}.png")
            for suffix in ("_nobg", "_2048", "_2048_nobg"):
                extra = os.path.join(job_dir, f"{key}{suffix}.png")
                if os.path.exists(extra):
                    zf.write(extra, arcname=f"{key}{suffix}.png")
        input_path = os.path.join(job_dir, "input.png")
        if os.path.exists(input_path):
            zf.write(input_path, arcname="input.png")
    zip_path = os.path.join(job_dir, "download.zip")
    with open(zip_path, "wb") as f:
        f.write(buf.getvalue())
    return zip_path


# 背景が「生成された白一色」であることを利用した切り抜きの閾値。
_CUTOUT_BRIGHT_T = 250   # これ以上明るい画素を「白」とみなす(背景候補)
# rembg のアルファを二値化する閾値。既定の 128 では**淡い色の部位で rembg が
# 「薄いゴースト」(alpha 1〜128)しか返さず手が消える**(ユーザー報告)。
# 実測(背面ビュー、閾値ごとの「淡い毛の取りこぼし / 背景の巻き込み」):
#   128 -> 9031px / 625px   64 -> **474px / 2353px**   32 -> 220px / 8131px
#     8 ->   67px / 40161px
# 64 が最良点(取りこぼしを95%削減、巻き込みは輪郭の約1pxの縁に留まる。純白の縁なので
# 透過画像として実害が小さい)。
_CUTOUT_REMBG_T = 64
_CUTOUT_RECOVER_T = 245  # rembg が落とした画素のうち、これ以上明るければ「淡い毛」として回収
_CUTOUT_FEATHER = 0.8    # 確定マスクの縁をぼかす量(px 相当。1px程度の自然な縁にする)


def _cutout_rgba(src_rgb: Image.Image, rembg_rgba: Image.Image) -> Image.Image:
    """白背景の生成画像から背景透過画像(RGBA)を組み立てる。

    実機で出た3つの不具合への対処:
      1. **アルファに内部の穴が空く**(オーバーオールの帯の間が半透明 alpha≈160 になり
         暗い色が透けた)-> `binary_fill_holes` で埋める。
      2. **RGBが濁る**(rembg 出力のRGBは穴の周辺で暗くなる)-> RGBは**白背景版から
         取り直し**、アルファだけを組み立てる。
      3. **淡い色の部位(白い毛の手)が消える**(ユーザー報告。背面ビューで実測
         40,072px = 被写体の約10%が落ちていた)-> 下記の「白の塗り分け」+
         `_CUTOUT_REMBG_T`(二値化閾値を 128 -> 64 へ下げる)で回収する。

    3. の方針: 背景は**生成された白一色**なので、ニューラル・マット(rembg)の
    ソフトアルファに頼る必要がない。
      - 画像の**境界から連結した明るい領域**(`lum >= _CUTOUT_BRIGHT_T`)だけを背景とする
        (被写体に囲まれた明るい画素 = 白飛びした毛は背景にならない)。
      - rembg が落とした画素のうち **明るいもの(`lum >= _CUTOUT_RECOVER_T`)だけ**を
        被写体として回収する。影は輝度が低い(実測: 影 ≈148 / 淡い毛 ≈253)ので
        この閾値で分離でき、接地影を巻き込まない。
      - 最終マスクは最大連結成分 + 穴埋めで確定させ、`_CUTOUT_FEATHER` でぼかして
        1px程度の自然な縁にする。**rembgの半透明マットは使わない**(淡い毛の内部が
        半透明のまま残るのが症状の本質だったため)。毛の房のような細部は
        「背景より暗い」ため上記の塗り分けで残ることを実機で確認済み(しっぽで比較)。
    scipy が import できない環境では従来どおり rembg のアルファをそのまま使う。
    """
    import numpy as np

    alpha = np.array(rembg_rgba.getchannel("A")).astype(int)
    try:
        from scipy import ndimage
    except ImportError:  # pragma: no cover - scipy が無い環境向けのフォールバック
        print("[apps.tpose] warning: scipy が無いため背景除去の後処理をスキップします")
        out = src_rgb.convert("RGBA")
        out.putalpha(Image.fromarray(np.where(alpha >= 250, 255, alpha).astype(np.uint8), "L"))
        return out

    lum = np.asarray(src_rgb.convert("RGB"), dtype=float).sum(2) / 3.0
    # 境界から連結した「白」= 背景
    labels, count = ndimage.label(lum >= _CUTOUT_BRIGHT_T)
    if count:
        border_ids = set(np.unique(np.concatenate(
            [labels[0], labels[-1], labels[:, 0], labels[:, -1]]))) - {0}
        background = np.isin(labels, list(border_ids))
    else:
        background = np.zeros_like(lum, dtype=bool)

    rembg_subject = alpha > _CUTOUT_REMBG_T
    recovered = (~background) & (~rembg_subject) & (lum >= _CUTOUT_RECOVER_T)
    mask = rembg_subject | recovered

    # 最大連結成分だけを残す(切り抜き残りの小さなゴミを落とす)
    comp, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(comp), comp, range(1, n + 1))
        mask = comp == (int(np.argmax(sizes)) + 1)
    mask = ndimage.binary_fill_holes(mask)

    out_alpha = ndimage.gaussian_filter(mask.astype(np.float32) * 255.0, _CUTOUT_FEATHER)
    out_alpha = np.where(out_alpha >= 250, 255, out_alpha).astype(np.uint8)
    out = src_rgb.convert("RGBA")
    out.putalpha(Image.fromarray(out_alpha, mode="L"))
    return out


def _remove_bg_pass(job_id: str, job_dir: str, keys: list, method: str = "rembg") -> None:
    """生成済み各ビューの背景を除去し `<key>_nobg.png`(RGBA)として保存する。

    白背景版(`<key>.png`)は残す(3Dツール側が白背景を前提にする場合や、
    切り抜き失敗時の退避のため)。rembg はCPU(ONNX)処理なので **GPUロックを
    解放した後に**呼ぶこと(呼び出し側 `_run_job` がそうしている)。
    1枚失敗しても他のビューの処理は続ける(部分的な成功を許容する)。
    """
    from core.bg import remove_background, resolve_method

    method = resolve_method(method)

    for key in keys:
        src = os.path.join(job_dir, f"{key}.png")
        if not os.path.exists(src):
            continue
        try:
            with Image.open(src) as im:
                src_rgb = im.convert("RGB")
                rgba = _cutout_rgba(src_rgb, remove_background(src_rgb, method=method))
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

        # 毛色の自動推定(rembg はCPU処理なのでGPUロック取得前に行う)。
        # 用途は2つ: (1) 爪抑制文("each toe is <毛色> fur right to the tip")、
        # (2) 背面ビューの後頭部が黒髪になる問題の対策("the back of the head is
        # covered in <毛色> fur" / 中立では "... the same <毛色> color as the front")。
        #
        # **自動推定は animal のときだけ使う**(2026-07-28修正、ユーザー報告
        # 「何も指定しないと髪型は維持されるが後ろだけ白くなる」):
        # sample_fur_color() は被写体全体の**低彩度画素**の中央値を毛色とみなす。
        # ぬいぐるみ(全身が毛)では妥当だが、**服や肌が低彩度な人物キャラでは誤推定**する。
        # 実測: 茶髪+白Tシャツのアニメキャラ -> "cream white" と推定 -> 背面プロンプトが
        # 「後頭部は cream white」になり、**後頭部だけ白くなる**(サーバが自分で
        # 白くしていた)。頭部だけを測る案も検証したが、帽子・アクセサリを拾って
        # 不安定だった(momo は上10%が帽子の赤、上記キャラは上15%以上で肌が混ざり pink)
        # ため、**推定は毛で覆われた被写体(animal)に限定**する。
        # 中立/人物で後頭部の色を固定したい場合は fur_color を明示指定する
        # (明示値は従来どおり中立でも使われる)。
        fur_color = (params.get("fur_color") or "").strip()
        if not fur_color and resolve_subject(
                params.get("subject", "auto"),
                params.get("paw_pads", "auto")) == "animal":
            fur_color = generate_mod.sample_fur_color(processed)
            _update_job(job_id, fur_color_detected=fur_color)

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
                claws=params.get("claws", "none"),
                fur_color=fur_color,
                tail=params["tail"],
                body=params["body"],
                costume=params.get("costume", ""),
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
                _bump_view_rev(job_id, key)

                # 色調整の2パス目(任意)。1パス目のプロンプト末尾を汚さずに色を
                # 変えられる(apps/tpose/prompts.py の recolor コメント参照)。
                # 正面は**調整後**の画像を後続ビューの参照に使う(色をビュー間で揃える)。
                recolor = (params.get("recolor") or "").strip()
                if recolor:
                    _update_view(job_id, key, status="recoloring")
                    with Image.open(out_path) as generated:
                        recolor_meta = generate_mod.generate_view(
                            [generated.convert("RGB")],
                            prompt=build_recolor_prompt(recolor),
                            seed=seed,
                            negative_prompt=NEGATIVE_PROMPT,
                            progress_extra={
                                "job_id": job_id,
                                "direction": f"{key}_recolor",
                                "direction_label": view.get("label_ja"),
                                "direction_index": idx + 1,
                                "direction_total": total,
                                "app": "tpose",
                            },
                        )
                    _copy_generated_image(recolor_meta, out_path)
                    _bump_view_rev(job_id, key)

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
            _remove_bg_pass(job_id, job_dir, keys, params.get("bg_method", "anime"))

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
    claws: str = Form("none"),
    fur_color: str = Form(""),
    tail: str = Form(""),
    body: str = Form(""),
    costume: str = Form(""),
    extra_prompt: str = Form(""),
    recolor: str = Form(""),
    remove_bg: bool = Form(False),
    bg_method: str = Form("anime"),
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
    - claws: 爪。"none"(既定、爪なし=ぬいぐるみでは自然)/ "auto"(参照画像に任せる)/
      自由記述(例 "short white claws")。`subject` が animal に解決されるときのみ有効
    - fur_color: 毛色の色名(任意、例 "cream white")。空なら `claws="none"` かつ動物型の
      ときに入力画像から自動推定する(爪を消しても後足の指の分離が保たれるようにするため。
      apps/tpose/prompts.py の「爪」コメント参照)
    - tail: しっぽ形状の自由記述(例 "a long fluffy tail with a black tip")。
      "none" でしっぽなし、空/"auto" で指定なし(未指定だとビューごとに形状が
      揺れるため、入力画像にしっぽが写っていない場合は指定推奨)
    - costume: **背面から見た衣装**の自由記述(任意、背面ビューにのみ効く)。
      前開きのベスト・カーディガン等は、既定だと**背中側にも前開きが描かれ**
      インナーが見えてしまう(apps/tpose/prompts.py の該当コメント参照)。例:
      "the short cream lace bolero ends at the waist and its back is one continuous
      piece of lace, the white pencil skirt below it is unchanged"。
      **丈・範囲まで書くこと**(書かないと衣装が伸びてワンピース化した実測あり)
    - extra_prompt: プロンプト末尾へ追記する自由記述(任意)
    - recolor: 生成後に色を調整する2パス目の指示(任意、空なら実行しない)。例
      "Make the fur a warmer cream tone with richer shading"。全ビューへ同じ指示を
      適用し、正面は調整後の画像を後続ビューの参照に使う(色をビュー間で揃えるため)。
      1ビューあたり生成が2回になる(所要時間はおよそ2倍)。**強い指示は同一性も動かす**
      (実測で毛色が黄褐色へ寄り爪が戻った例あり、apps/tpose/prompts.py参照)
    - bg_method: 背景除去の方式(既定 `anime`=アニメ・キャラクター向け /
      `rembg`=汎用。core/bg.py 参照)。Tポーズの被写体はキャラクターなので既定を
      `anime` にしている(淡い色の毛や髪の取りこぼしが少ない: 実測 27,232px → 13,733px)
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
        "bg_method": bg_method,
        "recolor": recolor,
        "subject": subject,
        "palms": palms,
        "paw_pads": paw_pads,
        "claws": claws,
        "fur_color": fur_color,
        "tail": tail,
        "body": body,
        "costume": costume,
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

    `<view>` は白背景版、`<view>_nobg` は背景透過版(remove_bg=true時のみ存在)、
    `<view>_2048` / `<view>_2048_nobg` はアップスケール版(upscale 実行時のみ存在)。
    """
    base = key
    for suffix in ("_2048_nobg", "_2048", "_nobg"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
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


# ============================================================================
# 生成済みビューへの追加編集(何度でも適用できる汎用 Edit)
#
# 生成時の `recolor` パラメータと同じ2パス目の仕組みを、**生成後に任意回数**
# 呼べるエンドポイントとして切り出したもの(色調整に限らず「帽子を外す」等の
# 汎用修正に使える)。charsheet の refine/undo と同じ流儀で、直前の画像を
# `<key>_prev.png` に1世代だけ退避して undo できるようにする。
# ============================================================================
def _parse_view_keys(raw: str, job: dict, require_done: bool = True) -> list:
    """views パラメータを「対象ビューのキー一覧」へ解決する(空なら完了済み全部)。"""
    valid = {v["key"] for v in job["views"]}
    if raw and raw.strip():
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        unknown = set(keys) - valid
        if unknown:
            raise HTTPException(
                status_code=400, detail=f"このジョブに無いビューIDです: {sorted(unknown)}")
    else:
        keys = [v["key"] for v in job["views"]]
    if require_done:
        done = {v["key"] for v in job["views"] if v["status"] in ("done", "edited")}
        keys = [k for k in keys if k in done]
    if not keys:
        raise HTTPException(status_code=409, detail="対象になる生成済みビューがありません。")
    return keys


def _refresh_has_prev(job_id: str, job_dir: str) -> None:
    with jobs_lock:
        for v in jobs[job_id]["views"]:
            v["has_prev"] = os.path.exists(os.path.join(job_dir, f"{v['key']}_prev.png"))


def _run_edit(job_id: str, keys: list, instruction: str, seed: int, keep_pose: bool,
              use_reference: bool = False):
    """選択ビューへ順に Edit をかけ、透過版・ZIPを作り直す。

    use_reference(**既定 False**): 元画像(input.png)を2枚目の参照として渡す
    (ユーザー提案 2026-07-28「生成後のEDITをレファレンス付きにすれば、オリジナルを
    参照して修正ができるのでは」)。生成の2段目以降が [生成した正面, 元画像] の
    2枚参照で連鎖しているのと同じ手で、Edit も QwenImageEditPlusPipeline なので
    参照は最大3枚まで渡せる(MAX_EDIT_IMAGES)。プロンプト側で
    「1枚目=編集対象、2枚目=同じキャラの元画像」と明示する
    (build_edit_prompt(with_reference=True))。

    **既定を False にした理由(実機A/B)**: 参照ありは元の見た目をよく思い出す
    (指示 "restore the original hairstyle" で元のまとめ髪・前髪に近づいた)一方、
    **元画像のポーズ・背景まで引き戻す事故**が起きる。被写体bboxの幅/高さ比
    (Tポーズは腕を広げるので大きい)で測ると
      編集前 0.90 / 参照なし 0.89 / 参照あり 0.87・0.87・**0.27**
    となり、3seed中1つで**Tポーズが壊れて元写真の自然な立ち姿に戻った**。
    別のseedでは背景が市松模様になった。衣装の修正でも、参照ありは
    「背中中央のインナー露出 1.3%(参照なし 6.1%)」と数値上は良いのに、
    **目視ではボレロが膝丈のワンピースへ伸びていた**(参照なしは腰丈で正解)。
    そのため「元の見た目を思い出させたいときだけ明示的にONにする」運用にする。
    """
    global current_job_id
    job_dir = _job_dir(job_id)
    got_lock = False
    reference = None
    if use_reference:
        ref_path = os.path.join(job_dir, "input.png")
        if os.path.exists(ref_path):
            with Image.open(ref_path) as ref:
                reference = ref.convert("RGB")
    prompt = build_edit_prompt(instruction, keep_pose=keep_pose,
                               with_reference=reference is not None)
    try:
        _update_job(job_id, status="editing", edit_error=None)
        got_lock = generate_mod.acquire_generation_lock(blocking=True)
        gpu.empty_cache()
        generate_mod.ensure_edit_loaded()

        for idx, key in enumerate(keys):
            img_path = os.path.join(job_dir, f"{key}.png")
            if not os.path.exists(img_path):
                continue
            _update_view(job_id, key, status="editing", edit_prompt=prompt)
            shutil.copy2(img_path, os.path.join(job_dir, f"{key}_prev.png"))
            try:
                with Image.open(img_path) as current:
                    refs = [current.convert("RGB")]
                    if reference is not None:
                        refs.append(reference)
                    meta = generate_mod.generate_view(
                        refs,
                        prompt=prompt,
                        seed=seed,
                        negative_prompt=NEGATIVE_PROMPT,
                        progress_extra={
                            "job_id": job_id,
                            "direction": f"{key}_edit",
                            "direction_index": idx + 1,
                            "direction_total": len(keys),
                            "app": "tpose",
                        },
                    )
                _copy_generated_image(meta, img_path)
                _bump_view_rev(job_id, key)
                _clear_upscaled(job_id, job_dir, key)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                _update_view(job_id, key, status="done")
                _update_job(job_id, status="done", edit_error=f"{key}: {exc}")
                return
            _update_view(job_id, key, status="done")

        # 透過版を持っていたビューは作り直す(古い切り抜きが残らないように)
        nobg_keys = [k for k in keys
                     if os.path.exists(os.path.join(job_dir, f"{k}_nobg.png"))]
        if nobg_keys:
            if got_lock:
                generate_mod.release_generation_lock()
                got_lock = False
            _update_job(job_id, status="removing_bg")
            _remove_bg_pass(job_id, job_dir, nobg_keys,
                            (jobs[job_id].get("params") or {}).get("bg_method", "anime"))

        _build_zip(job_dir, [v["key"] for v in jobs[job_id]["views"]])
        _refresh_has_prev(job_id, job_dir)
        _update_job(job_id, status="done")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        _update_job(job_id, status="done", edit_error=str(exc))
    finally:
        progress_mod.finish()
        if got_lock:
            generate_mod.release_generation_lock()
        with current_job_lock:
            current_job_id = None


@router.post("/jobs/{job_id}/edit")
async def edit_views(
    job_id: str,
    prompt: str = Form(...),
    views: str = Form(""),
    seed: int = Form(0),
    keep_pose: bool = Form(True),
    use_reference: bool = Form(False),
):
    """生成済みビューへ追加の Edit をかける(何度でも呼べる)。

    - prompt: 修正指示(例 "Make the fur a warmer cream tone with richer shading"、
      "remove the hat")
    - views: カンマ区切りのビューID(省略=完了済み全ビュー)
    - seed: 乱数シード(0=ランダム)
    - keep_pose: true(既定)なら「ポーズ・画角・背景と、それ以外の細部は変えない」を
      前置きする(指示文は末尾に置く)
    - use_reference: **既定 false**。true にすると元画像を2枚目の参照として渡し、
      「元の髪型・衣装へ戻す」系の修正で元の見た目を参照できる。ただし
      **元画像のポーズ・背景まで引き戻す事故がある**(実機A/Bで3seed中1つが
      Tポーズを失い自然な立ち姿に戻った。_run_edit のコメント参照)ので、
      必要なときだけONにすること
    直前の画像は `<key>_prev.png` に1世代退避され、`POST /jobs/{job_id}/undo` で戻せる。
    透過版を持つビューは Edit 後に作り直す。
    """
    global current_job_id
    _check_job_id(job_id)
    if not prompt or not prompt.strip():
        raise HTTPException(status_code=400, detail="修正指示を入力してください。")
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="ジョブが見つかりません")
        job = dict(job)
    keys = _parse_view_keys(views, job)

    with current_job_lock:
        if current_job_id is not None:
            raise HTTPException(
                status_code=409,
                detail="別のTポーズ生成/編集が実行中です。しばらく待ってから再試行してください。",
            )
        current_job_id = job_id

    thread = threading.Thread(
        target=_run_edit,
        args=(job_id, keys, prompt, seed, bool(keep_pose), bool(use_reference)),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "views": keys, "status": "editing"}


# ============================================================================
# アップスケール(2048)
#
# 3D化・リグへ渡す前に解像度を上げるための後処理(ユーザー要望 2026-07-28)。
# **Real-ESRGAN x2(GAN系、決定論的)を使い、拡散モデルでの再生成はしない**:
# 再生成だと解像度は上がっても細部が書き換わる(このアプリが繰り返し戦ってきた
# 髪型・衣装のドリフトがそのまま起きる)。詳細は core/upscale.py の冒頭コメント。
# 白背景版を拡大し、透過版を持つビューは**拡大後に切り抜きを作り直す**
# (2048で切り抜いた方が輪郭が精細になる)。
# ============================================================================
def _clear_upscaled(job_id: str, job_dir: str, key: str) -> None:
    """元画像が書き換わったら、古いアップスケール版は削除する(取り違え防止)。"""
    removed = False
    for suffix in ("_2048", "_2048_nobg"):
        path = os.path.join(job_dir, f"{key}{suffix}.png")
        if os.path.exists(path):
            os.remove(path)
            removed = True
    if removed:
        _update_view(job_id, key, up_url=None, up_download_url=None,
                     up_nobg_url=None, up_nobg_download_url=None, up_size=None)


def _run_upscale(job_id: str, keys: list, target: int):
    global current_job_id
    job_dir = _job_dir(job_id)
    got_lock = False
    try:
        _update_job(job_id, status="upscaling", upscale_error=None)
        got_lock = generate_mod.acquire_generation_lock(blocking=True)
        gpu.empty_cache()

        from core.upscale import upscale_image

        done_keys = []
        for key in keys:
            src = os.path.join(job_dir, f"{key}.png")
            if not os.path.exists(src):
                continue
            _update_view(job_id, key, status="upscaling")
            try:
                with Image.open(src) as im:
                    out = upscale_image(im.convert("RGB"), target=target)
                out.save(os.path.join(job_dir, f"{key}_2048.png"))
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                _update_view(job_id, key, status="done")
                _update_job(job_id, status="done", upscale_error=f"{key}: {exc}")
                return
            _update_view(
                job_id, key, status="done", up_size=f"{out.width}x{out.height}",
                up_url=f"/api/tpose/jobs/{job_id}/images/{key}_2048.png",
                up_download_url=f"/api/tpose/jobs/{job_id}/download/{key}_2048.png",
            )
            done_keys.append(key)

        # 透過版を持つビューは 2048 で切り抜きを作り直す(CPU処理なのでロック解放後)。
        nobg_keys = [k for k in done_keys
                     if os.path.exists(os.path.join(job_dir, f"{k}_nobg.png"))]
        if nobg_keys:
            if got_lock:
                generate_mod.release_generation_lock()
                got_lock = False
            _update_job(job_id, status="removing_bg")
            method = (jobs[job_id].get("params") or {}).get("bg_method", "anime")
            from core.bg import remove_background, resolve_method

            method = resolve_method(method)
            for key in nobg_keys:
                path = os.path.join(job_dir, f"{key}_2048.png")
                try:
                    with Image.open(path) as im:
                        rgb = im.convert("RGB")
                        rgba = _cutout_rgba(rgb, remove_background(rgb, method=method))
                    rgba.save(os.path.join(job_dir, f"{key}_2048_nobg.png"))
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    _update_view(job_id, key, nobg_error=str(exc))
                    continue
                _update_view(
                    job_id, key,
                    up_nobg_url=f"/api/tpose/jobs/{job_id}/images/{key}_2048_nobg.png",
                    up_nobg_download_url=f"/api/tpose/jobs/{job_id}/download/{key}_2048_nobg.png",
                )

        _build_zip(job_dir, [v["key"] for v in jobs[job_id]["views"]])
        _update_job(job_id, status="done")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        _update_job(job_id, status="done", upscale_error=str(exc))
    finally:
        progress_mod.finish()
        if got_lock:
            generate_mod.release_generation_lock()
        with current_job_lock:
            current_job_id = None


@router.post("/jobs/{job_id}/upscale")
async def upscale_views(job_id: str, views: str = Form(""), target: int = Form(2048)):
    """生成済みビューを `target` px(既定2048)へアップスケールする。

    - views: カンマ区切りのビューID(省略=完了済み全ビュー)
    - target: 出力の長辺(既定 2048)。1024〜4096

    Real-ESRGAN x2(`core/upscale.py`)で拡大するので**内容は書き換わらない**
    (拡散モデルでの再生成ではない)。元の1024版はそのまま残り、
    `<key>_2048.png` が追加される。透過版を持つビューは 2048 で切り抜きを作り直す。
    元画像を編集(`/edit`・`/undo`)すると、古いアップスケール版は自動的に破棄される。
    """
    global current_job_id
    _check_job_id(job_id)
    if target < 1024 or target > 4096:
        raise HTTPException(status_code=400, detail="target は 1024〜4096 で指定してください。")
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="ジョブが見つかりません")
        job = dict(job)
    keys = _parse_view_keys(views, job)

    with current_job_lock:
        if current_job_id is not None:
            raise HTTPException(
                status_code=409,
                detail="別のTポーズ生成/編集が実行中です。しばらく待ってから再試行してください。",
            )
        current_job_id = job_id

    thread = threading.Thread(
        target=_run_upscale, args=(job_id, keys, int(target)), daemon=True
    )
    thread.start()
    return {"job_id": job_id, "views": keys, "status": "upscaling", "target": target}


@router.post("/jobs/{job_id}/undo")
async def undo_views(job_id: str, views: str = Form("")):
    """直前の Edit を取り消す(`<key>_prev.png` から復元。1世代のみ)。"""
    _check_job_id(job_id)
    job_dir = _job_dir(job_id)
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="ジョブが見つかりません")
        job = dict(job)
    keys = _parse_view_keys(views, job, require_done=False)

    restored = []
    for key in keys:
        prev = os.path.join(job_dir, f"{key}_prev.png")
        if not os.path.exists(prev):
            continue
        shutil.copy2(prev, os.path.join(job_dir, f"{key}.png"))
        os.remove(prev)
        _bump_view_rev(job_id, key)
        _clear_upscaled(job_id, job_dir, key)
        restored.append(key)
    if not restored:
        raise HTTPException(status_code=409, detail="取り消せる編集がありません。")

    nobg_keys = [k for k in restored
                 if os.path.exists(os.path.join(job_dir, f"{k}_nobg.png"))]
    if nobg_keys:
        _remove_bg_pass(job_id, job_dir, nobg_keys,
                        (job.get("params") or {}).get("bg_method", "anime"))
    _build_zip(job_dir, [v["key"] for v in job["views"]])
    _refresh_has_prev(job_id, job_dir)
    return {"job_id": job_id, "restored": restored}


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
        "claws_modes": list(CLAWS_MODES),
    }
