#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diffusers-server 統合ベンチマーク(Phase 4)。

flux2_diffusers/benchmark.py(サブプロセス分離・attn/compileマトリクス方式)をベースに、
全ファミリー(Qwen-Image 系 T2I/Edit/ControlNet/Layered、FLUX.2 T2I/I2I)に対応させたもの。

旧 flux2_diffusers/benchmark.py との違い:
  - サーバ経由(HTTP)ではなく families 直接呼び出しで計測する点は同じだが、旧実装は
    flux2 の pipeline_manager.load_pipeline() を直接呼んでいた。本スクリプトは
    core.registry.registry.load(family, mode) 経由でロードする(REBUILD_PLAN の
    「registry.load() 経由でないと自動 unload が効かない」知見。CLAUDE.md にも記載)。
  - 対象がQwen系複数モード + FLUX.2 T2I/I2Iに拡張されている。
  - 結果は screen/matrix 用の一時 JSON ではなく、単一の benchmark_results.json に
    「実行するたびに追記」する(環境情報・構成・cold/warm・ピークVRAMを1レコードとして
    末尾に追加)。

使い方:
    # 単一構成を1回計測(cold=プロセス起動後の初回、warm=2回目)
    venv/bin/python benchmark.py --config qwen_t2i
    venv/bin/python benchmark.py --config flux2_t2i
    venv/bin/python benchmark.py --config qwen_edit
    venv/bin/python benchmark.py --config qwen_controlnet
    venv/bin/python benchmark.py --config qwen_layered
    venv/bin/python benchmark.py --config flux2_i2i

    # 複数構成をまとめて(各構成は同一プロセス内で順に実行。ファミリー切替を挟むと
    # registry の自動 unload が働く。VRAM リーク確認も兼ねる)
    venv/bin/python benchmark.py --config qwen_t2i,flux2_t2i,qwen_t2i

    # FLUX.2 の offload 比較(model_cpu_offload と none を両方計測、README §5 用)
    venv/bin/python benchmark.py --config flux2_t2i --flux2-offload none
    venv/bin/python benchmark.py --config flux2_t2i --flux2-offload model_cpu_offload

環境変数 DS_BENCH_CONFIG でも --config 相当を指定できる(CI 等での固定実行用)。

出力: benchmark_results.json(スクリプトと同じディレクトリ)に1構成1レコードを追記。
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

RESULTS_PATH = os.path.join(HERE, "benchmark_results.json")

# ============================================================================
# 構成定義
# ============================================================================
# 各構成は (family_name, internal_mode, build_request(fn), warmup_note) を持つ。
# request 関数は毎回新しい request dict を返す(画像はここでロード/生成する)。

DEFAULT_PROMPT_T2I = (
    "A photorealistic portrait of a red fox wearing a small wizard hat, "
    "standing in a misty autumn forest, cinematic lighting, high detail"
)
DEFAULT_PROMPT_EDIT = (
    "Show the character in a full body front view, facing directly toward the camera, "
    "neutral standing pose, full figure visible from head to toe."
)
DEFAULT_PROMPT_LAYERED = ""
DEFAULT_PROMPT_CONTROLNET = (
    "A photorealistic portrait of a red fox wearing a small wizard hat, "
    "standing in a misty autumn forest, cinematic lighting, high detail"
)


def _make_solid_image(size=(640, 640), color=(120, 140, 160)):
    from PIL import Image
    return Image.new("RGB", size, color)


def _make_canny_control_image(size=(624, 944)):
    """ControlNet canny 用の制御画像を生成する(参照画像が無い場合の合成フォールバック)。

    実際の評価では静的な参照画像(例: outputs/ 内の既存生成物)を使う方が意味のある
    結果になるが、ベンチマーク単体実行でも動くよう、簡単な幾何学図形から canny を作る。
    """
    import numpy as np
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, (30, 30, 30))
    draw = ImageDraw.Draw(img)
    w, h = size
    draw.ellipse([w * 0.2, h * 0.2, w * 0.8, h * 0.8], outline=(255, 255, 255), width=8)
    draw.rectangle([w * 0.35, h * 0.35, w * 0.65, h * 0.65], outline=(255, 255, 255), width=6)
    return img


def _canny_from_image(image):
    import cv2
    import numpy as np
    from PIL import Image

    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)


def _build_qwen_t2i_request(args):
    # 解像度は1024x1024(旧実測値との比較用。REBUILD_PLANの実測値
    # 「warm 2.65s / ピーク25.85GB」は旧構成(Qwen-Image-2512 bnb4bit+Lightning、廃止済み)、
    # 1024x1024・同一seed(42)での計測。現行既定は 2512 fp8-lightning fuse。
    # 既定の1328x1328(app.pyのT2IRequest既定値)ではないので注意)。
    return {
        "prompt": DEFAULT_PROMPT_T2I,
        "negative_prompt": "",
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "cfg": 1.0,
        "seed": 42,
        "lightning": True,
        "shift": None,
    }


def _build_qwen_edit_request(args):
    ref = _make_solid_image((640, 640))
    return {
        "prompt": DEFAULT_PROMPT_EDIT,
        "negative_prompt": "",
        "steps": 4,
        "cfg": 1.0,
        "seed": 42,
        "lightning": True,
        "shift": None,
        "width": args.edit_size,
        "height": args.edit_size,
        "_images": [ref],
    }


def _build_qwen_controlnet_request(args):
    ctrl_img = _make_canny_control_image((624, 944))
    ctrl_img = _canny_from_image(ctrl_img)
    return {
        "prompt": DEFAULT_PROMPT_CONTROLNET,
        "negative_prompt": " ",
        "control_type": "canny",
        "controlnet_conditioning_scale": 1.0,
        "width": 624,
        "height": 944,
        "steps": 4,
        "cfg": 1.0,
        "seed": 42,
        "_control_image": ctrl_img,
    }


def _build_qwen_layered_request(args):
    from PIL import Image
    src = _make_solid_image((640, 640)).convert("RGBA")
    return {
        "prompt": DEFAULT_PROMPT_LAYERED,
        "negative_prompt": "",
        "layers": 4,
        "resolution": 640,
        "steps": None,
        "cfg": None,
        "shift": None,
        "seed": 42,
        "cfg_normalize": False,
        "use_en_prompt": True,
        "_image": src,
    }


def _build_flux2_t2i_request(args):
    return {
        "prompt": DEFAULT_PROMPT_T2I,
        "steps": 28,
        "guidance_scale": 4.0,
        "width": 1024,
        "height": 1024,
        "seed": 42,
    }


def _build_flux2_i2i_request(args):
    ref = _make_solid_image((1024, 1024), (80, 120, 90))
    return {
        "prompt": "Place the fox from the reference image sitting in the same forest setting, photorealistic",
        "steps": 28,
        "guidance_scale": 4.0,
        "width": 1024,
        "height": 1024,
        "seed": 42,
        "_images": [ref],
    }


CONFIGS = {
    "qwen_t2i": {
        "family": "qwen_image",
        "mode": "t2i",
        "build_request": _build_qwen_t2i_request,
        "run_fn": "run_t2i",
        "desc": "Qwen T2I 2512 fp8-lightning 1024x1024 4steps",
    },
    "qwen_edit": {
        "family": "qwen_image",
        "mode": "edit",
        "build_request": _build_qwen_edit_request,
        "run_fn": "run_edit",
        "desc": "Qwen Edit fp8-lightning 640x640 明示 4steps",
    },
    "qwen_controlnet": {
        "family": "qwen_image",
        "mode": "controlnet",
        "build_request": _build_qwen_controlnet_request,
        "run_fn": "run_controlnet",
        "desc": "Qwen ControlNet canny fp8-lightning 624x944 4steps",
    },
    "qwen_layered": {
        "family": "qwen_image",
        "mode": "layered",
        "build_request": _build_qwen_layered_request,
        "run_fn": "run_layered",
        "desc": "Qwen Layered fp8-lightning 4層 640",
    },
    "flux2_t2i": {
        "family": "flux2",
        "mode": "t2i",
        "build_request": _build_flux2_t2i_request,
        "run_fn": "run_t2i",
        "desc": "FLUX.2 T2I 1024x1024 28steps",
    },
    "flux2_i2i": {
        "family": "flux2",
        "mode": "i2i",
        "build_request": _build_flux2_i2i_request,
        "run_fn": "run_i2i",
        "desc": "FLUX.2 I2I 参照1枚 1024x1024 28steps",
    },
}


# ============================================================================
# 環境情報
# ============================================================================
def collect_env_info() -> dict:
    info = {
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            info["gpu_total_vram_gb"] = round(total_bytes / (1024 ** 3), 1)
            free_bytes, _ = torch.cuda.mem_get_info()
            info["gpu_free_vram_gb_before_run"] = round(free_bytes / (1024 ** 3), 1)
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = str(exc)
    try:
        import diffusers
        info["diffusers_version"] = diffusers.__version__
    except Exception as exc:  # noqa: BLE001
        info["diffusers_error"] = str(exc)
    return info


# ============================================================================
# 1構成の計測(このプロセス内で実行。呼び出し元が in-process ループで複数構成を回す)
# ============================================================================
def run_one_config(config_name: str, args) -> dict:
    import torch

    from core import gpu
    from core.registry import registry

    # families を import すると import 副作用で registry に自動登録される
    # (app.py と同じ作法。REBUILD_PLAN/CLAUDE.md: "registry.load() 経由でないと
    # 自動 unload が効かない" ため、ここでも必ず registry.load() を通す)。
    import families.qwen_image  # noqa: F401
    import families.flux2  # noqa: F401

    cfg = CONFIGS[config_name]
    family_name = cfg["family"]
    mode = cfg["mode"]

    if family_name == "flux2" and args.flux2_offload:
        os.environ["DS_OFFLOAD"] = args.flux2_offload

    request = cfg["build_request"](args)
    request = {"mode": mode, **request}

    print(f"\n=== [{config_name}] {cfg['desc']} ===", flush=True)

    # --- cold: ロード込み ---
    t_cold0 = time.time()
    registry.load(family_name, mode)
    load_time_s = time.time() - t_cold0

    family = registry.get(family_name)

    gpu.reset_peak_stats()
    gpu.empty_cache()
    t0 = time.time()
    with gpu.generation_scope():
        meta_cold = family.generate(dict(request))
    cold_elapsed_s = time.time() - t0
    cold_peak_vram_gb = meta_cold.get("peak_vram_gb")

    # --- warm: 同一プロセス内、ロード済み状態での2回目生成 ---
    gpu.reset_peak_stats()
    gpu.empty_cache()
    with gpu.generation_scope():
        meta_warm = family.generate(dict(request))
    warm_elapsed_s = meta_warm.get("elapsed_s")
    warm_peak_vram_gb = meta_warm.get("peak_vram_gb")

    result = {
        "config": config_name,
        "desc": cfg["desc"],
        "family": family_name,
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        "load_time_s": round(load_time_s, 2),
        "cold_elapsed_s": round(cold_elapsed_s, 2),
        "cold_peak_vram_gb": round(cold_peak_vram_gb, 2) if cold_peak_vram_gb is not None else None,
        "warm_elapsed_s": round(warm_elapsed_s, 2) if warm_elapsed_s is not None else None,
        "warm_peak_vram_gb": round(warm_peak_vram_gb, 2) if warm_peak_vram_gb is not None else None,
        "request_summary": {k: v for k, v in request.items() if not k.startswith("_")},
        "env": collect_env_info(),
    }
    if family_name == "flux2":
        result["flux2_offload"] = os.environ.get("DS_OFFLOAD", "(default: model_cpu_offload)")

    print(
        f"[{config_name}] load={load_time_s:.1f}s cold={cold_elapsed_s:.2f}s "
        f"(peak {result['cold_peak_vram_gb']}GB) warm={result['warm_elapsed_s']}s "
        f"(peak {result['warm_peak_vram_gb']}GB)",
        flush=True,
    )
    return result


def append_result(result: dict) -> None:
    results = []
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, "r", encoding="utf-8") as f:
                results = json.load(f)
            if not isinstance(results, list):
                results = [results]
        except Exception:
            results = []
    results.append(result)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[benchmark] appended to {RESULTS_PATH}")


# ============================================================================
# サブプロセス分離モード(--isolated): 構成ごとにクリーンなプロセスで実行する。
# families 間の状態汚染や compile キャッシュの影響を避けたい場合に使う
# (旧 flux2_diffusers/benchmark.py の matrix モードと同じ発想)。既定は無効
# (同一プロセス内で複数構成を回した方が registry の自動 unload・VRAM リーク確認を
# 兼ねられるため。--isolated 時はこのリーク確認はできない点に注意)。
# ============================================================================
def run_isolated(config_name: str, args) -> dict:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--config", config_name,
        "--edit-size", str(args.edit_size),
        "--_worker",
    ]
    if args.flux2_offload:
        cmd += ["--flux2-offload", args.flux2_offload]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    print(proc.stdout[-3000:])
    print(proc.stderr[-3000:])
    return {"config": config_name, "error": f"subprocess exit {proc.returncode}"}


def main():
    parser = argparse.ArgumentParser(description="diffusers-server 統合ベンチマーク")
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("DS_BENCH_CONFIG", "qwen_t2i"),
        help=f"カンマ区切りで複数指定可。選択肢: {','.join(CONFIGS.keys())}",
    )
    parser.add_argument("--edit-size", type=int, default=640, help="qwen_edit の height=width(既定640、48GBでのOOM回避)")
    parser.add_argument(
        "--flux2-offload", type=str, default=None,
        choices=["none", "model", "model_cpu_offload", "group", "group_offload"],
        help="flux2_t2i/flux2_i2i の DS_OFFLOAD 上書き(既定は runtime.py の DEFAULT_OFFLOAD=model_cpu_offload)",
    )
    parser.add_argument("--isolated", action="store_true", help="構成ごとにクリーンなサブプロセスで実行する")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    config_names = [c.strip() for c in args.config.split(",") if c.strip()]
    for c in config_names:
        if c not in CONFIGS:
            parser.error(f"unknown --config {c!r}. choices: {list(CONFIGS)}")

    if args._worker:
        # --isolated から呼ばれるワーカーモード: 1構成だけ実行して RESULT 行を出す。
        assert len(config_names) == 1
        try:
            result = run_one_config(config_names[0], args)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            result = {"config": config_names[0], "error": str(exc)}
        print("RESULT " + json.dumps(result))
        return

    all_results = []
    for c in config_names:
        if args.isolated:
            result = run_isolated(c, args)
        else:
            try:
                result = run_one_config(c, args)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                result = {"config": c, "error": str(exc), "timestamp": datetime.now().isoformat()}
        append_result(result)
        all_results.append(result)

    print("\n=== summary ===")
    header = f"{'config':<18} {'load_s':>8} {'cold_s':>8} {'warm_s':>8} {'cold_VRAM':>10} {'warm_VRAM':>10}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        if "error" in r:
            print(f"{r['config']:<18} ERROR: {r['error']}")
            continue
        print(
            f"{r['config']:<18} {r['load_time_s']:>8} {r['cold_elapsed_s']:>8} "
            f"{r.get('warm_elapsed_s', '-'):>8} {r.get('cold_peak_vram_gb', '-'):>10} "
            f"{r.get('warm_peak_vram_gb', '-'):>10}"
        )


if __name__ == "__main__":
    main()
