# -*- coding: utf-8 -*-
"""
Phase 2 タスク1: diffusers 0.40.0.dev0 での Flux2Pipeline 回帰確認スタンドアロンスクリプト。

flux2_diffusers/pipeline_manager.py は診断上、旧 diffusers バージョン(0.36系)で動作確認
されていた。REBUILD_PLAN §5 の筆頭リスク「Flux2 と diffusers git 版の互換性が0.40.devで
未検証」を解消するため、bnb-4bit + model_cpu_offload で T2I 1枚生成を試す。

実行:
  venv/bin/python tests/regress_flux2.py

成功条件: エラーなく1枚のPNGが outputs/ に保存され、サイズ・所要時間・ピークVRAMが出力される。
失敗時: フルスタックトレースを出力して終了コード1。
"""
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# core.config を先に import して expandable_segments を適用(他ファミリーと同じ作法)。
from core import config  # noqa: F401,E402

import torch  # noqa: E402
from diffusers import Flux2Pipeline  # noqa: E402

MODEL_ID_BNB_4BIT = "diffusers/FLUX.2-dev-bnb-4bit"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main() -> int:
    print(f"[regress_flux2] diffusers version check begins. torch={torch.__version__}, cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        print(f"[regress_flux2] free VRAM before load: {free_bytes/1024**3:.1f} / {total_bytes/1024**3:.1f} GB")

    t0 = time.time()
    try:
        print(f"[regress_flux2] loading Flux2Pipeline.from_pretrained({MODEL_ID_BNB_4BIT!r}, torch_dtype=bfloat16) ...")
        pipe = Flux2Pipeline.from_pretrained(MODEL_ID_BNB_4BIT, torch_dtype=torch.bfloat16)
        print(f"[regress_flux2] from_pretrained OK in {time.time() - t0:.1f}s")
    except Exception:
        print("[regress_flux2] FAILED at from_pretrained()")
        traceback.print_exc()
        return 1

    try:
        print("[regress_flux2] enable_model_cpu_offload() ...")
        pipe.enable_model_cpu_offload()
        print("[regress_flux2] enable_model_cpu_offload OK")
    except Exception:
        print("[regress_flux2] FAILED at enable_model_cpu_offload()")
        traceback.print_exc()
        return 1

    prompt = "a photo of an astronaut riding a horse on the moon, cinematic lighting"
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        print("[regress_flux2] running T2I generation (image=None) ...")
        t1 = time.time()
        result = pipe(
            prompt=prompt,
            image=None,
            num_inference_steps=28,
            guidance_scale=4.0,
            width=1024,
            height=1024,
            generator=torch.Generator(device="cpu").manual_seed(42),
        )
        elapsed = time.time() - t1
        image = result.images[0]
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
        out_path = os.path.join(OUTPUT_DIR, "flux2_regress_t2i.png")
        image.save(out_path)
        size_bytes = os.path.getsize(out_path)
        print(
            f"[regress_flux2] SUCCESS: generated in {elapsed:.2f}s, peak_vram={peak_gb:.2f}GB, "
            f"saved {out_path} ({size_bytes} bytes, size={image.size})"
        )
        return 0
    except Exception:
        print("[regress_flux2] FAILED at pipe() generation call")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
