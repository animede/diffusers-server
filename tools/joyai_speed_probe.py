# -*- coding: utf-8 -*-
"""JoyAI-Edit-Plus の速度異常(174s/step)の診断プローブ。

phase1 実測で 1024²・1参照・CFG時に 174s/step(期待値の数十倍)を観測した。
GPU 127W / util 58% / CPU 1コア飽和という症状から、以下を切り分ける:
  1. 解像度スケーリング: 512² と 1024² で 2 step ずつ計測
     -> attention の S² 律速なら 16 倍程度縮む。CPU 側の固定費律速なら縮まない
  2. torch.profiler で 1 step の CPU/CUDA 上位オペを取得
     -> Python/CPU のどこで時間が消えているかを直接特定

joyai_regress.py の load_pipeline()(component-wise device_map="cuda"、RAM安全版)を流用。
"""
import os
import sys
import time

import torch
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "tools"))
sys.path.insert(0, _REPO_ROOT)

from joyai_regress import load_pipeline, PORTRAIT_PATH  # noqa: E402

SEED = 12345


def timed_run(pipe, portrait, size, steps, cfg=4.0, tag=""):
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    torch.cuda.synchronize()
    t0 = time.time()
    pipe(
        images=[portrait],
        prompt="change the cardigan color to red",
        height=size,
        width=size,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=gen,
    )
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"[probe] {tag}: {size}x{size} steps={steps} cfg={cfg} -> total {dt:.1f}s "
          f"({dt/steps:.1f}s/step, 前処理込み)", flush=True)
    return dt


def main():
    pipe, load_t = load_pipeline()
    print(f"[probe] load {load_t:.1f}s", flush=True)
    portrait = Image.open(PORTRAIT_PATH).convert("RGB")

    # 解像度スケーリング(2 step ずつ)。
    # 注意: torch.profiler での1step解剖は実施しない。16Bモデルの174秒ステップを
    # フルトレースするとイベントログがホストRAMに大量に積まれ、OOMキラーの引き金に
    # なりうる(実際に初回プローブ実行時、profilerフェーズ前後でシステム巻き添えの
    # 強制終了が発生した疑い)。切り分けは解像度・CFGスケーリングの比率だけで行う。
    timed_run(pipe, portrait, 512, 2, tag="A: 512^2")
    timed_run(pipe, portrait, 1024, 2, tag="B: 1024^2")
    # CFGなし(forward 1回/step)の寄与も確認
    timed_run(pipe, portrait, 1024, 2, cfg=1.0, tag="C: 1024^2 cfg=1")
    print("[probe] done", flush=True)


if __name__ == "__main__":
    main()
