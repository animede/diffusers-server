# -*- coding: utf-8 -*-
"""
tpose アプリ層(2026-07-26追加)。

1枚のキャラクター画像から **Tポーズ(両腕を水平に広げた姿勢)の4ビュー**
(正面 / 背面 / 左前45度 / 右前45度)を生成する。image-3d の
マルチビュー入力(Hunyuan3D-2mv)と rig-service(Tポーズ前提の自動リグ・VRM化)
向けの前処理を担う。

charsheet / scene_angles との違い(実機検証に基づく設計判断は
apps/tpose/prompts.py の冒頭コメント参照):
  - Multiple-angles LoRA を使わず、families/qwen_image の通常 Edit を使う
    (Tポーズでは同一性・速度とも優位)。
  - 真横(90度)ビューを持たない(Tポーズの真横投影は構造的に破綻する)。
  - front → 他ビューの2段生成(前面出力を参照画像に連鎖)。

app.py 側は `from apps.tpose import router` して
`app.include_router(router, prefix="/api/tpose")` するだけでよい。
"""
from apps.tpose.jobs import router

__all__ = ["router"]
