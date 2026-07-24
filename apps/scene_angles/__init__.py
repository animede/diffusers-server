# -*- coding: utf-8 -*-
"""
scene_angles アプリ層(2026-07-24追加、CLAUDE.md 53番)。

1枚のシーン画像から「カメラ指示プロンプト8種のEdit」で同一シーンの8アングル画像を
生成する。ComfyUIワークフロー templates-1_click_multiple_scene_angles-v1.0_api.json
(Qwen-Image-Edit-2509 fp8 + Multiple-angles LoRA + Lightning 4steps の並列Edit×8)の
diffusers-server版。

パイプライン層は charsheet と完全に同一(edit_angles系グループ、既定 bf16-group)のため、
apps.charsheet.generate をそのまま流用する(charsheet側のコードは一切変更しない)。
違いはプロンプトセット(キャラ8方向 → カメラアングル8種)と後処理
(シート合成・背景除去なし)のみ。

app.py 側は `from apps.scene_angles import router` して
`app.include_router(router, prefix="/api/scene_angles")` するだけでよい。
"""
from apps.scene_angles.jobs import router

__all__ = ["router"]
