# -*- coding: utf-8 -*-
"""
charsheet アプリ層(Phase 3)。

旧 /home/animede/charsheet/(app.py・pipeline.py・prompts.py・sheet.py・split.py・bg.py)
の移植。生成は旧 pipeline.py(QwenImageEditPlusPipeline を独自ロード + Lightning/
Multiple-angles LoRA)を使わず、families/qwen_image の Edit(fp8-lightning、
プロンプトのみ多アングル)を core.registry 経由で利用する(REBUILD_PLAN §4 Phase 3)。

app.py 側は `from apps.charsheet import router` して
`app.include_router(router, prefix="/api/charsheet")` するだけでよい。
"""
from apps.charsheet.jobs import router

__all__ = ["router"]
