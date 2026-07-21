# -*- coding: utf-8 -*-
"""
生成中の進捗状態(グローバル1本)。

生成は core.gpu.generation_lock により全ファミリー横断で GPU 同時1件に排他されるため、
進捗状態もグローバルに1つ持てば足りる(REBUILD_PLAN の設計方針どおり)。

- 生成スレッド(app.py のリクエストハンドラスレッド、charsheet のジョブスレッド)が
  start_loading() / start_generating() / update_step() / finish() を呼んで更新する。
- API スレッド(GET /api/progress)は snapshot() で読むだけ。ロック不要・GPU不使用の
  非ブロッキング読み取りのみのため、生成中でも自由に呼べる。

スレッドセーフ化: 単純な dict 更新は GIL 下で読み取り側が壊れた値を見ることは
ほぼ無いが、複数フィールドの整合性(例: step と total_steps の組)を保証するため
threading.Lock で更新・読み取りとも保護する。ロックの保持時間は極小(dict代入のみ)
のため、生成スレッドのボトルネックにはならない。
"""
import threading
import time

__all__ = [
    "start_loading",
    "start_generating",
    "update_step",
    "set_phase",
    "finish",
    "snapshot",
]

_lock = threading.Lock()

_state = {
    "active": False,
    "mode": None,  # "t2i" | "i2i" | "edit" | ... | "flux2_t2i" | ... | "charsheet"
    "phase": None,  # "loading" | "generating" | "decoding"
    "step": 0,
    "total_steps": 0,
    "started_at": None,
    "extra": None,  # 任意の追加情報(charsheet の "n/8 方向" 等、dict)
}


def _reset_locked(mode: str, phase: str, extra=None) -> None:
    _state["active"] = True
    _state["mode"] = mode
    _state["phase"] = phase
    _state["step"] = 0
    _state["total_steps"] = 0
    _state["started_at"] = time.time()
    _state["extra"] = extra


def start_loading(mode: str, extra: "dict | None" = None) -> None:
    """モデルロード開始。ステップ不定(不確定バー表示用)。"""
    with _lock:
        _reset_locked(mode, "loading", extra)


def start_generating(mode: str, total_steps: int, extra: "dict | None" = None) -> None:
    """denoise ループ開始。total_steps=0 なら callback_on_step_end 非対応
    (呼び出し元が使えない)パイプライン向けの不確定表示にフォールバックする。
    """
    with _lock:
        if _state["mode"] != mode or _state["phase"] != "generating" or not _state["active"]:
            _reset_locked(mode, "generating", extra)
        _state["phase"] = "generating"
        _state["total_steps"] = total_steps
        _state["step"] = 0
        if extra is not None:
            _state["extra"] = extra


def update_step(step: int, total_steps: "int | None" = None) -> None:
    """callback_on_step_end から呼ぶ。step は 1-origin(1step目完了時点で1)。"""
    with _lock:
        if not _state["active"]:
            return
        _state["step"] = step
        if total_steps is not None:
            _state["total_steps"] = total_steps


def set_phase(phase: str, extra: "dict | None" = None) -> None:
    """phase のみ変更(例: "generating" -> "decoding")。step/total はリセットしない。"""
    with _lock:
        if not _state["active"]:
            return
        _state["phase"] = phase
        if extra is not None:
            _state["extra"] = extra


def finish() -> None:
    """生成完了・エラー・例外いずれでも呼ぶこと(try/finally 推奨)。"""
    with _lock:
        _state["active"] = False
        _state["phase"] = None
        _state["step"] = 0
        _state["total_steps"] = 0
        _state["started_at"] = None
        _state["extra"] = None


def snapshot() -> dict:
    """GET /api/progress 用。ロック不要・GPU不使用の読み取り専用スナップショット。"""
    with _lock:
        st = dict(_state)
    if st["started_at"] is not None:
        st["elapsed_s"] = time.time() - st["started_at"]
    else:
        st["elapsed_s"] = None
    return st
