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

ターミナル進捗バー(CLAUDE.md 55番、DS_TERMINAL_PROGRESS): 上記の状態更新関数
(start_loading/start_generating/update_step/set_phase/finish)は、全ファミリー
(qwen_image/flux2/z_image/ltx2/joyai)・charsheet/scene_angles の多段ジョブが
共通で通る唯一の経路(REBUILD_PLAN の設計どおり)なので、ここに1箇所フックを
入れるだけでサーバ全体の進捗をターミナル(stderr)へ描画できる。
DS_TERMINAL_PROGRESS=0(既定)の間は `_terminal_enabled` が False のままで、
各更新関数の末尾で呼ぶ `_maybe_render()` が即 return するだけなので、既存の
状態管理ロジック・戻り値・呼び出し側の挙動は一切変化しない。
"""
import sys
import threading
import time

from core import config

__all__ = [
    "start_loading",
    "start_generating",
    "update_step",
    "set_phase",
    "finish",
    "snapshot",
]

_lock = threading.Lock()
_terminal_enabled = config.TERMINAL_PROGRESS
_render_lock = threading.Lock()  # stderr書き込み自体の直列化(_lockとは別、I/O用)
_last_render_ts = 0.0
_RENDER_THROTTLE_S = 0.15  # 0.1〜0.2秒スロットリング(仕様どおり)
_last_line_len = 0  # \r上書き時に前回行より短い場合の残留文字を消すためのパディング用
# 直前に描画した内容のキー(mode, phase, step, total_steps)。elapsed_s(経過秒)は含めない
# (56番、進捗バー二重出力バグ修正で追加)。finish() 確定行が直前の進行中行と実質同一
# (denoiseが1stepも進まないままエラー終了した場合等)かどうかの判定に使う。
_last_render_key: "tuple | None" = None

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
    _maybe_render()


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
    _maybe_render(force=True)  # phase切り替わり相当なのでスロットリングをバイパス


def update_step(step: int, total_steps: "int | None" = None) -> None:
    """callback_on_step_end から呼ぶ。step は 1-origin(1step目完了時点で1)。"""
    with _lock:
        if not _state["active"]:
            return
        _state["step"] = step
        if total_steps is not None:
            _state["total_steps"] = total_steps
    _maybe_render()


def set_phase(phase: str, extra: "dict | None" = None) -> None:
    """phase のみ変更(例: "generating" -> "decoding")。step/total はリセットしない。"""
    with _lock:
        if not _state["active"]:
            return
        _state["phase"] = phase
        if extra is not None:
            _state["extra"] = extra
    _maybe_render(force=True)  # phase切り替わりは必ず表示する(スロットリングをバイパス)


def finish() -> None:
    """生成完了・エラー・例外いずれでも呼ぶこと(try/finally 推奨)。"""
    _render_finish_line()  # active=False にする前に最終状態を確定行として描画する
    with _lock:
        _state["active"] = False
        _state["phase"] = None
        _state["step"] = 0
        _state["total_steps"] = 0
        _state["started_at"] = None
        _state["extra"] = None


# ============================================================================
# ターミナル進捗バー(DS_TERMINAL_PROGRESS=1 時のみ動作、CLAUDE.md 55番)
# ============================================================================
_BAR_WIDTH = 10


def _format_line(st: dict) -> "str | None":
    """snapshot 相当の dict から1行分のバー文字列を組み立てる(先頭の \\r は付けない)。

    - step粒度あり(phase="generating" かつ total_steps>0): ブロック文字バー + N/M (P%) + 経過秒。
    - step粒度なし(loading/decoding、または total_steps=0 の不確定 generating):
      phase名 + 経過秒のみ(バーは描画しない、"..." で不確定であることを示す)。
    - extra に direction_index/direction_total があれば "direction i/n" を差し込む
      (charsheet/scene_angles 等の多段ジョブ向け、仕様どおり)。
    """
    if not st["active"] or not st["mode"]:
        return None
    mode = st["mode"]
    phase = st["phase"] or "?"
    elapsed = st["elapsed_s"] or 0.0
    extra = st.get("extra") or {}

    direction = ""
    d_idx = extra.get("direction_index")
    d_total = extra.get("direction_total")
    if d_idx is not None and d_total is not None:
        direction = f" direction {d_idx}/{d_total}"

    total_steps = st["total_steps"] or 0
    step = st["step"] or 0
    if phase == "generating" and total_steps > 0:
        filled = int(_BAR_WIDTH * min(step, total_steps) / total_steps)
        bar = "█" * filled + "░" * (_BAR_WIDTH - filled)
        pct = int(100 * min(step, total_steps) / total_steps)
        return f"[{mode}]{direction} {phase} {bar} {step}/{total_steps} ({pct}%) {elapsed:.1f}s"
    # step粒度が無いフェーズ(モデルロード・VAEデコード等)や、total_steps不明の
    # generating(callback_on_step_end非対応パイプライン向けフォールバック): phase名+経過秒のみ。
    return f"[{mode}]{direction} {phase}... {elapsed:.1f}s"


def _render_key(st: dict) -> tuple:
    """再描画要否の判定キー(mode, phase, step, total_steps)。経過秒は含めない
    (56番: 経過秒だけが違う実質同一内容の行を「別内容」と誤判定しないため)。
    """
    return (st.get("mode"), st.get("phase"), st.get("step"), st.get("total_steps"))


def _write_line(line: str, *, final: bool, key: "tuple | None" = None) -> None:
    """stderr へ1行書く。final=False は \\r で上書きし続ける行、final=True は
    確定行として改行する(仕様: finish時は改行して確定行)。
    """
    global _last_line_len, _last_render_key
    with _render_lock:
        pad = max(0, _last_line_len - len(line))
        sys.stderr.write("\r" + line + (" " * pad))
        if final:
            sys.stderr.write("\n")
            _last_line_len = 0
        else:
            sys.stderr.flush()
            _last_line_len = len(line)
        _last_render_key = key


def _maybe_render(force: bool = False) -> None:
    """DS_TERMINAL_PROGRESS=1 の間だけ、0.1〜0.2秒スロットリングでバーを描画する。

    force=True(phase切り替わり相当のタイミング: start_generating/set_phase)は
    スロットリングをバイパスして必ず描画する(フェーズ変化を取りこぼさないため)。
    OFF(既定)時は snapshot() を呼ぶことすらせず即 return する(オーバーヘッドなし)。
    """
    if not _terminal_enabled:
        return
    global _last_render_ts
    now = time.time()
    if not force and (now - _last_render_ts) < _RENDER_THROTTLE_S:
        return
    _last_render_ts = now
    st = snapshot()
    line = _format_line(st)
    if line is not None:
        _write_line(line, final=False, key=_render_key(st))


def _render_finish_line() -> None:
    """finish() 直前に、確定行(末尾改行あり)として最終状態を1回描画する。

    active=False にする前の状態を読む必要があるため、finish() 本体より先に呼ぶこと。

    2026-07-24修正(CLAUDE.md 56番、進捗バー二重出力バグ): OOM等、denoiseループへ
    1stepも入らないまま例外終了するケースでは、直前に「generating 0/30 (0%) 0.0s」を
    描画した直後に例外が発生し、finish()が全く同じ内容(step/phase/total_steps不変、
    経過秒だけ微増)をもう一度描画していた。ターミナル上は同一のバーが2回連続で
    (間に例外trace付きで)出るように見え、実質ノイズでしかない。直前の描画キー
    (mode, phase, step, total_steps。経過秒を含まない)と今回描画しようとしている
    内容が一致する場合は、新しい行としては書かず、直前の \\r 上書き行をそのまま
    改行して確定させるだけにする(内容自体は直前の描画で既にターミナルに出ている)。
    進捗が実際に進んだ場合(step増加やphase変化を伴う正常終了)は従来どおり
    確定行が描画される(キーが変わるため)。
    """
    global _last_line_len, _last_render_key
    if not _terminal_enabled:
        return
    st = snapshot()
    line = _format_line(st)
    if line is None:
        return
    key = _render_key(st)
    if key == _last_render_key:
        # 内容不変: 直前の \r 上書き行を改行するだけ(重複した行の再描画を避ける)。
        with _render_lock:
            sys.stderr.write("\n")
            _last_line_len = 0
            _last_render_key = None
        return
    _write_line(line, final=True, key=None)


def snapshot() -> dict:
    """GET /api/progress 用。ロック不要・GPU不使用の読み取り専用スナップショット。"""
    with _lock:
        st = dict(_state)
    if st["started_at"] is not None:
        st["elapsed_s"] = time.time() - st["started_at"]
    else:
        st["elapsed_s"] = None
    return st


def disable_diffusers_tqdm(pipe) -> None:
    """DS_TERMINAL_PROGRESS=1 時のみ、diffusers パイプライン自前の tqdm
    (denoise ループの "25%|██▌|..." 進捗バー)を抑制する(CLAUDE.md 55番)。

    各ファミリーの get_*_pipeline() 系関数(初回ロード完了・return 直前)から呼ぶ。
    `pipe.set_progress_bar_config(disable=True)` は `DiffusionPipeline` が持つ設定用
    dict への代入のみ(`self._progress_bar_config = {...}`)で GPU 操作を伴わず、
    ロード済みパイプラインに対して何度呼んでも安全(冪等)。個々のサブパイプライン
    (I2V/FLF/IC-LoRA 等、base.components から遅延構築される派生パイプライン)は
    base とは別の tqdm 設定を持つため、派生パイプラインが新規構築されるたびに
    個別に呼ぶ必要がある(呼び出し側で対応済み)。

    DS_TERMINAL_PROGRESS=0(既定)の間は何もしない(従来どおり diffusers 標準の
    tqdm がそのまま stdout/stderr に出る)。`set_progress_bar_config` を持たない
    オブジェクト(念のためのガード)が渡された場合も黙って無視する。
    """
    if not _terminal_enabled:
        return
    set_config = getattr(pipe, "set_progress_bar_config", None)
    if set_config is None:
        return
    try:
        set_config(disable=True)
    except Exception:  # noqa: BLE001
        # tqdm抑制はあくまで見た目の改善であり、失敗しても生成自体を止めるべきではない
        # (「壊すより安全優先」の方針どおり、二重表示を許容してでも生成を継続する)。
        pass
