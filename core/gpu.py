# -*- coding: utf-8 -*-
"""
VRAM 計測(ピーク計測含む)、生成グローバルロック(GPU 同時1件)、empty_cache 方針。

抽出元:
  - Ｑwenimage-edit-diffusers/pipeline_manager.py
    - generation_lock = threading.Lock() (行664) — app.py から使う生成排他ロック
    - _free_vram_gb() (行354-358)
    - get_status() の vram セクション (行1658-1667):
      allocated_gb / max_allocated_gb / reserved_gb / free_gb / total_gb
    - unload() 内の gc.collect() + torch.cuda.empty_cache() +
      torch.cuda.reset_peak_memory_stats() (行1616-1619)

方針:
  - 「生成前に empty_cache」は CLAUDE.md 8番の知見(expandable_segments と併用してOOM対策)を
    踏襲し、generation_scope() コンテキストマネージャとして提供する。元実装は各 app.py の
    エンドポイントで個別に torch.cuda.empty_cache() を呼んでいた(pipeline_manager.py 自体には
    その呼び出し箇所はないが、CLAUDE.md 8番に「app.py の各生成エンドポイントで生成直前に
    torch.cuda.empty_cache() を呼ぶこと(既に実装済み)」とあるため、Phase 1 で app.py を
    移植する際にこの関数を使えるよう先に用意する)。
  - torch の import はモジュールトップレベルで行うが、CUDA 初期化を伴う呼び出し
    (torch.cuda.*)は関数内に閉じ込める(import 自体は CUDA を初期化しない)。
  - 生成ロックは全ファミリー横断で1本(REBUILD_PLAN §3.1: "生成は全ファミリー横断で
    グローバルロック1本(GPU 同時1件)")。ファミリーごとに別ロックを作らないこと。
"""
import threading
import time
from contextlib import contextmanager

import torch

__all__ = [
    "generation_lock",
    "is_cuda_available",
    "free_vram_gb",
    "vram_snapshot",
    "empty_cache",
    "reset_peak_stats",
    "generation_scope",
    "available_ram_gb",
]

# 生成の排他制御(GPU 同時1件)。families/* すべてがこの1本を共有する
# (抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py 行664 generation_lock)。
generation_lock = threading.Lock()


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def free_vram_gb() -> float:
    """空き VRAM を GB 単位で返す。CUDA が無ければ 0.0。

    抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py の _free_vram_gb()(行354-358)。
    """
    if not torch.cuda.is_available():
        return 0.0
    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    return free_bytes / (1024 ** 3)


def available_ram_gb() -> "float | None":
    """ホスト RAM の空き(MemAvailable、GB単位)を返す。/proc/meminfo が無ければ None。

    "bf16-group"(block-level group offload、CLAUDE.md 33番)方式は transformer
    (bf16 約38GB)をホスト RAM にピン留めしてブロック単位で GPU へストリームする設計のため、
    ロード前に十分な空き RAM があることを確認する用途で使う(新規依存を避けるため
    psutil は使わず /proc/meminfo の MemAvailable を直接読む。MemAvailable は
    buff/cache の再利用可能分を差し引いた「実質使える空き」を示す Linux カーネルの指標)。
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
    except (OSError, ValueError, IndexError):
        return None
    return None


def vram_snapshot() -> "dict | None":
    """現在の VRAM 使用状況のスナップショットを返す。CUDA が無ければ None。

    抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py の get_status() 内 vram セクション
    (行1658-1667)。キー名(allocated_gb / max_allocated_gb / reserved_gb / free_gb / total_gb)
    は既存 API レスポンス互換のためそのまま踏襲する。
    """
    if not torch.cuda.is_available():
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
        "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
        "free_gb": free_bytes / (1024 ** 3),
        "total_gb": total_bytes / (1024 ** 3),
    }


def empty_cache() -> None:
    """torch.cuda.empty_cache() のラッパー(CUDA 無ければ何もしない)。"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_peak_stats() -> None:
    """ピーク VRAM 計測用カウンタをリセットする(unload() 相当の後処理で使う)。

    抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py の unload() 内
    torch.cuda.reset_peak_memory_stats() 呼び出し(行1619)。
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


@contextmanager
def generation_scope():
    """1回の生成をこのコンテキストで包む: グローバルロック取得 -> 生成前 empty_cache ->
    生成 -> (呼び出し側で計測) -> ロック解放。

    CLAUDE.md 8番「OOM 対策: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True + 生成前
    empty_cache()」を踏襲。ピーク VRAM を計測したい呼び出し側は、この with ブロックに
    入る前に reset_peak_stats() を呼び、ブロックを抜けた後に vram_snapshot()["max_allocated_gb"]
    を読めばよい。

    使用例(Phase 1 の families/qwen_image.py 想定):
        with gpu.generation_scope():
            image = pipe(**kwargs).images[0]
    """
    with generation_lock:
        empty_cache()
        t0 = time.time()
        try:
            yield t0
        finally:
            pass
