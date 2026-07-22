# -*- coding: utf-8 -*-
"""
Qwen-Image ファミリーのシングルトン状態(グループ辞書群 + プロセス内ロック)。

抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py 行660-771
  - _lock, _runtime_config, _offload_mode
  - _shared / _t2i_group / _edit_group /
    _controlnet_union_group / _controlnet_inpaint_group / _layered_group
  - _get_runtime_config() / _get_offload_mode()

注意(REBUILD_PLAN 申し送り事項): この _lock は非再入。core.registry.FamilyRegistry.load()
が既に1本のロック(_registry_lock)を取った状態で family.load(mode) を呼ぶが、
registry 側のロックとこの _lock は別物(このファミリー内部のグループ別ロード制御用)。
families 側の "_load_*_locked" 関数はこのモジュールの _lock を呼び出し側が
保持している前提の規約を維持する(二重ロックしない。旧 CLAUDE.md 13番の罠)。
"""
import threading

from core.gpu import free_vram_gb
from core.optimize import resolve_offload_mode

from families.qwen_image.runtime import RuntimeConfig

# ============================================================================
# シングルトン状態
# ============================================================================
lock = threading.Lock()

_runtime_config: "RuntimeConfig | None" = None
_offload_mode: "str | None" = None  # プロセス内で一度決定したら使い回す

shared = {
    "vae": None,
    "text_encoder": None,
    "tokenizer": None,
    "loaded": False,
    "load_time_s": None,
}

# T2I/I2I グループ。DS_T2I_MODEL(またはリクエストの model)に応じて
# 無印 Qwen/Qwen-Image または Qwen/Qwen-Image-2512 のいずれか一方の transformer を保持する
# (2026-07-18変更: 旧 2512-4bit の独立グループ t2i_2512_group は廃止し、2512 も
# 無印と同じ fp8-lightning fuse 構造でこのグループに統合した。どちらのモデルが
# 入っているかは runtime_config.t2i_model が示す)。
t2i_group = {
    "transformer": None,
    "t2i_pipe": None,
    "i2i_pipe": None,
    "scheduler": None,
    "loaded": False,
    "load_time_s": None,
    "fallback": False,
    "quant": None,  # None(非量子化) または "Q4_K_M" 等 / "fp8-lightning"
    "lora_available": None,  # None=未試行, True/False
    "lora_unavailable_reason": None,
    "lightning_merged": False,  # fp8-lightning: LoRAが重みにfuse済みで無効化不可
}

edit_group = {
    "transformer": None,
    "edit_pipe": None,
    "scheduler": None,
    "processor": None,
    "loaded": False,
    "load_time_s": None,
    "fallback": False,
    "quant": None,
    "lora_available": None,
    "lora_unavailable_reason": None,
    "lightning_merged": False,  # fp8-lightning: LoRAが重みにfuse済みで無効化不可
}

# charsheet 専用の fp8-lightning-angles 変種(bf16 2511 -> Lightning fuse ->
# Multiple-angles fuse(weight 1.0) -> fp8 layerwise cast)。通常の edit_group とは
# 別グループとして管理し、同時常駐しない(families/qwen_image/edit_angles.py 参照。
# 排他は families/qwen_image/family.py の load()/generate() 側で edit <-> edit_angles
# の相互排他として実装する)。
edit_angles_group = {
    "transformer": None,
    "edit_pipe": None,
    "scheduler": None,
    "processor": None,
    "loaded": False,
    "load_time_s": None,
    "fallback": False,
    "quant": None,
    "lightning_merged": False,  # 常に True(fuse済みで無効化不可)。edit_groupと形を揃える
}

# charsheet 専用の fp8-base-adapters 変種(旧 /home/animede/charsheet 方式への復元)。
# bf16 2511 transformer をロードしてから fuse せずベースのみ fp8_e4m3fn ストレージ化し、
# その上に Lightning + Multiple-angles を「adapter として適用」する:
# load_lora_weights ×2 -> set_adapters(["lightning","angles"], adapter_weights=[1.0, 1.0])。
# 旧 pipeline.py の _apply_loras() と数値的に同等になる(fuseしていないのでLoRA差分が
# fp8の量子化グリッドに丸め込まれない。families/qwen_image/edit_angles_bf16.py 参照)。
# 注意(事故の教訓): 当初は「fuse・fp8化しない純bf16常駐(~38GB)」設計だったが、
# 48GB VRAMに収まらずtransformerをホストRAMへ退避するスワップ機構を実装した結果、
# システムフリーズ事故を2回起こした。fp8ベース化により transformer 常駐サイズを
# ~19-20GBまで落とし、transformerのCPU退避を完全に不要にしたのが現行設計。
# edit_angles_group(fp8-lightning-angles、両LoRAを重みにfuse)とは別グループとして
# 管理する(相互に排他: edit / edit_angles / edit_angles_bf16 / t2i)。
edit_angles_bf16_group = {
    "transformer": None,
    "edit_pipe": None,
    "scheduler": None,
    "processor": None,
    "loaded": False,
    "load_time_s": None,
    "fallback": False,
    "quant": None,
    "lightning_merged": False,  # adapter方式なので常に False(disable_lora()で無効化可能)
}

# charsheet 専用の "bf16-group" 変種(旧 /home/animede/charsheet 完全再現、CLAUDE.md 33番、
# 新既定)。bf16 2511 transformer をfuse・fp8化せずCPUへstreamingロードし、Lightning +
# Multiple-angles を adapter として適用した上で、transformer を block-level group
# offloading(旧 pipeline.py _apply_group_offload_to_transformer() の完全移植、
# families/qwen_image/edit_angles_bf16group.py 参照)で GPU/CPU 入れ替える。
# transformer は常時ホストRAM上に bf16 のまま(~38GB)ピン留めされる点が
# edit_angles_bf16_group(fp8ベース化、~19-20GB常駐)と異なる(VRAM収支ではなく
# ホストRAM収支が問題になる方式のため、ロード前にRAM空きガードを行う)。
# edit / edit_angles / edit_angles_bf16 / edit_angles_bf16group は相互に排他
# (families/qwen_image/family.py _unload_other_edit_variants() 参照)。
edit_angles_bf16group_group = {
    "transformer": None,
    "edit_pipe": None,
    "scheduler": None,
    "processor": None,
    "loaded": False,
    "load_time_s": None,
    "fallback": False,
    "quant": None,
    "lightning_merged": False,  # adapter方式なので常に False
    "group_offload_config": None,
}

# ControlNet(Union: canny/depth/pose等 + Inpainting)。base Qwen-Image の transformer/vae/
# text_encoder を共有し、ControlNetモジュールのみ追加ロードする(t2i_group に依存)。
controlnet_union_group = {
    "controlnet": None,
    "pipe": None,
    "scheduler": None,
    "loaded": False,
    "load_time_s": None,
}

controlnet_inpaint_group = {
    "controlnet": None,
    "pipe": None,
    "scheduler": None,
    "loaded": False,
    "load_time_s": None,
}

# Qwen-Image-Layered(画像 -> 複数RGBAレイヤー分解)。vae は専用ロード、
# text_encoder/tokenizer は shared を再利用(sha256一致を確認済み)。
layered_group = {
    "transformer": None,
    "vae": None,
    "processor": None,
    "pipe": None,
    "scheduler": None,
    "loaded": False,
    "load_time_s": None,
    "quant": None,  # None(非量子化bf16) / "Q4_K_M" 等 / "fp8-lightning"
    "lightning_merged": False,  # fp8-lightning: LoRAが重みにfuse済みで無効化不可
}


def get_runtime_config() -> RuntimeConfig:
    global _runtime_config
    if _runtime_config is None:
        _runtime_config = RuntimeConfig()
    return _runtime_config


def get_offload_mode(small_transformer_active: bool = False) -> str:
    """オフロードモードはプロセス内で一度だけ決定し、以後のロードで使い回す
    (共有コンポーネントへの二重フック登録を避けるため)。

    small_transformer_active=True で最初に呼ばれた場合、"none"判定の閾値を
    core.config.GGUF_VRAM_FREE_THRESHOLD_GB まで緩和する(GGUF量子化 transformer は
    小さいため、より少ない空きVRAMでも全常駐できる)。

    抽出元: pipeline_manager.py _get_offload_mode()(行755-771)。ロジック無変更、
    _auto_offload_mode/_resolve_offload_mode は core.optimize.resolve_offload_mode に委譲。
    """
    global _offload_mode
    if _offload_mode is None:
        config = get_runtime_config()
        free_gb = free_vram_gb()
        _offload_mode = resolve_offload_mode(config.offload, free_gb, small_transformer_active=small_transformer_active)
        print(
            f"[families.qwen_image] free VRAM: {free_gb:.1f} GB -> offload_mode={_offload_mode}"
            f"{' (小型transformer向け閾値緩和適用)' if small_transformer_active else ''}"
        )
    return _offload_mode


def t2i_load_device(offload_mode: str):
    """抽出元: pipeline_manager.py _t2i_load_device()(行804-807)。"""
    import torch

    if offload_mode == "none":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return "cpu"


def reset_offload_mode_cache() -> None:
    """テスト/unload時に offload_mode の再決定を許可したい場合に使う(元実装には無いが、
    unload("all") 後に次のロードで空きVRAMが変わっている可能性を考慮した安全策として用意。
    元実装は明示的にリセットしていなかったため、既定の unload() 実装ではこれを呼ばない
    (挙動互換を優先)。
    """
    global _offload_mode
    _offload_mode = None
