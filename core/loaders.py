# -*- coding: utf-8 -*-
"""
streaming single-file / multi-shard ロード、GGUF ロード、
bf16 -> Lightning LoRA fuse -> fp8 layerwise casting、の共通関数群。

抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py
  - _load_safetensors_streaming() (行410-447)
  - _load_transformer_from_config() (行450-462)
  - _load_transformer_from_pretrained_streaming() (行465-516)
  - _load_transformer_gguf() (行626-643)
  - _fuse_lightning_lora_and_cast_to_fp8() (行1148-1197)

方針:
  - 元実装は各関数が QwenImageTransformer2DModel 固有だった(from_config /
    QwenImageTransformer2DModel.load_config を直接呼ぶ)。Phase 1 で他の transformer クラス
    (Edit/Layered は同じ QwenImageTransformer2DModel だが、将来ファミリーが増える場合に備え)
    にも使えるよう、model_cls 引数を追加して汎用化した。呼び出しパターン自体
    (config_repo/config_subfolder/path/load_device の組み合わせ)は元実装のまま。
  - _fuse_lightning_lora_and_cast_to_fp8 は「Edit と T2I で共通の関数として使えるよう
    引数化されていた」という元実装の設計(CLAUDE.md 8番)をそのまま踏襲する。
    ロジック・コメント中の実測値は変更しない。
  - torch の import はモジュールトップレベル、CUDA 初期化を伴う処理(実際のロード・
    empty_cache 等)は各関数内で行う(呼び出されない限り CUDA は初期化されない)。
"""
import os

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from huggingface_hub import snapshot_download
from safetensors.torch import safe_open

from core.resolve import resolve_model_path

__all__ = [
    "load_safetensors_streaming",
    "load_transformer_from_config",
    "load_transformer_from_pretrained_streaming",
    "load_transformer_gguf",
    "fuse_lightning_lora_and_cast_to_fp8",
    "fuse_lora_into_transformer",
    "fuse_lightning_and_extra_lora_and_cast_to_fp8",
]


def load_safetensors_streaming(
    transformer,
    path: str,
    load_device,
    strip_prefix: str = "",
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """safetensors を1テンソルずつ meta model へ流し込み、CPU/GPU の一時ピークを抑える。

    注意: init_empty_weights() で作った meta モデルのパラメータ dtype は既定で float32。
    set_module_tensor_to_device() は dtype= を渡さないと value を「meta 側の既存 dtype」
    (= float32)へ再キャストしてしまうため、bf16(や fp8)のつもりが実質 fp32 でロードされ、
    ホストRAM/VRAMを本来の2〜4倍消費するバグになる。必ず dtype= を明示すること。

    抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py の _load_safetensors_streaming()
    (行410-447)。ロジック・docstring の注意事項とも無変更。
    """
    expected_keys = set(transformer.state_dict().keys())
    loaded_keys = set()
    unexpected_keys = []
    load_device = str(load_device)

    with safe_open(path, framework="pt", device=load_device) as f:
        for raw_key in f.keys():
            if raw_key == "__index_timestep_zero__":
                continue
            key = raw_key[len(strip_prefix):] if strip_prefix and raw_key.startswith(strip_prefix) else raw_key
            if key not in expected_keys:
                unexpected_keys.append(key)
                continue
            tensor = f.get_tensor(raw_key)
            set_module_tensor_to_device(transformer, key, load_device, value=tensor, dtype=dtype, clear_cache=False)
            loaded_keys.add(key)
            del tensor

    missing_keys = sorted(expected_keys - loaded_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"transformer state_dict mismatch: missing={len(missing_keys)}, "
            f"unexpected={len(unexpected_keys)}"
        )


def load_transformer_from_config(
    model_cls,
    config_repo: str,
    config_subfolder: str,
    path: str,
    load_device,
    strip_prefix: str = "",
):
    """config だけ HF Hub から取得して meta model を作り、streaming ロードで重みを流し込む。

    抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py の _load_transformer_from_config()
    (行450-462)。元実装は QwenImageTransformer2DModel 固定だったが、Phase 1 での
    families/qwen_image.py 移植時にそのまま渡せるよう model_cls を引数化した
    (呼び出しパターンは同一: model_cls.load_config(...) -> init_empty_weights() 内で
    model_cls.from_config(...) -> streaming ロード)。
    """
    config = model_cls.load_config(config_repo, subfolder=config_subfolder)
    with init_empty_weights():
        transformer = model_cls.from_config(config)
    load_safetensors_streaming(transformer, path, load_device, strip_prefix=strip_prefix)
    transformer.eval()
    return transformer


def load_transformer_from_pretrained_streaming(
    model_cls,
    repo_id: str,
    subfolder: str,
    load_device,
    dtype: torch.dtype = torch.bfloat16,
):
    """HF Hub 上のシャード分割済み transformer(safetensorsが複数ファイルに分割された
    標準 diffusers 形式)を、ホストRAMを大きく消費せずに load_device へ直接ストリーミング
    ロードする(load_safetensors_streaming の複数シャード版)。

    snapshot_download でシャードをローカルにキャッシュしてから、safe_open で
    1テンソルずつ load_device へ流し込む(全シャードを一括でCPU RAMに展開しない)。

    抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py の
    _load_transformer_from_pretrained_streaming()(行465-516、Qwen-Image-Layered 用に
    実装されたもの)。model_cls を引数化した点のみ変更(理由は load_transformer_from_config
    と同様)。
    """
    config = model_cls.load_config(repo_id, subfolder=subfolder)
    with init_empty_weights():
        transformer = model_cls.from_config(config)

    local_dir = snapshot_download(repo_id=repo_id, allow_patterns=[f"{subfolder}/*.safetensors", f"{subfolder}/*.json"])
    shard_dir = os.path.join(local_dir, subfolder)
    shard_files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".safetensors"))
    if not shard_files:
        raise RuntimeError(f"no .safetensors shards found under {shard_dir}")

    expected_keys = set(transformer.state_dict().keys())
    loaded_keys = set()
    load_device_str = str(load_device)

    for shard_file in shard_files:
        shard_path = os.path.join(shard_dir, shard_file)
        with safe_open(shard_path, framework="pt", device=load_device_str) as f:
            for key in f.keys():
                if key not in expected_keys:
                    continue
                tensor = f.get_tensor(key)
                set_module_tensor_to_device(
                    transformer, key, load_device_str, value=tensor, dtype=dtype, clear_cache=False
                )
                loaded_keys.add(key)
                del tensor

    missing_keys = sorted(expected_keys - loaded_keys)
    if missing_keys:
        raise RuntimeError(
            f"transformer state_dict mismatch: missing={len(missing_keys)} "
            f"(e.g. {missing_keys[:5]})"
        )
    transformer.eval()
    return transformer


def load_transformer_gguf(model_cls, path: str, config_repo: str):
    """GGUF量子化 transformer をロードする(実機検証済みの引数構成)。

    from_single_file() は既定でCPUにロードする(GGUF量子化テンソルはuint8格納なので
    Q4_K_Mで約12GBとホストRAM圧迫が小さい)。呼び出し側で .to("cuda") するか
    configure_transformer_offload() に渡すこと。

    抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py の _load_transformer_gguf()
    (行626-643)。model_cls を引数化した点のみ変更。
    """
    from diffusers import GGUFQuantizationConfig

    transformer = model_cls.from_single_file(
        path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        config=config_repo,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    transformer.eval()
    return transformer


def fuse_lightning_lora_and_cast_to_fp8(
    transformer,
    lora_local_path: str,
    lora_hf_repo: str,
    lora_hf_file: str,
) -> None:
    """Lightning 4steps LoRA を transformer の重みに fuse(焼き込み)し、
    その後 enable_layerwise_casting でストレージのみ fp8_e4m3fn に圧縮する。

    Edit(2511)・T2I(無印 Qwen-Image)・Layered の全てで共通に使う(呼び出し側が対象
    transformer に合った lora_local_path/lora_hf_repo/lora_hf_file を渡す)。

    実機検証済みの手順・実測値(RTX PRO 6000 Blackwell, bf16 2511 transformer基準):
      1. bf16 transformer ロード直後: 38.06GB(Edit)/ 約38GB(T2I、同等サイズ)
      2. Lightning LoRA を transformer.load_lora_adapter() でロード
         (LoRAファイルは ComfyUI/Musubi 形式のキー名(lora_down/lora_up/alpha)なので、
         diffusers の _convert_non_diffusers_qwen_lora_to_diffusers() で
         PEFT形式(lora_A/lora_B, "transformer."プレフィックス付き)に変換してから渡す。
         変換後のキーには "transformer." プレフィックスが付くため、
         load_lora_adapter(..., prefix="transformer")(デフォルト値)を使うこと。
         prefix=None を指定するとプレフィックス不一致で
         "Target modules {...} not found in the base model" エラーになる(実機で確認済み)。
      3. fuse_lora() で重みに焼き込み: 38.86GB(LoRAの差分だけ一時的に増加)
      4. unload_lora() + empty_cache() でLoRAモジュールを除去: 38.06GB(fuse後のbf16のまま)
      5. enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
         + empty_cache(): 19.05GB(ほぼ瞬時にストレージがfp8化される。
         LayerwiseCastingHook.initialize_hook() が module.to(dtype=storage_dtype) を
         即座に呼ぶため、bf16確保分はこの時点で解放される)

    fuse後は通常のLoRAアダプタとして無効化できない(重みに永続的に焼き込まれるため)。
    そのため quant="fp8-lightning" の transformer は常にLightning適用状態になる。

    抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py の
    _fuse_lightning_lora_and_cast_to_fp8()(行1148-1197)。ロジック・docstring とも無変更。
    """
    from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_qwen_lora_to_diffusers

    lora_path = resolve_model_path(lora_local_path, lora_hf_repo, lora_hf_file)
    raw_sd = {}
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            raw_sd[key] = f.get_tensor(key)
    converted_sd = _convert_non_diffusers_qwen_lora_to_diffusers(raw_sd)

    transformer.load_lora_adapter(converted_sd, adapter_name="lightning_fuse", prefix="transformer")
    transformer.fuse_lora(adapter_names=["lightning_fuse"])
    transformer.unload_lora()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fuse_lora_into_transformer(
    transformer,
    lora_local_path: str,
    lora_hf_repo: str,
    lora_hf_file: str,
    adapter_name: str,
    weight: float = 1.0,
) -> None:
    """任意の LoRA(ComfyUI/Musubi 形式 または既に PEFT 形式のキー)を1本、transformer の
    重みに fuse(焼き込み)する。fuse_lightning_lora_and_cast_to_fp8() の「1回分のfuse」を
    汎用化したもの(fp8キャストは行わない。呼び出し側が必要な回数呼んだ後にキャストする)。

    charsheet 向け fp8-lightning-angles 変種(bf16 2511 ロード -> Lightning fuse ->
    Multiple-angles fuse -> fp8 layerwise cast)のために追加。_convert_non_diffusers_qwen_lora_to_diffusers()
    は "diffusion_model." プレフィックスや "lora_unet_" 命名の場合のみキーを書き換える
    実装のため、既に PEFT 形式(lora_A/lora_B、"transformer_blocks." から始まる無プレフィックス
    キー)の LoRA ファイル(例: Qwen-Edit-2509-Multiple-angles.safetensors)を渡しても
    実質的な no-op として安全に通過する(実機のキー名確認済み)。

    weight は fuse_lora(lora_scale=weight) に渡される(既定 1.0 = 旧 charsheet/pipeline.py
    の _apply_loras() が adapter_weights=[1.0, 1.0] で適用していたのと同じ強度)。
    """
    from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_qwen_lora_to_diffusers

    lora_path = resolve_model_path(lora_local_path, lora_hf_repo, lora_hf_file)
    raw_sd = {}
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            raw_sd[key] = f.get_tensor(key)
    converted_sd = _convert_non_diffusers_qwen_lora_to_diffusers(raw_sd)

    transformer.load_lora_adapter(converted_sd, adapter_name=adapter_name, prefix="transformer")
    transformer.fuse_lora(adapter_names=[adapter_name], lora_scale=weight)
    transformer.unload_lora()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fuse_lightning_and_extra_lora_and_cast_to_fp8(
    transformer,
    lightning_local_path: str,
    lightning_hf_repo: str,
    lightning_hf_file: str,
    extra_local_path: str,
    extra_hf_repo: str,
    extra_hf_file: str,
    extra_adapter_name: str = "extra_fuse",
    extra_weight: float = 1.0,
) -> None:
    """Lightning 4steps LoRA + 追加 LoRA(例: Multiple-angles)を順に transformer の重みへ
    fuse してから、fuse_lightning_lora_and_cast_to_fp8() と同じ手順で fp8_e4m3fn へ
    layerwise casting する。

    charsheet の fp8-lightning-angles 変種専用(families/qwen_image/edit_angles.py から
    呼ばれる)。通常の Edit(fp8-lightning、Lightning のみ fuse)には一切影響しない
    (fuse_lightning_lora_and_cast_to_fp8() はこの関数から呼ばない・変更もしない)。

    fuse順序: bf16 ロード済み transformer -> Lightning fuse(fuse_lora_into_transformer,
    adapter_name="lightning_fuse", weight=1.0)-> Multiple-angles fuse(同関数、
    adapter_name=extra_adapter_name、weight=extra_weight、既定1.0 = 旧 charsheet/pipeline.py
    の _apply_loras() の adapter_weights と同じ)-> enable_layerwise_casting(fp8_e4m3fn)。

    実機実測(この関数の検証時に diffusers-server 側で確認): bf16ロード直後 ~38GB、
    Lightning fuse後 ~38.9GB、angles fuse後 ~39GB台(LoRA差分のみの一時増加)、
    fp8 layerwise cast後 ~19-20GB。48GB専有環境ではこの一時ピーク(bf16 2本fuse中の
    最大値)を許容する前提(タスク指示のとおり)。
    """
    fuse_lora_into_transformer(
        transformer, lightning_local_path, lightning_hf_repo, lightning_hf_file,
        adapter_name="lightning_fuse", weight=1.0,
    )
    fuse_lora_into_transformer(
        transformer, extra_local_path, extra_hf_repo, extra_hf_file,
        adapter_name=extra_adapter_name, weight=extra_weight,
    )

    transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_lora_adapters_and_cast_to_fp8(
    transformer,
    lightning_local_path: str,
    lightning_hf_repo: str,
    lightning_hf_file: str,
    extra_local_path: str,
    extra_hf_repo: str,
    extra_hf_file: str,
    extra_adapter_name: str = "angles",
    extra_weight: float = 1.0,
) -> None:
    """Lightning 4steps LoRA + 追加 LoRA(例: Multiple-angles)を transformer に
    「adapter として」ロードし(fuseしない)、両方を有効化してから
    enable_layerwise_casting(fp8_e4m3fn) でベースのみストレージ圧縮する。

    charsheet の fp8-base-adapters 変種専用(families/qwen_image/edit_angles_bf16.py
    から呼ばれる)。fuse_lightning_and_extra_lora_and_cast_to_fp8() との違いは
    「重みに焼き込まない」点のみ(LoRA計算は毎回 bf16 フル精度の adapter forward で
    行われる。fuse+fp8化で起きるLoRA差分の量子化グリッド丸め込みを避ける狙い)。

    重要(実機で特定したバグの回避、必ずこの順序を守ること): LoRA アダプタは必ず
    fp8 キャストより **先に** ロードすること。PEFT の load_lora_adapter()/
    update_layer() は新規 lora_A/lora_B の nn.Linear をベース層の「その時点の
    dtype」に合わせて作成する。先に enable_layerwise_casting(fp8_e4m3fn) で
    ベースを fp8 化してから LoRA をロードすると、lora_A/lora_B 自体が fp8 の
    nn.Linear として作られてしまい、forward 時に `NotImplementedError:
    "addmm_cuda" not implemented for 'Float8_e4m3fn'` で落ちる(diffusers の
    layerwise-casting hook は「先に登録されていたモジュール」の forward だけを
    差し替える実装のため、後から追加される lora_A/lora_B には hook が適用されず、
    かつ生成時の dtype も fp8 のまま計算されてしまうため)。そのため
    「LoRAロード(bf16のまま)-> fp8キャスト」の順を厳守する。fp8キャスト後も
    base_layer の呼び出しは layerwise-casting の hook で bf16 に復元されるため、
    PEFT の標準 forward(`result = base_layer(x) + lora_B(lora_A(x)) * scaling`)は
    正しく動作する(lora_A/lora_B はこの時点で既に bf16 として存在するため
    fp8化の対象にならない)。

    実機実測(RTX PRO 5000 Blackwell 48GB専有): bf16ロード直後 ~38GB、
    LoRA(lightning+angles)アダプタロード後 ~38.2GB(adapter自体は小さい)、
    fp8 layerwise cast後 ~19-20GB + adapter分。ロード過程のピークは
    fuse系とほぼ同じ(bf16 transformer ~38GB)。
    """
    from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_qwen_lora_to_diffusers

    def _load_converted_sd(local_path, hf_repo, hf_file):
        lora_path = resolve_model_path(local_path, hf_repo, hf_file)
        raw_sd = {}
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                raw_sd[key] = f.get_tensor(key)
        return _convert_non_diffusers_qwen_lora_to_diffusers(raw_sd)

    lightning_sd = _load_converted_sd(lightning_local_path, lightning_hf_repo, lightning_hf_file)
    transformer.load_lora_adapter(lightning_sd, adapter_name="lightning", prefix="transformer")

    extra_sd = _load_converted_sd(extra_local_path, extra_hf_repo, extra_hf_file)
    transformer.load_lora_adapter(extra_sd, adapter_name=extra_adapter_name, prefix="transformer")

    transformer.set_adapters(["lightning", extra_adapter_name], weights=[1.0, extra_weight])

    if torch.cuda.is_available():
        print(
            "[core.loaders] LoRA adapters loaded (lightning+"
            f"{extra_adapter_name}), VRAM={torch.cuda.memory_allocated()/1024**3:.2f}GB"
        )

    transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
