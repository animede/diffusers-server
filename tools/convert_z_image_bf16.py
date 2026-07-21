# -*- coding: utf-8 -*-
"""
Z-Image-Turbo の transformer を HF Hub 配布の fp32(3シャード, 約24.6GB)から
bf16 単一ファイル(約12.3GB)へローカル変換するスクリプト。

背景(diffusers-server タスク、2026-07-19実施):
  Tongyi-MAI/Z-Image-Turbo の transformer は Hub 上に fp32 でしか配布されていない
  (families/z_image/pipeline.py は from_pretrained(torch_dtype=torch.bfloat16) で
  ロード時にキャストしているだけで、ダウンロード自体は毎回 fp32 24.6GB を要する)。
  実行時は元々 bf16 キャストして使っているため、事前に bf16 化して保存しておけば
  品質・VRAM使用量を変えずにディスク使用量を半減できる(24.6GB -> 12.3GB)。

方式: 1テンソルずつストリーミングで読み(safe_open は mmap ベースなのでシャード全体を
RAM に展開しない)、bf16 にキャストして辞書に積み、最後に save_file() で単一ファイルへ
書き出す。Z-Image transformer の最大テンソルは約157MB(context_refiner.*.feed_forward.w1.weight)
なので、fp32一時テンソルによるRAMスパイクは無視できる。bf16辞書自体(最終的に約12.3GB)は
最後まで保持するが、fp32シャードの合計(24.6GB)は同時に保持しない(safe_open は
get_tensor() の呼び出し時点で該当テンソル分だけロードする実装のため)。

保存先: ComfyUI 優先パターン(core.resolve.resolve_comfyui_path)に合わせ、
$DS_COMFYUI_DIR/models/diffusion_models/z_image_turbo_bf16.safetensors に単一ファイルとして
保存する(他ファミリーの transformer 単一ファイル配置と同じ規約。families/qwen_image/paths.py
の *_TRANSFORMER_PATH と同型)。config.json 等の非テンソルファイルは変換不要
(HFキャッシュのまま使う、pipeline.py 側は load_transformer_from_config() で
Tongyi-MAI/Z-Image-Turco の config を Hub から取得しつつ重みだけこのファイルから読む)。

使い方:
  venv/bin/python tools/convert_z_image_bf16.py            # 変換のみ(検証込み)
  venv/bin/python tools/convert_z_image_bf16.py --verify-only  # 既存ファイルの検証のみ
  venv/bin/python tools/convert_z_image_bf16.py --force     # 既存ファイルがあっても再変換

fp32 の再ダウンロードが必要な場合(HFキャッシュから消した後にこのスクリプトを再実行する
場合)は huggingface_hub.snapshot_download が自動的に行う(このスクリプトが呼ぶ
huggingface_hub.hf_hub_download 経由)。
"""
import argparse
import os
import sys
import time

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.resolve import resolve_comfyui_path  # noqa: E402

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
TRANSFORMER_SUBFOLDER = "transformer"
SHARD_FILENAMES = [
    "diffusion_pytorch_model-00001-of-00003.safetensors",
    "diffusion_pytorch_model-00002-of-00003.safetensors",
    "diffusion_pytorch_model-00003-of-00003.safetensors",
]

OUTPUT_PATH = resolve_comfyui_path("diffusion_models", "z_image_turbo_bf16.safetensors")


def _download_fp32_shards() -> list[str]:
    """HFキャッシュに fp32 シャードが無ければダウンロードし、ローカルパスのリストを返す。"""
    paths = []
    for filename in SHARD_FILENAMES:
        subfolder_filename = f"{TRANSFORMER_SUBFOLDER}/{filename}"
        print(f"[convert_z_image_bf16] resolving {MODEL_ID}/{subfolder_filename} ...")
        path = hf_hub_download(repo_id=MODEL_ID, filename=subfolder_filename)
        paths.append(path)
    return paths


def convert(force: bool = False) -> str:
    if os.path.exists(OUTPUT_PATH) and not force:
        print(f"[convert_z_image_bf16] {OUTPUT_PATH} は既に存在します(--force で再変換)。")
        return OUTPUT_PATH

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    shard_paths = _download_fp32_shards()

    t0 = time.time()
    bf16_tensors: dict[str, torch.Tensor] = {}
    total_keys = 0

    for shard_path in shard_paths:
        print(f"[convert_z_image_bf16] streaming {shard_path} ...")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if tensor.dtype != torch.float32:
                    print(
                        f"[convert_z_image_bf16] warning: {key} has unexpected dtype "
                        f"{tensor.dtype} (expected float32); casting to bf16 anyway"
                    )
                bf16_tensors[key] = tensor.to(torch.bfloat16).contiguous()
                total_keys += 1
                del tensor

    print(f"[convert_z_image_bf16] streamed {total_keys} tensors in {time.time() - t0:.1f}s, saving...")
    t1 = time.time()
    save_file(bf16_tensors, OUTPUT_PATH, metadata={"format": "pt", "source": MODEL_ID, "converted_dtype": "bf16"})
    print(f"[convert_z_image_bf16] saved {OUTPUT_PATH} in {time.time() - t1:.1f}s")

    size_gb = os.path.getsize(OUTPUT_PATH) / (1024 ** 3)
    print(f"[convert_z_image_bf16] output size: {size_gb:.2f} GB")

    del bf16_tensors
    return OUTPUT_PATH


def verify(force_redownload_fp32: bool = False) -> bool:
    """bf16版が fp32版と「キー集合が完全一致・shapeが一致・値が bf16丸めのみ」であることを検証する。

    fp32 の HFキャッシュが既に削除されている場合は force_redownload_fp32=True で
    再ダウンロードして比較する(通常運用では削除前に検証するため不要)。
    """
    if not os.path.exists(OUTPUT_PATH):
        raise FileNotFoundError(f"{OUTPUT_PATH} が存在しません。先に convert() を実行してください。")

    shard_paths = _download_fp32_shards()

    print(f"[convert_z_image_bf16] verifying {OUTPUT_PATH} against fp32 source...")
    expected_keys = set()
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    checked = 0

    with safe_open(OUTPUT_PATH, framework="pt", device="cpu") as bf16_f:
        bf16_keys = set(bf16_f.keys())

        for shard_path in shard_paths:
            with safe_open(shard_path, framework="pt", device="cpu") as fp32_f:
                for key in fp32_f.keys():
                    expected_keys.add(key)
                    fp32_tensor = fp32_f.get_tensor(key)
                    if key not in bf16_keys:
                        raise AssertionError(f"key missing in bf16 file: {key}")
                    bf16_tensor = bf16_f.get_tensor(key)
                    if bf16_tensor.dtype != torch.bfloat16:
                        raise AssertionError(f"{key}: expected bf16, got {bf16_tensor.dtype}")
                    if tuple(bf16_tensor.shape) != tuple(fp32_tensor.shape):
                        raise AssertionError(
                            f"{key}: shape mismatch fp32={tuple(fp32_tensor.shape)} "
                            f"bf16={tuple(bf16_tensor.shape)}"
                        )
                    # bf16丸めのみであることを確認: fp32 -> bf16 -> fp32 の往復と一致するはず
                    roundtrip = fp32_tensor.to(torch.bfloat16).to(torch.float32)
                    actual = bf16_tensor.to(torch.float32)
                    diff = (roundtrip - actual).abs()
                    if diff.numel() > 0:
                        d = diff.max().item()
                        max_abs_diff = max(max_abs_diff, d)
                    checked += 1
                    del fp32_tensor, bf16_tensor, roundtrip, actual

    if expected_keys != bf16_keys:
        missing = expected_keys - bf16_keys
        unexpected = bf16_keys - expected_keys
        raise AssertionError(f"key set mismatch: missing={len(missing)} unexpected={len(unexpected)}")

    print(
        f"[convert_z_image_bf16] verified {checked} tensors, key sets match, "
        f"max abs diff from expected bf16 rounding = {max_abs_diff} (should be 0.0)"
    )
    if max_abs_diff != 0.0:
        raise AssertionError("bf16 file does not match exact bf16 rounding of fp32 source")
    print("[convert_z_image_bf16] OK: bf16 file is a lossless bf16 rounding of the fp32 source.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="既存の bf16 ファイルがあっても再変換する")
    parser.add_argument("--verify-only", action="store_true", help="変換を行わず、既存ファイルの検証だけ行う")
    parser.add_argument("--skip-verify", action="store_true", help="変換後の検証をスキップする(非推奨)")
    args = parser.parse_args()

    if args.verify_only:
        verify()
    else:
        convert(force=args.force)
        if not args.skip_verify:
            verify()
