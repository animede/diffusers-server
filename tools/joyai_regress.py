# -*- coding: utf-8 -*-
"""
JoyAI-Image-Edit-Plus 統合可否のスタンドアロン回帰確認スクリプト(STEP 1)。

目的: jdopensource/JoyAI-Image-Edit-Plus-Diffusers (Apache 2.0、複数参照画像編集モデル)を
venv の diffusers(0.40.0.dev0、JoyImageEditPlusPipeline/JoyImageEditPlusTransformer3DModel
が既に実装済みであることを確認済み)でロードし、単一参照編集・複数参照合成の2本を
生成できるか確認する。families/ / core/ / app.py には一切触れない、調査専用の使い捨て
スクリプト(tools/ltx2_regress.py・tools/convert_z_image_bf16.py と同じ位置づけ)。

構成:
  - transformer: 16B, bf16, 約32.5GB
  - text_encoder: Qwen3-VL-8B, 約17.5GB
  - vae: AutoencoderKLWan, 約0.25GB
  - 合計ダウンロード 約50.3GB(HF標準キャッシュ ~/.cache/huggingface/hub に保存される。
    resolve_comfyui_path 等は使わない、そもそも families/ 化する前の調査段階のため)。

呼び出し規約(モデルカードで確認済み): `pipe(images=[...], prompt=..., height=..., width=...)`。
images は単体画像でも必ずリストで渡す(CLAUDE.md 2番、QwenImageEditPlusPipelineと同じ規約)。

Phase 1: 96GBデフォルト(オフロード無し)でロード+生成2本、VRAM/RAM実測。
Phase 2: 48GB相当を疑似再現(ダミーVRAM ballastで空きを制限)し、
  (A) text_encoder CPU退避、(B) transformer fp8 layerwise casting、の2方式を試す。

安全策: transformer(32.5GB)を丸ごとCPUへスワップする実装は絶対に行わない
(CLAUDE.md 33番の事故の教訓。ホストRAM 62GB環境でのスワップ暴走→OOMキラー実績あり)。
text_encoder(17.5GB)のみ、ロード直後の片道スワップ+pipe()直前の復帰、という
narrow windowでのみ許容する。

使い方:
  venv/bin/python tools/joyai_regress.py --stage phase1
  venv/bin/python tools/joyai_regress.py --stage phase2a   # TE CPU offload, 48GB相当
  venv/bin/python tools/joyai_regress.py --stage phase2b   # fp8 layerwise casting, 48GB相当
  venv/bin/python tools/joyai_regress.py --stage all       # phase1 -> phase2a -> phase2b 順に実行
"""
import argparse
import gc
import hashlib
import os
import subprocess
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_ID = "jdopensource/JoyAI-Image-Edit-Plus-Diffusers"
CACHE_DIR_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--jdopensource--JoyAI-Image-Edit-Plus-Diffusers"
)

PORTRAIT_PATH = "/home/animede/ComfyUI/output/RemoveBG-result_00003_.png"
APPLE_PATH = "/home/animede/diffusers-server/outputs/t2i_20260721_134949_a49636ab.png"

OUTPUT_DIR = "/home/animede/diffusers-server/outputs"

SEED = 12345
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 4.0

EDIT_PROMPT = "Change the color of her clothes to red, keep everything else the same."
COMPOSE_PROMPT = "The woman is holding the apple in her hand, smiling."

# 48GB相当を疑似するためのballastサイズ。96GB機で空きを ~44-46GB程度まで削る。
BALLAST_TARGET_FREE_GB = 45.0


def log(msg):
    print(f"[joyai_regress] {msg}", flush=True)


def vram_alloc_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024**3


def vram_peak_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def nvidia_smi_snapshot(tag):
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        log(f"nvidia-smi [{tag}]: {out}")
    except Exception as e:
        log(f"nvidia-smi failed: {e}")


def free_h_snapshot(tag):
    try:
        out = subprocess.run(["free", "-h"], capture_output=True, text=True, timeout=10).stdout
        log(f"free -h [{tag}]:\n{out}")
        return out
    except Exception as e:
        log(f"free -h failed: {e}")
        return ""


def swap_used_gb():
    """free -h の Swap used 列(GiB)を返す。増加監視用。"""
    try:
        out = subprocess.run(["free", "-b"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if line.startswith("Swap:"):
                parts = line.split()
                used_bytes = int(parts[2])
                return used_bytes / 1024**3
    except Exception as e:
        log(f"swap_used_gb failed: {e}")
    return -1.0


def assert_swap_not_growing(baseline_swap_gb, tag, tolerance_gb=0.5):
    current = swap_used_gb()
    if current < 0:
        return
    delta = current - baseline_swap_gb
    if delta > tolerance_gb:
        log(
            f"!!! SWAP GROWTH DETECTED at [{tag}]: baseline={baseline_swap_gb:.2f}GB "
            f"current={current:.2f}GB delta=+{delta:.2f}GB — ABORTING per safety policy !!!"
        )
        raise RuntimeError(
            f"Swap growth of {delta:.2f}GB detected at stage '{tag}'. Aborting to avoid "
            f"OOM-killer risk (see CLAUDE.md incident history)."
        )
    log(f"swap check [{tag}]: OK (baseline={baseline_swap_gb:.2f}GB current={current:.2f}GB)")


def poll_download_progress(interval_s=20, max_wait_s=3600):
    """ダウンロード中、キャッシュディレクトリのサイズを定期的にログする(進捗の目視確認用)。
    このポーリングはメインスレッドではなく、from_pretrained() 呼び出し前に別プロセスとして
    起動できないため(1プロセススクリプトのため)、実際には from_pretrained 実行中は
    ブロックされる。よってこの関数は使わず、from_pretrained の前後でサイズを記録するのみ
    にとどめる(シンプルな one-shot ログ)。
    """
    pass


def log_cache_size(tag):
    if os.path.isdir(CACHE_DIR_GLOB):
        try:
            out = subprocess.run(["du", "-sh", CACHE_DIR_GLOB], capture_output=True, text=True, timeout=60).stdout.strip()
            log(f"cache size [{tag}]: {out}")
        except Exception as e:
            log(f"du failed: {e}")
    else:
        log(f"cache size [{tag}]: not yet created")


def save_output_image(img: Image.Image, prefix: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    h = hashlib.sha1(f"{prefix}{ts}{time.time()}".encode()).hexdigest()[:8]
    filename = f"{prefix}_{ts}_{h}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    img.save(path)
    log(f"saved {path}")
    return path


# ---------------------------------------------------------------------------
# Ballast (48GB-equivalent VRAM simulation)
# ---------------------------------------------------------------------------

_ballast_tensor = None


def allocate_ballast(target_free_gb=BALLAST_TARGET_FREE_GB):
    """空きVRAMが target_free_gb 程度になるまでダミーテンソルを確保する。"""
    global _ballast_tensor
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / 1024**3
    total_gb = total_bytes / 1024**3
    to_consume_gb = free_gb - target_free_gb
    if to_consume_gb <= 0:
        log(f"ballast: free={free_gb:.2f}GB already <= target={target_free_gb:.2f}GB, skipping")
        return
    n_elements = int(to_consume_gb * 1024**3 / 4)  # float32 = 4 bytes
    log(f"ballast: consuming ~{to_consume_gb:.2f}GB (free={free_gb:.2f}GB, target={target_free_gb:.2f}GB, total={total_gb:.2f}GB)")
    _ballast_tensor = torch.empty(n_elements, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    nvidia_smi_snapshot("after ballast allocation")


def free_ballast():
    global _ballast_tensor
    if _ballast_tensor is not None:
        del _ballast_tensor
        _ballast_tensor = None
        gc.collect()
        torch.cuda.empty_cache()
        log("ballast freed")
        nvidia_smi_snapshot("after ballast free")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_pipeline():
    """RAM安全なロード(2026-07-21修正)。

    当初実装の `JoyImageEditPlusPipeline.from_pretrained(...)` -> `.to("cuda")` は、
    transformer 32.5GB + text_encoder 17.5GB = 約50GB を一度ホストRAM(このマシンは
    62GB、実効空き ~38GB)に全展開してから GPU へ移すため、スワップ暴走で
    プロセスが死ぬ(実際に phase1 初回実行が silent death した死因とみられる。
    CLAUDE.md 33番のRAM安全方針違反)。

    修正: 大型コンポーネント(transformer / text_encoder)は `device_map="cuda"` で
    シャード単位に GPU へ直接ロードし(ホストRAMのピークはシャード1枚 ~5GB)、
    パイプラインには構築済みコンポーネントとして渡す。vae 等の小物のみ
    `pipe.to("cuda")` で移動する。
    """
    from diffusers import JoyImageEditPlusPipeline, JoyImageEditPlusTransformer3DModel
    from transformers import Qwen3VLForConditionalGeneration

    log(f"loading {MODEL_ID} (torch_dtype=bfloat16, component-wise device_map='cuda') ...")
    log_cache_size("before load")
    t0 = time.time()
    transformer = JoyImageEditPlusTransformer3DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cuda"
    )
    log("transformer loaded (device_map=cuda)")
    free_h_snapshot("after transformer load")
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map="cuda"
    )
    log("text_encoder loaded (device_map=cuda)")
    free_h_snapshot("after text_encoder load")
    pipe = JoyImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
    )
    t_download_and_construct = time.time() - t0
    log(f"component-wise from_pretrained() done in {t_download_and_construct:.1f}s")
    log_cache_size("after from_pretrained")

    t1 = time.time()
    pipe.to("cuda")  # vae等の小物のみ移動(transformer/TEは既にcuda)
    torch.cuda.synchronize()
    t_to_cuda = time.time() - t1
    log(f"pipe.to('cuda') done in {t_to_cuda:.1f}s")

    total_load_time = t_download_and_construct + t_to_cuda
    log(f"TOTAL load time (download+construct+to_cuda): {total_load_time:.1f}s")
    nvidia_smi_snapshot("after full load")
    free_h_snapshot("after full load")
    return pipe, total_load_time


def load_reference_images():
    portrait = Image.open(PORTRAIT_PATH).convert("RGB")
    apple = Image.open(APPLE_PATH).convert("RGB")
    log(f"portrait: {PORTRAIT_PATH} size={portrait.size}")
    log(f"apple: {APPLE_PATH} size={apple.size}")
    return portrait, apple


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def run_single_ref_edit(pipe, portrait, prefix="joyai_edit"):
    torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    t0 = time.time()
    result = pipe(
        images=[portrait],
        prompt=EDIT_PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=gen,
    )
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak = vram_peak_gb()
    log(f"[single-ref edit] elapsed={elapsed:.1f}s peak_vram={peak:.2f}GB")
    path = save_output_image(result.images[0], prefix)
    return {"elapsed_s": elapsed, "peak_vram_gb": peak, "path": path}


def run_multi_ref_compose(pipe, portrait, apple, prefix="joyai_compose"):
    torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    t0 = time.time()
    result = pipe(
        images=[portrait, apple],
        prompt=COMPOSE_PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=gen,
    )
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak = vram_peak_gb()
    log(f"[multi-ref compose] elapsed={elapsed:.1f}s peak_vram={peak:.2f}GB")
    path = save_output_image(result.images[0], prefix)
    return {"elapsed_s": elapsed, "peak_vram_gb": peak, "path": path}


# ---------------------------------------------------------------------------
# Phase 1: 96GB baseline, no offload
# ---------------------------------------------------------------------------

def phase1():
    log("=== PHASE 1: 96GB baseline (no offload) ===")
    baseline_swap = swap_used_gb()
    free_h_snapshot("phase1 start")
    nvidia_smi_snapshot("phase1 start")

    torch.cuda.reset_peak_memory_stats()
    pipe, load_time = load_pipeline()
    assert_swap_not_growing(baseline_swap, "after load")
    free_h_snapshot("phase1 after load")

    portrait, apple = load_reference_images()

    r1 = run_single_ref_edit(pipe, portrait)
    assert_swap_not_growing(baseline_swap, "after test1")
    free_h_snapshot("phase1 after test1")

    r2 = run_multi_ref_compose(pipe, portrait, apple)
    assert_swap_not_growing(baseline_swap, "after test2")
    free_h_snapshot("phase1 after test2")

    cumulative_peak = vram_peak_gb()

    log("=== PHASE 1 SUMMARY ===")
    log(f"load_time_s={load_time:.1f}")
    log(f"test1 (single-ref edit): elapsed={r1['elapsed_s']:.1f}s peak_vram={r1['peak_vram_gb']:.2f}GB path={r1['path']}")
    log(f"test2 (multi-ref compose): elapsed={r2['elapsed_s']:.1f}s peak_vram={r2['peak_vram_gb']:.2f}GB path={r2['path']}")
    log(f"cumulative peak_vram={cumulative_peak:.2f}GB")

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    nvidia_smi_snapshot("phase1 after cleanup")

    return {"load_time_s": load_time, "test1": r1, "test2": r2, "cumulative_peak_vram_gb": cumulative_peak}


# ---------------------------------------------------------------------------
# Phase 2A: 48GB-equivalent, text_encoder CPU offload
# ---------------------------------------------------------------------------

def install_te_encode_wrapper(pipe, baseline_swap):
    """Approach A の正しい実装: encode_prompt_multiple_images をラップし、
    「エンコード呼び出しの間だけ text_encoder を GPU へ、終了後に CPU へ戻す」。
    これにより transformer が動く denoise 中は TE(17.5GB)が GPU に居ないため
    ピークVRAMから TE 分が外れる。vae は cuda 常駐のため _execution_device は cuda に
    解決される(vae が components 内で最初の nn.Module のため、TE が cpu でも
    self.device=cuda。CLAUDE.md 23番の device-mismatch は _execution_device が cpu に
    転ぶことが原因なので、vae を cuda に残す限り発生しない)。
    cfg>1 では正・負で2回 encode されるが、その都度この短い窓で up/down する。
    transformer は一切 CPU へ動かさない(CLAUDE.md 33番の禁止事項を厳守)。"""
    import types
    orig = pipe.encode_prompt_multiple_images.__func__  # unbound function

    def wrapped(self, prompt, device=None, images=None, max_sequence_length=None):
        self.text_encoder.to("cuda")
        torch.cuda.synchronize()
        try:
            return orig(self, prompt, device=torch.device("cuda"),
                        images=images, max_sequence_length=max_sequence_length)
        finally:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()
            assert_swap_not_growing(baseline_swap, "phase2a te-swap window")

    pipe.encode_prompt_multiple_images = types.MethodType(wrapped, pipe)
    log("installed encode-time text_encoder GPU<->CPU swap wrapper")


def phase2a():
    log("=== PHASE 2A: 48GB-equivalent, text_encoder CPU offload ===")
    baseline_swap = swap_used_gb()
    free_h_snapshot("phase2a start")

    allocate_ballast()
    nvidia_smi_snapshot("phase2a after ballast")

    torch.cuda.reset_peak_memory_stats()
    pipe, load_time = load_pipeline()
    assert_swap_not_growing(baseline_swap, "phase2a after load")
    nvidia_smi_snapshot("phase2a after load (with ballast)")

    # Move text_encoder to CPU right after load. It will be brought up to GPU only
    # during the (wrapped) encode call and pushed back to CPU for the denoise loop,
    # so the transformer-heavy denoise never has TE resident on GPU.
    log("moving text_encoder to CPU (resident on CPU during denoise)...")
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    nvidia_smi_snapshot("phase2a after text_encoder->cpu")
    assert_swap_not_growing(baseline_swap, "phase2a after te->cpu")

    install_te_encode_wrapper(pipe, baseline_swap)

    portrait, apple = load_reference_images()

    try:
        r1 = run_single_ref_edit(pipe, portrait, prefix="joyai_48gb_teoffload_edit")
    except RuntimeError as e:
        log(f"phase2a test1 FAILED: {type(e).__name__}: {e}")
        raise
    assert_swap_not_growing(baseline_swap, "phase2a after test1")

    try:
        r2 = run_multi_ref_compose(pipe, portrait, apple, prefix="joyai_48gb_teoffload_compose")
    except RuntimeError as e:
        log(f"phase2a test2 FAILED: {type(e).__name__}: {e}")
        raise
    assert_swap_not_growing(baseline_swap, "phase2a after test2")

    log("=== PHASE 2A SUMMARY ===")
    log(f"load_time_s={load_time:.1f}")
    log(f"test1: elapsed={r1['elapsed_s']:.1f}s peak_vram={r1['peak_vram_gb']:.2f}GB path={r1['path']}")
    log(f"test2: elapsed={r2['elapsed_s']:.1f}s peak_vram={r2['peak_vram_gb']:.2f}GB path={r2['path']}")

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    free_ballast()
    nvidia_smi_snapshot("phase2a after cleanup")

    return {"load_time_s": load_time, "test1": r1, "test2": r2}


# ---------------------------------------------------------------------------
# Phase 2B: 48GB-equivalent, fp8 layerwise casting on transformer
# ---------------------------------------------------------------------------

def phase2b():
    log("=== PHASE 2B: 48GB-equivalent, fp8 layerwise casting (transformer) ===")
    baseline_swap = swap_used_gb()
    free_h_snapshot("phase2b start")

    allocate_ballast()
    nvidia_smi_snapshot("phase2b after ballast")

    torch.cuda.reset_peak_memory_stats()
    pipe, load_time = load_pipeline()
    assert_swap_not_growing(baseline_swap, "phase2b after load")
    nvidia_smi_snapshot("phase2b after load (with ballast)")

    log("applying enable_layerwise_casting(storage=fp8_e4m3fn, compute=bf16) to transformer...")
    t_cast0 = time.time()
    pipe.transformer.enable_layerwise_casting(
        storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16
    )
    torch.cuda.empty_cache()
    t_cast = time.time() - t_cast0
    log(f"layerwise casting applied in {t_cast:.1f}s")
    nvidia_smi_snapshot("phase2b after fp8 layerwise casting")
    assert_swap_not_growing(baseline_swap, "phase2b after fp8 cast")

    portrait, apple = load_reference_images()

    try:
        r1 = run_single_ref_edit(pipe, portrait, prefix="joyai_48gb_fp8_edit")
    except RuntimeError as e:
        log(f"phase2b test1 FAILED: {type(e).__name__}: {e}")
        raise
    assert_swap_not_growing(baseline_swap, "phase2b after test1")

    try:
        r2 = run_multi_ref_compose(pipe, portrait, apple, prefix="joyai_48gb_fp8_compose")
    except RuntimeError as e:
        log(f"phase2b test2 FAILED: {type(e).__name__}: {e}")
        raise
    assert_swap_not_growing(baseline_swap, "phase2b after test2")

    log("=== PHASE 2B SUMMARY ===")
    log(f"load_time_s={load_time:.1f} fp8_cast_time_s={t_cast:.1f}")
    log(f"test1: elapsed={r1['elapsed_s']:.1f}s peak_vram={r1['peak_vram_gb']:.2f}GB path={r1['path']}")
    log(f"test2: elapsed={r2['elapsed_s']:.1f}s peak_vram={r2['peak_vram_gb']:.2f}GB path={r2['path']}")

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    free_ballast()
    nvidia_smi_snapshot("phase2b after cleanup")

    return {"load_time_s": load_time, "fp8_cast_time_s": t_cast, "test1": r1, "test2": r2}


LOCKFILE = "/tmp/claude-1000/-home-animede-diffusers-server/004ec3eb-8a16-41bb-90bc-6905fddac044/scratchpad/joyai_regress.lock"


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def acquire_lock():
    """単一実行を保証する。既存ロックの PID が生きていれば即 exit(0)。"""
    os.makedirs(os.path.dirname(LOCKFILE), exist_ok=True)
    if os.path.exists(LOCKFILE):
        try:
            with open(LOCKFILE) as f:
                other = int(f.read().strip())
            if other != os.getpid() and _pid_alive(other):
                log(f"another joyai_regress run is active (PID {other}); exiting to avoid "
                    f"concurrent 50GB load / OOM. This run does nothing.")
                sys.exit(0)
        except (ValueError, OSError):
            pass
    with open(LOCKFILE, "w") as f:
        f.write(str(os.getpid()))
    log(f"acquired lock {LOCKFILE} (PID {os.getpid()})")


def release_lock():
    try:
        if os.path.exists(LOCKFILE):
            with open(LOCKFILE) as f:
                if int(f.read().strip()) == os.getpid():
                    os.remove(LOCKFILE)
    except (ValueError, OSError):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["phase1", "phase2a", "phase2b", "all"], default="phase1",
        help="which stage to run",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        log("CUDA not available, aborting.")
        sys.exit(1)

    import atexit
    acquire_lock()
    atexit.register(release_lock)

    results = {}
    try:
        if args.stage in ("phase1", "all"):
            results["phase1"] = phase1()
        if args.stage in ("phase2a", "all"):
            results["phase2a"] = phase2a()
        if args.stage in ("phase2b", "all"):
            results["phase2b"] = phase2b()
    finally:
        free_ballast()
        gc.collect()
        torch.cuda.empty_cache()
        nvidia_smi_snapshot("final (script exit)")
        free_h_snapshot("final (script exit)")

    log(f"ALL RESULTS: {results}")
