# -*- coding: utf-8 -*-
"""
JoyImageEditPlusPipeline のロード + sm_120 パッチ化Conv3d対策 + text_encoder CPU退避。

抽出元: tools/joyai_regress.py の load_pipeline()(RAM安全ロード)+
scratchpad/joyai_probe7.py(PatchifyLinear、CLAUDE.md 46番)。

RAM安全ロード(CLAUDE.md 33番の丸ごとCPUスワップ禁止を厳守):
  transformer(32.5GB)/ text_encoder(17.5GB)を `device_map="cuda"` でコンポーネント単位に
  直接 GPU へロードする。`JoyImageEditPlusPipeline.from_pretrained(...).to("cuda")` のように
  パイプライン全体をまずホストRAMに展開してから GPU へ移す実装は、62GB RAM 環境で
  transformer+TE合計50GB超を一時的にRAM上に載せてしまいスワップ暴走のリスクがある
  (tools/joyai_regress.py docstring参照、実機で silent death を観測した経緯あり)。

sm_120 パッチ化Conv3d対策(CLAUDE.md 46番): kernel_size==stride の Conv3d
(JoyAI transformer の img_in、Qwen3-VL vision の patch_embed.proj)は sm_120 で
cuDNN が病的に遅いカーネルを選ぶ(1step 65秒等)。数学的に等価な reshape+linear
モジュール(PatchifyLinear)へランタイム置換することで解消する(1step 0.45秒、146倍)。
このパッチは venv の diffusers/transformers 本体には一切手を入れず、ロード直後に
モジュール差し替えのみで適用する。

text_encoder CPU退避: 48GB専有では transformer(~32.5GB)+ text_encoder(~17.5GB)の
常駐だけで ~50GB となり収まらないため、エンコード時だけ transformer <-> text_encoder を
相互排他でGPU/CPU入れ替える(families/qwen_image の Edit系と同じ「_execution_device
解決タイミング」の罠(CLAUDE.md 23番)に注意。JoyImageEditPlusPipeline も
`__call__()` 内で `device = self._execution_device` をエンコードより前に解決する
一枚岩の実装だが、`_execution_device` は「アルファベット順で最初に見つかる
nn.Module」(このパイプラインでは vae ではなく text_encoder)の現在地を返すため、
GPU復帰は pipe(...) 呼び出し**直前**(prepare_for_encode())に行う必要がある。
詳細は enable_text_encoder_cpu_offload() のdocstring・generate.py 参照)。
"""
import time

import torch

from families.joyai import state
from families.joyai.runtime import JoyAIRuntimeConfig

MODEL_ID = "jdopensource/JoyAI-Image-Edit-Plus-Diffusers"

DEFAULT_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 4.0
MAX_REF_IMAGES = 3


class PatchifyLinear(torch.nn.Module):
    """kernel_size == stride の Conv3d と数値等価な reshape+linear(sm_120 conv3d回避)。

    抽出元: scratchpad/joyai_probe7.py(CLAUDE.md 46番で実証済み)。`weight` プロパティは
    Qwen3-VL の patch_embed.forward() が `self.proj.weight.dtype` を参照するために必要
    (probe7 の初回実装はこのエイリアスを持たず AttributeError で失敗した。probe7b で
    修正・実証済み)。
    """

    def __init__(self, conv: torch.nn.Conv3d):
        super().__init__()
        self.out_channels = conv.out_channels
        self.register_buffer("W", conv.weight.reshape(conv.out_channels, -1))
        self.register_buffer("b", conv.bias if conv.bias is not None else None)

    @property
    def weight(self):
        # Qwen3VL patch_embed が proj.weight.dtype を参照するためのエイリアス。
        return self.W

    def forward(self, x):
        y = torch.nn.functional.linear(x.flatten(1), self.W, self.b)
        return y.reshape(y.shape[0], self.out_channels, 1, 1, 1)


def apply_patchify_linear_patch(pipe) -> None:
    """JoyAI transformer の img_in と Qwen3-VL vision の patch_embed.proj を
    PatchifyLinear へランタイム置換する(冪等: 既に置換済みなら何もしない)。
    """
    if isinstance(pipe.transformer.img_in, PatchifyLinear):
        return  # 既に適用済み

    device = next(pipe.transformer.parameters()).device
    pipe.transformer.img_in = PatchifyLinear(pipe.transformer.img_in).to(device)

    vt = pipe.text_encoder.model.visual
    te_device = next(vt.patch_embed.proj.parameters()).device
    vt.patch_embed.proj = PatchifyLinear(vt.patch_embed.proj).to(te_device)
    print("[families.joyai] PatchifyLinear patch applied (transformer.img_in + Qwen3-VL patch_embed.proj)")


def _load_pipeline_locked() -> None:
    """state.lock を呼び出し側が保持している前提でロードする(冪等)。

    抽出元: tools/joyai_regress.py load_pipeline()(RAM安全版)。

    48GB対応の罠(実機で発見・修正): tools/joyai_regress.py 通りに transformer と
    text_encoder を両方 `device_map="cuda"` でロードすると、その時点で
    transformer(~32.5GB)+ text_encoder(~17.5GB)が**ロード完了直後から同時に GPU 常駐**
    してしまう。text_encoder CPU退避(enable_text_encoder_cpu_offload)は
    encode_prompt_multiple_images 呼び出しの前後でしか効かないため、ロード直後の
    この瞬間には間に合わない。48GB専有(空き~43-46GB)ではロード自体がここで
    CUDA OOM することを実機で確認した。対策: te_offload が有効な場合、
    text_encoder は最初から `device_map="cuda"` を使わず CPU 上にロードし、pipe構築後に
    transformer を先に確保した状態を保ったまま text_encoder を CPU に残す
    (transformer を丸ごとCPUスワップするのではなく、text_encoder 側を最初から
    GPUに載せない設計のため CLAUDE.md 33番の禁止パターンには当たらない)。
    """
    ps = state.pipeline_state
    if ps["loaded"]:
        return

    from diffusers import JoyImageEditPlusPipeline, JoyImageEditPlusTransformer3DModel
    from transformers import Qwen3VLForConditionalGeneration

    config = state.get_runtime_config()
    from families.joyai.runtime import should_offload_te

    te_offload = should_offload_te(config)
    te_device_map = "cpu" if te_offload else "cuda"
    print(
        f"[families.joyai] loading {MODEL_ID} (transformer device_map='cuda', "
        f"text_encoder device_map={te_device_map!r}): {config}"
    )
    t0 = time.time()

    transformer = JoyImageEditPlusTransformer3DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cuda"
    )
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map=te_device_map
    )
    pipe = JoyImageEditPlusPipeline.from_pretrained(
        MODEL_ID,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch.bfloat16,
    )
    # vae 等の小物のみ GPU へ移動する(transformer は既に cuda、text_encoder は
    # te_offload 有効時 CPU のまま残すため pipe.to("cuda") を使わない = 誤って
    # text_encoder まで GPU へ引っ張られることを避ける)。
    pipe.vae.to("cuda")
    torch.cuda.synchronize()

    apply_patchify_linear_patch(pipe)

    ps["pipe"] = pipe
    ps["loaded"] = True
    ps["load_time_s"] = time.time() - t0
    ps["te_offload"] = config.te_offload
    ps["patched"] = True
    print(f"[families.joyai] loaded in {ps['load_time_s']:.1f}s (te_offload={config.te_offload})")


def get_pipeline():
    """JoyImageEditPlusPipeline を返す(冪等ロード)。"""
    if not state.pipeline_state["loaded"]:
        with state.lock:
            if not state.pipeline_state["loaded"]:
                _load_pipeline_locked()
    return state.pipeline_state["pipe"]


# ============================================================================
# text_encoder CPU退避(JoyAI固有: encode_prompt_multiple_images をラップする)
# ============================================================================
# families/qwen_image 系の Edit(core.optimize.enable_text_encoder_cpu_offload)は
# `pipe.encode_prompt` をラップする実装だが、JoyAI のエンコード関数名は
# `encode_prompt_multiple_images`(pipeline_joyimage_edit_plus.py)のため、専用の
# ラップ関数をここに実装する。罠は qwen_image (CLAUDE.md 23番)と同種だが、
# 「_execution_device 解決時に最初に見つかる nn.Module」が vae ではなく
# text_encoder である点が異なる(下記 enable_text_encoder_cpu_offload() の
# docstring 参照)。GPU復帰(および48GB対応のtransformer退避)は
# encode_prompt_multiple_images 内ではなく、呼び出し側が pipe(...) を呼ぶ**直前**に
# prepare_for_encode() を呼んで行う(generate.py 参照)。


def enable_text_encoder_cpu_offload(pipe) -> None:
    """pipe.encode_prompt_multiple_images をラップし、呼び出し後に transformer を
    GPU へ戻す・text_encoder を CPU へ退避する(手動 swap 方式)。同一 pipe に複数回
    呼んでも二重ラップしない。

    48GB対応(実機で発見・修正、重要な罠): transformer(~32.5GB)+ text_encoder
    (~17.5GB)は合計~50GBのため、48GB専有(空き~43-46GB)では両方を同時にGPUへ
    置けない。当初「エンコード呼び出しの中で transformer をCPUへ退避 -> TEをGPUへ」
    という実装にしたが、これは **手遅れ**であることを実機で特定した:
    `JoyImageEditPlusPipeline.__call__()` は encode_prompt_multiple_images を
    呼ぶより**前**の行(`device = self._execution_device`)で device を解決し、
    この device を encode_prompt_multiple_images に明示的に渡す。`_execution_device`
    (= `self.device`)は `_get_signature_keys()` が返す `sorted(expected_modules)`
    (**アルファベット順**、コンストラクタの宣言順ではない)の先頭から見つかった
    nn.Module の `.device` を返す実装で、JoyImageEditPlusPipeline のコンポーネント名を
    アルファベット順に並べると "text_encoder" が "scheduler"/"transformer"/"vae" より
    先に来る(processor/schedulerはnn.Moduleではないため実質的にtext_encoderが最初に
    見つかる)。そのため **`pipe(...)` を呼んだ瞬間の text_encoder の位置が
    `_execution_device` を決定する**。text_encoder が CPU に残ったまま `pipe(...)` を
    呼ぶと `device="cpu"` に解決され、processor 出力(input_ids 等)が
    `.to("cpu")` されてしまい、エンコード呼び出しの中で(遅れて)text_encoder を
    GPU へ移しても `embed_tokens.weight`(cuda)と `input_ids`(cpu)の食い違いで
    `RuntimeError: Expected all tensors to be on the same device` になることを
    実機で再現・特定した(qwen_image の23番と同種だが、"最初に見つかるモジュール"が
    vaeではなくtext_encoderである点が異なる)。

    対策: text_encoder の GPU 復帰(および transformer の CPU 退避)は
    **pipe(...) を呼ぶ直前**(prepare_for_encode())に行う。

    重要な罠その2(実機で発見・修正): エンコード完了後の入れ替え(text_encoder->cpu,
    transformer->cuda)を encode_prompt_multiple_images 呼び出し直後の finally で
    行うと、`do_classifier_free_guidance=True`(guidance_scale>1、既定4.0)の場合に
    **2回目(negative prompt)のエンコード呼び出しで失敗する**。`__call__()` は
    `device = self._execution_device` を**1回だけ**冒頭で解決し、その同じ device
    変数を正・負2回のエンコード呼び出しに使い回す。1回目の finally で
    text_encoder を CPU へ戻してしまうと、2回目の呼び出し時点では
    device="cuda"(1回目呼び出し時点の解決結果のまま)なのに実際の
    text_encoder は CPU、という食い違いになり同種のdevice mismatchで失敗する
    ことを実機で再現・特定した。対策: encode_prompt_multiple_images 側での
    入れ替えは行わず、**transformer の forward 直前(forward pre-hook)**で
    「text_encoderをCPUへ・transformerをGPUへ」を保証する設計に変更した。
    denoiseループの最初の transformer(...) 呼び出し時に発火し、以後は
    transformer が既にGPU上にあるため実質的にno-opになる(2回目以降のCFG内
    forwardや複数stepのforwardで毎回チェックはするがtoは呼ばれない)。

    これは CLAUDE.md 33番が禁止する「丸ごとCPUスワップ(transformerを常時CPU常駐させ
    denoise/VAE処理のたびに丸ごと行き来させる)」とは異なる: transformer の退避は
    「エンコード呼び出しという短い窓の間だけ」であり、denoise開始後は一貫してGPUに
    常駐したままである(denoise中の毎ステップスワップは一切行わない、pre-hookは
    デバイスチェックのみで実際のto()は最初の1回しか発生しない)。
    """
    if getattr(pipe, "_ds_joyai_te_offload_enabled", False):
        return
    if not torch.cuda.is_available():
        return

    def _pre_forward_hook(_module, _args, _kwargs):
        # denoiseループの最初の transformer forward 呼び出し時に、
        # text_encoder(エンコード用にGPUへ載せていた)をCPUへ戻し、
        # transformer を確実にGPUへ戻す(既にGPUなら no-op)。
        if next(pipe.text_encoder.parameters()).device.type == "cuda":
            pipe.text_encoder.to("cpu")
            torch.cuda.empty_cache()
        if next(_module.parameters()).device.type != "cuda":
            _module.to("cuda")
            torch.cuda.synchronize()

    handle = pipe.transformer.register_forward_pre_hook(_pre_forward_hook, with_kwargs=True)

    # VAE decode 時の一時ピーク対策(実機で発見: transformer(~32.5GB)が denoise後も
    # GPUに常駐したままだと、decode終盤のconv3d一時ピーク(~5GB)を含めても48GB専有
    # (空き~43-46GB)ではOOMする)。denoise終了後・decode開始前に transformer を
    # CPUへ退避し、decode完了後に戻す(decodeにtransformerは不要なため安全)。
    # transformer.forward の pre-hook(上記)により次の生成の denoise 開始時に
    # 自動的にGPUへ戻るため、明示的な復帰処理は decode 側の finally のみで足りる。
    original_decode = pipe.vae.decode

    def _wrapped_decode(*args, **kwargs):
        if next(pipe.transformer.parameters()).device.type == "cuda":
            pipe.transformer.to("cpu")
            torch.cuda.empty_cache()
        try:
            return original_decode(*args, **kwargs)
        finally:
            pipe.transformer.to("cuda")
            torch.cuda.synchronize()

    pipe.vae.decode = _wrapped_decode

    pipe._ds_joyai_te_offload_enabled = True
    pipe._ds_joyai_te_offload_hook_handle = handle
    pipe._ds_joyai_te_offload_original_decode = original_decode
    print(
        "[families.joyai] text_encoder CPU offload enabled (forward-pre-hook swap, "
        "transformer<->text_encoder mutual exclusion; transformer also evicted during VAE decode)"
    )


def disable_text_encoder_cpu_offload(pipe) -> None:
    """enable_text_encoder_cpu_offload() で登録した forward pre-hook / vae.decode
    ラップを解除し、text_encoder を GPU へ戻す。"""
    if not getattr(pipe, "_ds_joyai_te_offload_enabled", False):
        return
    pipe._ds_joyai_te_offload_hook_handle.remove()
    del pipe._ds_joyai_te_offload_hook_handle
    pipe.vae.decode = pipe._ds_joyai_te_offload_original_decode
    del pipe._ds_joyai_te_offload_original_decode
    pipe._ds_joyai_te_offload_enabled = False
    if torch.cuda.is_available():
        pipe.text_encoder.to("cuda")
        torch.cuda.empty_cache()
    print("[families.joyai] text_encoder CPU offload disabled (text_encoder back on GPU)")


def prepare_for_encode(pipe, te_offload: bool) -> None:
    """pipe(...) を呼ぶ**直前**に呼ぶこと(_execution_device 解決タイミングの罠、
    enable_text_encoder_cpu_offload() のdocstring参照)。

    te_offload 有効時: transformer を CPU へ退避し、text_encoder を GPU へ載せる
    (48GB専有では両方同時にGPUへ置けないため)。te_offload 無効時: 何もしない
    (transformer/text_encoderとも常時GPU常駐のまま、96GB機など大容量VRAM環境向け)。
    """
    if not torch.cuda.is_available():
        return
    if not te_offload:
        return
    if next(pipe.transformer.parameters()).device.type == "cuda":
        pipe.transformer.to("cpu")
        torch.cuda.empty_cache()
    if next(pipe.text_encoder.parameters()).device.type != "cuda":
        pipe.text_encoder.to("cuda")
        torch.cuda.synchronize()
