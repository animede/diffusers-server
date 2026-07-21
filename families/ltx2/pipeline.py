# -*- coding: utf-8 -*-
"""
LTX2Pipeline のロード(全コンポーネントストリーミングロード + none/group オフロード適用)。
I2V パイプラインは base の components から遅延構築する。

latent upsampler(LTX2LatentUpsamplerModel、2026-07-20追加): get_upsampler() が
T2V/I2Vのupscale=1オプション用に初回要求時だけ遅延ロードする独立モデル(GPU常駐固定、
~0.93GBと小さいためgroup offloadの対象にしない)。詳細は families/ltx2/convert.py の
load_upsampler()、families/ltx2/generate.py の _run_upscaled() 参照。

抽出元: tools/ltx2_regress.py の stage_full()(行709-887)。ロジックは実証済みのまま、
オフロードモード分岐(48GB対応の "group")を新規に追加した点が回帰確認スクリプトとの差分。

コンポーネントサイズの内訳(実測・96GB機、bf16デクオンタイズ後):
  transformer(LTX2VideoTransformer3DModel) ~35.4GB  <- "group" モードの対象
  text_encoder(Gemma3ForConditionalGeneration)  ~23GB
  connectors(LTX2TextConnectors)                ~6.3GB
  vae + audio_vae + vocoder                     数GB程度(小)
  合計 常駐: ~67.7GB、denoise時ピーク: ~70.0GB(none モード)

"group" モード(48GB向け本命): transformer だけを block-level group offload
(CLAUDE.md 33番・34番のパターン踏襲、CPU側静的確保・丸ごとスワップ禁止)し、他は
GPU常駐のままにする。GPU使用見込み: 67.7 - 35.4 = 32.3GB 程度(常駐時)。

ロード順序の注意: group モードでは transformer を直接 cuda:0 へ streaming ロードせず
load_device="cpu" でロードしてから enable_group_offload() を呼ぶ(旧charsheet
"bf16-group" と同じ理由: enable_group_offload() の hook が効く前に transformer 本体
(~35GB)だけで空きVRAMを使い切ってOOMするのを防ぐため)。

text_encoder 量子化(DS_LTX2_TE_QUANT、2026-07-20追加、詳細は families/ltx2/runtime.py の
LTX2RuntimeConfig docstring・families/ltx2/convert.py の load_text_encoder() 参照):
  "none"(既定、無変更) / "fp8"(layerwise casting、~23GB->~12GB見込み) /
  "nf4"(QAT版をbitsandbytes NF4でロード、~8GB見込み)。

IC-LoRA(In-Context LoRA、MergeGreen、2026-07-20追加): get_iclora_pipeline() が
LTX2InContextPipeline を base の transformer に MergeGreen LoRA アダプタをロードした
上で構築する。CLAUDE.md 34番(charsheet bf16-group)・17番と同じ「LoRAロード ->
enable_group_offload()」の順序を厳守する必要があるため、_load_base_pipeline_locked()
に load_lora フラグを追加し、transformer 構築直後・group offload 登録前に
transformer.load_lora_adapter() を呼ぶ。base が既に LoRA 無しでロード済みの場合は、
get_iclora_pipeline() が一旦 families/ltx2/lifecycle.unload("all") してから
load_lora=True で再ロードする(state.base_pipeline_state["lora_loaded"] で判定)。
他モード(t2v/i2v/flf等)への切り替え時は pipe.disable_lora() で無効化するのみで
アンロードはしない(再度 iclora が要求されたら pipe.set_adapters() で再有効化する
だけで済み、reload コストは発生しない。families/ltx2/family.py 参照)。
"""
import time

import torch

from core import gpu
from core.optimize import apply_attention_backend, apply_compile, apply_group_offload_to_transformer

from families.ltx2 import convert
from families.ltx2 import state
from families.ltx2.runtime import LTX2RuntimeConfig, resolve_offload_mode

__all__ = [
    "get_base_pipeline",
    "get_i2v_pipeline",
    "get_flf_pipeline",
    "get_iclora_pipeline",
    "get_upsampler",
    "disable_iclora",
]

ICLORA_ADAPTER_NAME = "iclora"


def _check_ram_guard(config: LTX2RuntimeConfig) -> None:
    """group モードロード前のホストRAM空きチェック(CLAUDE.md 34番のパターン踏襲)。

    transformer(bf16 ~35.4GB)を一時的に CPU 上に構築してから group offload に登録する
    ため、これを大きく下回る空きRAMでロードを強行するとスワップ暴走のリスクがある
    (transformer丸ごとのCPU/GPU往復ではなく、あくまでロード時の一時経路。
    enable_group_offload() 登録後は block 単位の転送のみになる)。
    """
    available = gpu.available_ram_gb()
    if available is None:
        return
    if available < config.group_offload_min_ram_gb:
        raise RuntimeError(
            f"[families.ltx2] group offload ロードを中止しました: 空きRAM {available:.1f}GB が "
            f"閾値 {config.group_offload_min_ram_gb:.1f}GB を下回っています "
            f"(transformer bf16 ~35.4GB を CPU に確保する設計のため、RAM不足でのロード強行は "
            f"システムフリーズのリスクがあります。DS_LTX2_GROUP_OFFLOAD_MIN_RAM_GB で閾値を "
            f"調整できますが、実際に空きRAMを確保してから再試行することを推奨します)。"
        )


def _load_base_pipeline_locked(load_lora: bool = False) -> None:
    """state.lock を呼び出し側が保持している前提でロードする(冪等)。

    load_lora=True の場合、MergeGreen IC-LoRA アダプタを transformer 構築直後・
    group offload 登録前にロードする(CLAUDE.md 17番・34番の順序厳守パターン、
    モジュールdocstring参照)。既に(LoRA無しで)ロード済みの場合は何もしない
    (呼び出し側の get_iclora_pipeline() が「lora_loaded=False かつ既にロード済み」を
    検出して事前に unload してから呼び直す設計のため、ここでは単純化のため
    「未ロード時のみ」の冪等ガードのままにする)。
    """
    ps = state.base_pipeline_state
    if ps["loaded"]:
        return

    from diffusers import LTX2Pipeline, FlowMatchEulerDiscreteScheduler
    from transformers import GemmaTokenizerFast
    from huggingface_hub import snapshot_download

    config = state.get_runtime_config()
    device = config.device
    config_repo = "diffusers/LTX-2.3-Diffusers"

    free_gb = gpu.free_vram_gb()
    offload_mode = resolve_offload_mode(config, free_gb)
    print(
        f"[families.ltx2] loading LTX2Pipeline: {config}, resolved offload_mode={offload_mode!r} "
        f"(free_vram={free_gb:.1f}GB)"
    )

    if offload_mode == "group":
        _check_ram_guard(config)

    t0 = time.time()
    breakdown = {}

    # 1. VAE + audio_vae(小さい、GPU常駐固定)
    tv = time.time()
    vae, audio_vae = convert.load_vae_pair(config.ckpt_path, device, config_repo)
    breakdown["vae"] = time.time() - tv

    # 2. transformer + connectors(同一チェックポイント読み取りパスを共有)
    tt = time.time()
    transformer_load_device = "cpu" if offload_mode == "group" else device
    transformer, connectors = convert.load_transformer_and_connectors(
        config.ckpt_path, transformer_load_device, config_repo
    )
    breakdown["transformer_connectors"] = time.time() - tt

    # 2.5 IC-LoRA(MergeGreen、2026-07-20追加、load_lora=True 時のみ): 必ず
    # enable_group_offload() より **前** にロードする(CLAUDE.md 17番・34番の
    # 実証済み順序。逆順だとPEFTが新規作成するlora_A/lora_Bがgroup offload hookの
    # 対象外になり、hookが管理するデバイス配置と食い違って壊れる)。adapter方式
    # (fuseしない)のため、set_adapters()/disable_lora() でいつでも切替可能。
    if load_lora:
        tl = time.time()
        iclora_sd = convert.load_iclora_state_dict(config.iclora_path)
        transformer.load_lora_adapter(iclora_sd, adapter_name=ICLORA_ADAPTER_NAME, prefix="transformer")
        transformer.set_adapters([ICLORA_ADAPTER_NAME], weights=[config.iclora_strength])
        del iclora_sd
        breakdown["iclora_load_s"] = time.time() - tl
        print(
            f"[families.ltx2] IC-LoRA (MergeGreen) loaded onto transformer "
            f"(strength={config.iclora_strength}) in {breakdown['iclora_load_s']:.1f}s"
        )

    if offload_mode == "group":
        apply_group_offload_to_transformer(transformer, num_blocks=config.group_offload_blocks)
        breakdown["group_offload_num_blocks"] = config.group_offload_blocks
        connectors.to(device)
    else:
        connectors.to(device)

    # 3. vocoder(小さい、GPU常駐固定)
    tvoc = time.time()
    vocoder = convert.load_vocoder(config.ckpt_path, device, config_repo)
    breakdown["vocoder"] = time.time() - tvoc

    # 4. text_encoder + tokenizer(Gemma、GPU常駐固定。48GB環境でも text_encoder(~23GB)+
    # connectors(~6.3GB)+ vae類(数GB)+ transformer活性化ピークは "group" で収まる想定)
    tte = time.time()
    vram_before_te = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else None
    text_encoder, gemma_cfg = convert.load_text_encoder(
        config.gemma_path, device, config_repo,
        te_quant=config.te_quant, te_qat_dir=config.te_qat_dir,
    )
    breakdown["text_encoder"] = time.time() - tte
    if torch.cuda.is_available():
        vram_after_te = torch.cuda.memory_allocated() / 1024**3
        breakdown["text_encoder_vram_gb"] = vram_after_te - vram_before_te
        print(
            f"[families.ltx2] text_encoder VRAM contribution (te_quant={config.te_quant!r}): "
            f"{vram_after_te - vram_before_te:.2f}GB"
        )

    tok_dir = snapshot_download(config_repo, allow_patterns=["tokenizer/*"])
    import os
    tokenizer = GemmaTokenizerFast.from_pretrained(os.path.join(tok_dir, "tokenizer"))

    # 5. scheduler(HF config をそのまま使用、shift関連には手を入れない)
    sched_cfg = FlowMatchEulerDiscreteScheduler.load_config(config_repo, subfolder="scheduler")
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(sched_cfg)

    pipe = LTX2Pipeline(
        scheduler=scheduler,
        vae=vae,
        audio_vae=audio_vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        connectors=connectors,
        transformer=transformer,
        vocoder=vocoder,
    )

    apply_attention_backend(pipe.transformer, config.attention_backend, device)
    apply_compile(pipe.transformer, config.compile)

    ps["pipe"] = pipe
    ps["loaded"] = True
    ps["offload_mode"] = offload_mode
    ps["load_time_s"] = time.time() - t0
    ps["load_time_breakdown"] = breakdown
    ps["lora_loaded"] = load_lora
    ps["lora_enabled"] = load_lora  # ロード直後は既定で有効化されている(set_adapters済み)
    if torch.cuda.is_available():
        print(
            f"[families.ltx2] loaded in {ps['load_time_s']:.1f}s (offload={offload_mode}), "
            f"VRAM={torch.cuda.memory_allocated()/1024**3:.2f}GB"
        )


def get_base_pipeline():
    """T2V用の LTX2Pipeline を返す(冪等ロード、LoRAなし)。"""
    if not state.base_pipeline_state["loaded"]:
        with state.lock:
            if not state.base_pipeline_state["loaded"]:
                _load_base_pipeline_locked(load_lora=False)
    pipe = state.base_pipeline_state["pipe"]
    _ensure_iclora_disabled_for_non_iclora_use(pipe)
    return pipe


def _ensure_iclora_disabled_for_non_iclora_use(pipe) -> None:
    """t2v/i2v/flf/ia2v/keyframes/v2a 等、iclora 以外のモードで base を取得する際、
    IC-LoRA がロード済みかつ有効化されたままなら無効化する(タスク要件:「iclora 以外の
    モードへ切り替わる際は pipe.disable_lora() で無効化」)。

    disable_lora() は空リスト set_adapters([]) の KeyError バグ(CLAUDE.md 13番)を
    踏まないよう、必ずこのヘルパー経由で呼ぶ。LoRA未ロードなら何もしない(冪等)。
    """
    ps = state.base_pipeline_state
    if pipe is None or not ps.get("lora_loaded"):
        return
    if ps.get("lora_enabled"):
        pipe.disable_lora()
        ps["lora_enabled"] = False


def get_i2v_pipeline():
    """LTX2ImageToVideoPipeline を返す。base の components から遅延構築
    (参照共有、コピーではないため VRAM 使用量は増えない)。
    """
    from diffusers import LTX2ImageToVideoPipeline

    base = get_base_pipeline()
    ps = state.i2v_pipeline_state
    if not ps["loaded"]:
        with state.lock:
            if not ps["loaded"]:
                ps["pipe"] = LTX2ImageToVideoPipeline(**base.components)
                ps["loaded"] = True
                print("[families.ltx2] LTX2ImageToVideoPipeline derived from base pipeline components")
    return ps["pipe"]


def get_flf_pipeline():
    """LTX2ConditionPipeline を返す(FLF: First-Last-Frame 用)。base の components から
    遅延構築(参照共有、コピーではないため VRAM 使用量は増えない、I2V と同じ設計)。

    LTX2ConditionPipeline は conditions=[LTX2VideoCondition(...), ...] 引数で任意フレーム
    位置への画像条件付けを受け付ける汎用パイプラインで、FLF はその特殊ケース(先頭
    index=0 + 末尾 index=-1 の2条件)にすぎない。ComfyUI ワークフロー
    (video_ltx2_3-22b-flf-bf8.json)の LTXVAddGuide ×2(frame_idx=0/-1, strength=0.7)構成と
    1:1対応する。

    注意: base.components には LTX2Pipeline 固有の "processor" キーが含まれるが、
    LTX2ConditionPipeline.__init__() はこれを受け付けず(代わりに任意の "audio_scheduler"
    を持つ)、そのまま **base.components を渡すと
    TypeError: unexpected keyword argument 'processor' になる(実機で確認済み)。
    LTX2ConditionPipeline が実際に受け付けるキーだけを inspect.signature() で絞り込んで
    渡す(将来 diffusers 側でコンポーネント構成が変わっても追従できるようにするため)。
    """
    import inspect

    from diffusers import LTX2ConditionPipeline

    base = get_base_pipeline()
    ps = state.flf_pipeline_state
    if not ps["loaded"]:
        with state.lock:
            if not ps["loaded"]:
                accepted = set(inspect.signature(LTX2ConditionPipeline.__init__).parameters.keys())
                components = {k: v for k, v in base.components.items() if k in accepted}
                ps["pipe"] = LTX2ConditionPipeline(**components)
                ps["loaded"] = True
                print("[families.ltx2] LTX2ConditionPipeline (FLF) derived from base pipeline components")
    return ps["pipe"]


def get_iclora_pipeline():
    """IC-LoRA(MergeGreen)用の LTX2InContextPipeline を返す。

    LoRAロード順序(タスク要件・CLAUDE.md 17番・34番): group offload 登録前に
    ロードする必要があるため、base が既に(LoRA無しで)ロード済みの場合は
    families.ltx2.lifecycle.unload("all") で一旦解放してから、load_lora=True で
    base を再ロードする(ロード時間は再度かかるが、CLAUDE.md 34番の charsheet
    bf16-group と同じ設計判断: LoRAロード後に enable_group_offload() を呼ぶ順序を
    崩さないことを優先する)。

    base が未ロードなら通常どおり load_lora=True で新規ロードする。base が既に
    LoRA込みでロード済みなら、有効化されているか確認して再有効化するだけで済む
    (再ロード不要、CLAUDE.md 13番の禁止パターンを避けるため disable_lora() /
    set_adapters() のみで切替する)。

    base.components のフィルタリングは get_flf_pipeline() と同じ理由
    (LTX2InContextPipeline.__init__() は "processor" を受け付けず、代わりに
    任意の "audio_scheduler" を持つ)で必要。
    """
    import inspect

    from diffusers import LTX2InContextPipeline

    from families.ltx2 import lifecycle

    bps = state.base_pipeline_state
    if bps["loaded"] and not bps.get("lora_loaded"):
        print(
            "[families.ltx2] IC-LoRA未ロードのbaseが常駐中のため、一旦unloadしてから"
            "LoRA込みで再ロードします(CLAUDE.md 17番・34番: group offload登録前に"
            "LoRAをロードする必要があるため)"
        )
        lifecycle.unload("all")

    if not bps["loaded"]:
        with state.lock:
            if not bps["loaded"]:
                _load_base_pipeline_locked(load_lora=True)

    base = bps["pipe"]
    if not bps.get("lora_enabled"):
        # 注意: これは pipeline-level の LoraBaseMixin.set_adapters() であり、
        # transformer-level(PeftAdapterMixin)の set_adapters() とは引数名が異なる
        # (pipeline側: adapter_weights、transformer側: weights)。base はここでは
        # LTX2Pipeline インスタンスのため adapter_weights を使う(実機のTypeErrorで
        # 発見・修正済み: 逆にすると "unexpected keyword argument" になる)。
        base.set_adapters([ICLORA_ADAPTER_NAME], adapter_weights=[state.get_runtime_config().iclora_strength])
        bps["lora_enabled"] = True

    ps = state.iclora_pipeline_state
    if not ps["loaded"]:
        with state.lock:
            if not ps["loaded"]:
                accepted = set(inspect.signature(LTX2InContextPipeline.__init__).parameters.keys())
                components = {k: v for k, v in base.components.items() if k in accepted}
                ps["pipe"] = LTX2InContextPipeline(**components)
                ps["loaded"] = True
                print("[families.ltx2] LTX2InContextPipeline (IC-LoRA) derived from base pipeline components")
    return ps["pipe"]


def disable_iclora() -> None:
    """base transformer 上の IC-LoRA アダプタを無効化する(アンロードはしない)。

    他モード(t2v/i2v/flf等)への切替時に呼ばれる想定だが、実際には
    get_base_pipeline() が内部で _ensure_iclora_disabled_for_non_iclora_use() を
    呼ぶため冗長。families.ltx2.family.LTX2Family.load() から明示的に呼ばれる
    エントリポイントとして用意する(将来 base を経由しない呼び出し経路が増えても
    安全なように)。
    """
    ps = state.base_pipeline_state
    pipe = ps.get("pipe")
    _ensure_iclora_disabled_for_non_iclora_use(pipe)


def get_upsampler():
    """LTX2LatentUpsamplerModel を返す(冪等ロード、初回 upscale=1 要求時に遅延ロード)。

    GPU常駐固定(~0.93GBと小さいため、group offloadの対象にしない。CLAUDE.md 33番の
    「大型transformerのみblock-level group offload」の対象は transformer(~35.4GB)であり、
    upsamplerはそれとは無関係の小型モデル)。base(T2V)の vae と同じ latent 空間を扱うが、
    重みは独立(共有コンポーネントではない)。
    """
    ps = state.upsampler_state
    if not ps["loaded"]:
        with state.lock:
            if not ps["loaded"]:
                config = state.get_runtime_config()
                device = config.device
                model = convert.load_upsampler(
                    config.upsampler_path, device, config.upsampler_config_repo
                )
                ps["model"] = model
                ps["loaded"] = True
    return ps["model"]
