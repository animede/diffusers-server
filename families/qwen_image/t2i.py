# -*- coding: utf-8 -*-
"""
T2I(QwenImagePipeline)/ I2I(QwenImageImg2ImgPipeline)のロードと Lightning LoRA 制御。

T2I モデルは2種類("qwen-image" 無印 / "2512" = Qwen/Qwen-Image-2512)で、どちらも
同じ t2i_group に排他的にロードされる(48GBでは両方常駐させない)。

2026-07-18変更: 旧「2512-4bit」(ovedrive/Qwen-Image-2512-4bit、bnb NF4 の自己完結
パイプライン + PEFT adapter 方式)を廃止し、公式 Qwen/Qwen-Image-2512 bf16 transformer に
2512用 Lightning LoRA を自前 fuse -> fp8 layerwise cast する方式(無印の fp8-lightning
分岐と同一レシピ)へ置き換えた。これに伴い 2512 も無印と同じ共有コンポーネント
(state.shared)+ transformer 差し替え構造になり、I2I(QwenImageImg2ImgPipeline)にも
対応した(旧NF4は Img2Img クラスを持たない自己完結パイプラインのため T2I 専用だった)。
共有コンポーネントの互換性確認結果は paths.py の T2I_2512_REPO コメント参照
(text_encoder/vae/tokenizer は sha256 完全一致、scheduler 設定も全キー同値)。

抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py
  - _load_t2i_group_locked()(行810-926)
  - _apply_t2i_loras()(行943-970)
  - set_t2i_lightning() / _set_lightning_adapters()(行1114-1146)
  - get_t2i_pipeline() / get_i2i_pipeline()(行1508-1529)

ロジック・既定値・実測コメントは無変更(勝手な改良はしない方針)。
"""
import time

import torch
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    QwenImageImg2ImgPipeline,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)

from core.loaders import (
    fuse_lightning_lora_and_cast_to_fp8,
    load_transformer_from_config,
    load_transformer_from_pretrained_streaming,
    load_transformer_gguf,
)
from core.optimize import apply_attention_backend, apply_compile, configure_transformer_offload
from core.resolve import COMFYUI_MODELS_DIR, resolve_model_path

from families.qwen_image import state
from families.qwen_image.paths import (
    BASE_REPO,
    T2I_2512_LORA_4STEP_HF_FILE,
    T2I_2512_LORA_4STEP_HF_REPO,
    T2I_2512_LORA_4STEP_PATH,
    T2I_2512_REPO,
    T2I_GGUF_CONFIG_REPO,
    T2I_GGUF_FILENAME_TEMPLATE,
    T2I_GGUF_HF_REPO,
    T2I_LORA_4STEP_HF_FILE,
    T2I_LORA_4STEP_HF_REPO,
    T2I_LORA_4STEP_PATH,
    T2I_LORA_8STEP_HF_FILE,
    T2I_LORA_8STEP_HF_REPO,
    T2I_LORA_8STEP_PATH,
    T2I_TRANSFORMER_HF_FILE,
    T2I_TRANSFORMER_HF_REPO,
    T2I_TRANSFORMER_PATH,
    is_2512_model,
)
from families.qwen_image.runtime import is_fp8_lightning_quant, quant_suffix
from families.qwen_image.shared import load_shared_components_locked, warn_if_model_cpu_with_quant

import os


def _load_t2i_group_locked() -> None:
    """T2I transformer をロードし、T2I/I2I パイプラインを構築する(共有コンポーネントを再利用)。

    どのモデル/量子化でロードするかは runtime_config(t2i_model / quant)で決まる:
      - t2i_model="2512": 常に fp8-lightning fuse 方式(Qwen/Qwen-Image-2512 bf16 シャード
        -> 2512用 Lightning 4steps V1.0 fuse -> fp8 layerwise cast)。DS_QUANT の
        GGUF/none 指定は 2512 には適用されない(2512のGGUF/ComfyUI単一ファイル経路は
        用意していないため。指定されていたら警告して fp8-lightning にフォールバック)。
      - t2i_model="qwen-image": DS_QUANT に応じて GGUF / fp8-lightning / bf16 の3経路(従来どおり)。

    fp8-lightning 選択時は Edit と同じ理由でロード順序を変える: bf16フルtransformer
    (約38GB)を先にGPUへロードしてLightning LoRAをfuseする必要があるため、共有コンポーネント
    (text_encoder等)のロードはそれより後に回す(先にロードするとVRAMの空きが減り、
    fuse時の一時ピークが収まらなくなるリスクが高まるため)。GGUF/非量子化時は従来通り
    共有コンポーネントを先にロードする。

    抽出元: pipeline_manager.py _load_t2i_group_locked()(行810-926)。
    """
    t2i_group = state.t2i_group
    if t2i_group["loaded"]:
        return
    raw_quant = state.get_runtime_config().quant
    quant = quant_suffix(raw_quant)
    fp8_lightning = is_fp8_lightning_quant(raw_quant)
    is_2512 = is_2512_model(state.get_runtime_config().t2i_model)
    if is_2512 and not fp8_lightning:
        # 2512 は fp8-lightning fuse 方式のみ提供(上記docstring参照)。
        print(
            f"[families.qwen_image] warning: t2i_model=2512 は fp8-lightning 方式のみ対応です"
            f"(DS_QUANT={raw_quant!r} は無視して fp8-lightning でロードします)"
        )
    if is_2512:
        quant = None
        fp8_lightning = True
    small_transformer = bool(quant) or fp8_lightning

    if not fp8_lightning:
        load_shared_components_locked(small_transformer_active=small_transformer)

    t0 = time.time()
    offload_mode = state.get_offload_mode(small_transformer_active=small_transformer)
    load_device = state.t2i_load_device(offload_mode)
    if not fp8_lightning:
        warn_if_model_cpu_with_quant(quant, offload_mode)

    if quant:
        filename = T2I_GGUF_FILENAME_TEMPLATE.format(suffix=quant)
        local_path = os.path.join(COMFYUI_MODELS_DIR, "diffusion_models", filename)
        path = resolve_model_path(local_path, T2I_GGUF_HF_REPO, filename)
        print(f"[families.qwen_image] loading T2I transformer as GGUF({quant}) from {path}")
        transformer = load_transformer_gguf(QwenImageTransformer2DModel, path, T2I_GGUF_CONFIG_REPO)
        # GGUFは小さいためCPU RAM常駐を避け、常にフルGPU常駐にする(group系オフロードは使わない)。
        transformer.to("cuda")
        transformer_offload_mode = "none"
    elif fp8_lightning:
        # bf16 transformer(無印 Qwen/Qwen-Image または Qwen/Qwen-Image-2512)を直接GPUへ
        # ロード(実測 約38GB)し、その場で対応する Lightning LoRA(4steps)をfuseしてから
        # enable_layerwise_casting でストレージをfp8化する(Edit(2511)と同一手法)。
        # 共有コンポーネント読み込み前にVRAMの空きを最大限確保するため、offload_modeに
        # 関わらず常に "cuda:0" へ直接ロードする(group系オフロードは使わない)。
        #
        # 2026-07-18バグ修正(T2I fp8-lightning 4steps の霧がかかったような品質崩壊の根本原因):
        # ComfyUI配布の qwen_image_fp8_e4m3fn.safetensors(T2I_TRANSFORMER_PATH)は、
        # 元々fp8_e4m3fn精度で量子化された重みを bf16 として保存し直したファイルであることが
        # 実機検証で判明した(load_safetensors_streaming(dtype=bf16)でロードした後、代表8層の
        # 重みを一つずつ .to(float8_e4m3fn).to(float32) で往復させたところ、全要素が完全に
        # ロスレスでfp8往復できた=100%の要素で元の値と誤差ゼロ。つまりbf16の仮数部下位ビットは
        # 全て0で、実質的な精度はfp8のまま)。Lightning LoRA(4steps)をfuseすると加わる補正量は
        # 実測で平均絶対値 約3e-5(fp8_e4m3fnの量子化誤差の平均絶対値ともほぼ同じ大きさ)しかなく、
        # fuse後に enable_layerwise_casting で再度fp8へキャストすると、LoRAが加えた補正がfp8の
        # 量子化グリッドに再度丸め込まれてほぼ消滅してしまう(delta/quant_err比 ≈ 1.0を実測)。
        # これが「30steps/cfg4(Lightningなし)は完璧なのに、fp8-lightning 4stepsだけ霧がかかる」
        # 症状の直接原因(Lightning LoRAのfuseが実質no-op化していた)。対照実験として adapter
        # モード(fuseせずLoRAをPEFT経由で加算、baseのみfp8ストレージ)は同一seed/prompt/shiftで
        # シャープな出力になることを確認済み(LoRAの寄与がbf16 compute_dtypeで守られるため)。
        #
        # 対策: fp8-lightning経路は HF Hub の bf16 シャード(真にフル精度)を直接ストリーミング
        # ロードし、そこにLoRAをfuseしてからfp8化する(ComfyUIの事前量子化fp8ファイルは使わない)。
        # T2I_TRANSFORMER_PATH/HF_REPO/HF_FILE(ComfyUIファイル)は GGUF不使用・非量子化時の
        # 通常ロード(下の else 分岐)では引き続き使う(そちらはfuse+再量子化をしないため無害)。
        #
        # 2512(Qwen/Qwen-Image-2512)も全く同じレシピを適用する(2026-07-18導入):
        # ベースrepoとLoRA(lightx2v/Qwen-Image-2512-Lightning 4steps V1.0 bf16)を
        # 差し替えるだけで、fuse手順・fp8化・VRAMピーク特性は無印と同一。
        if is_2512:
            repo = T2I_2512_REPO
            lora4 = (T2I_2512_LORA_4STEP_PATH, T2I_2512_LORA_4STEP_HF_REPO, T2I_2512_LORA_4STEP_HF_FILE)
        else:
            repo = BASE_REPO
            lora4 = (T2I_LORA_4STEP_PATH, T2I_LORA_4STEP_HF_REPO, T2I_LORA_4STEP_HF_FILE)
        print(
            f"[families.qwen_image] loading T2I transformer for fp8-lightning fuse from "
            f"{repo}/transformer (HF Hub bf16 shards, 直接cuda:0へストリーミング)"
        )
        transformer = load_transformer_from_pretrained_streaming(
            QwenImageTransformer2DModel, repo, "transformer", "cuda:0"
        )
        print("[families.qwen_image] fusing T2I Lightning LoRA and casting to fp8_e4m3fn storage...")
        fuse_lightning_lora_and_cast_to_fp8(transformer, *lora4)
        if torch.cuda.is_available():
            print(f"[families.qwen_image] T2I fp8-lightning transformer ready, VRAM={torch.cuda.memory_allocated()/1024**3:.2f}GB")
        transformer_offload_mode = "none"
    else:
        path = resolve_model_path(T2I_TRANSFORMER_PATH, T2I_TRANSFORMER_HF_REPO, T2I_TRANSFORMER_HF_FILE)
        print(f"[families.qwen_image] loading T2I transformer from {path} (load_device={load_device})")
        transformer = load_transformer_from_config(
            QwenImageTransformer2DModel, BASE_REPO, "transformer", path, load_device
        )
        transformer_offload_mode = offload_mode

    if fp8_lightning:
        # transformer をVRAMに確保した後で共有コンポーネント(text_encoder等)をロードする
        # (上記の順序に関する注意を参照)。
        load_shared_components_locked(small_transformer_active=small_transformer)
        warn_if_model_cpu_with_quant("fp8-lightning", offload_mode)

    # scheduler は常に無印 BASE_REPO の設定を使う(Qwen/Qwen-Image-2512 の
    # scheduler_config.json と全キー同値であることを確認済み。paths.py 参照)。
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(BASE_REPO, subfolder="scheduler")

    if transformer_offload_mode == "model_cpu":
        t2i_pipe = QwenImagePipeline(
            scheduler=scheduler,
            vae=state.shared["vae"],
            text_encoder=state.shared["text_encoder"],
            tokenizer=state.shared["tokenizer"],
            transformer=transformer,
        )
        t2i_pipe.enable_model_cpu_offload()
    else:
        if not quant and not fp8_lightning:  # quant/fp8_lightning時は上ですでに .to("cuda") 済み
            configure_transformer_offload(transformer, transformer_offload_mode)
        t2i_pipe = QwenImagePipeline(
            scheduler=scheduler,
            vae=state.shared["vae"],
            text_encoder=state.shared["text_encoder"],
            tokenizer=state.shared["tokenizer"],
            transformer=transformer,
        )

    apply_attention_backend(transformer, state.get_runtime_config().attention_backend, state.get_runtime_config().device)
    apply_compile(transformer, state.get_runtime_config().compile)

    # I2I パイプラインは同一インスタンスを共有するラッパー(二重ロードしない)。
    # 2512 も同じ transformer 共有構造のため、I2I はモデルを問わず利用できる。
    i2i_pipe = QwenImageImg2ImgPipeline(
        scheduler=scheduler,
        vae=state.shared["vae"],
        text_encoder=state.shared["text_encoder"],
        tokenizer=state.shared["tokenizer"],
        transformer=transformer,
    )

    quant_label = "fp8-lightning" if fp8_lightning else quant
    if fp8_lightning:
        # Lightning LoRAはすでに重みにfuse済み。追加のLoRAロードは不要で、常にLightning
        # 適用状態として扱う(無効化不可)。
        t2i_group["lora_available"] = True
        t2i_group["lightning_merged"] = True
    else:
        _apply_t2i_loras(t2i_pipe, gguf_quantized=bool(quant))

    t2i_group["transformer"] = transformer
    t2i_group["t2i_pipe"] = t2i_pipe
    t2i_group["i2i_pipe"] = i2i_pipe
    t2i_group["scheduler"] = scheduler
    t2i_group["loaded"] = True
    t2i_group["load_time_s"] = time.time() - t0
    t2i_group["quant"] = quant_label
    print(
        f"[families.qwen_image] T2I/I2I group loaded in {t2i_group['load_time_s']:.1f}s "
        f"(model={'2512' if is_2512 else 'qwen-image'}, offload_mode={transformer_offload_mode}, "
        f"quant={quant_label})"
    )


def _apply_t2i_loras(pipe, gguf_quantized: bool = False) -> None:
    """T2I/I2I 用 Lightning LoRA(4step / 8step)を両方ロードし、既定は無効化しておく。
    有効化は set_t2i_lightning() でリクエストごとに切り替える。

    この関数は無印 qwen-image の GGUF/bf16 経路専用(2512 は常に fp8-lightning fuse のため
    ここには来ない)。GGUF量子化 transformer(GGUFLinear層)には現行の diffusers/peft では
    LoRAを適用できない(PEFTのターゲットモジュール検出がGGUFLinearを認識しないため
    "Target modules {...} not found" で失敗する。実機検証済み)。失敗時はエラーにせず
    警告ログを出し、lora_available=False としてフォールバックする。

    抽出元: pipeline_manager.py _apply_t2i_loras()(行943-970)。
    """
    t2i_group = state.t2i_group
    try:
        path4 = resolve_model_path(T2I_LORA_4STEP_PATH, T2I_LORA_4STEP_HF_REPO, T2I_LORA_4STEP_HF_FILE)
        pipe.load_lora_weights(path4, adapter_name="lightning4")
        path8 = resolve_model_path(T2I_LORA_8STEP_PATH, T2I_LORA_8STEP_HF_REPO, T2I_LORA_8STEP_HF_FILE)
        pipe.load_lora_weights(path8, adapter_name="lightning8")
        pipe.disable_lora()  # 既定は無効
        t2i_group["lora_available"] = True
    except Exception as e:  # noqa: BLE001
        t2i_group["lora_available"] = False
        t2i_group["lora_unavailable_reason"] = f"{type(e).__name__}: {e}"[:300]
        if gguf_quantized:
            print(
                f"[families.qwen_image] warning: GGUF量子化transformerへのT2I Lightning LoRA適用に失敗しました"
                f"(既知の制約: PEFTがGGUFLinearを未対応)。LoRAなしにフォールバックします: "
                f"{type(e).__name__}: {e}"
            )
        else:
            print(f"[families.qwen_image] warning: T2I Lightning LoRA のロードに失敗しました: {e}")


def _set_lightning_adapters(pipe, group_state: dict, steps: int) -> bool:
    """steps に応じて Lightning LoRA(4step/8step アダプタ)を有効/無効化する共通処理。
    4 または 8 のときのみ有効化。有効化できた場合 True を返す
    (LoRA自体が未ロード/非対応の場合は常に False)。

    抽出元: pipeline_manager.py _set_lightning_adapters()(行1114-1132)。
    """
    if not group_state.get("lora_available"):
        return False
    try:
        if steps == 4:
            pipe.set_adapters(["lightning4"], adapter_weights=[1.0])
            return True
        if steps == 8:
            pipe.set_adapters(["lightning8"], adapter_weights=[1.0])
            return True
        pipe.disable_lora()
        return False
    except Exception as e:  # noqa: BLE001
        print(f"[families.qwen_image] warning: Lightning LoRA 切替に失敗しました: {e}")
        return False


def set_t2i_lightning(pipe, steps: int) -> bool:
    """T2I グループの Lightning LoRA を有効/無効化する。

    fp8-lightning(重みにfuse済み。2512 は常にこれ)の場合は常に True を返す(無効化不可)。

    抽出元: pipeline_manager.py set_t2i_lightning()(行1135-1145)。
    """
    group = state.t2i_group
    if group.get("lightning_merged"):
        return True
    return _set_lightning_adapters(pipe, group, steps)


def _switch_t2i_model_if_needed(model: "str | None") -> None:
    """リクエストで明示された T2I モデル("2512" | "qwen-image"。旧値 "2512-4bit" は
    後方互換で "2512" として扱う)に応じて、現在アクティブなモデル
    (state.get_runtime_config().t2i_model)を切り替える。

    タスク1(UIからのT2Iモデル切替)対応: 48GBでは2512と無印Qwen-Imageの transformer を
    両方常駐させない方針のため、要求モデルが現在のグループと異なる場合は、T2Iグループを
    先に unload してからランタイム設定(t2i_model)を更新する。生成はグローバルロック1本
    (core.gpu.generation_lock)で排他されるため、この関数が呼ばれる時点で他の生成は
    走っていない前提(app.py の _generate_or_409() がロック取得後に registry.load() 経由で呼ぶ)。

    model が None(未指定)の場合は何もしない(現行の DS_T2I_MODEL 既定のまま)。
    """
    if model is None:
        return
    requested_2512 = is_2512_model(model)
    current_2512 = is_2512_model(state.get_runtime_config().t2i_model)
    if requested_2512 == current_2512:
        return  # 既に要求モデルがアクティブ(まだ未ロードの可能性はあるが同一グループ)

    from families.qwen_image import lifecycle

    need_switch = False
    with state.lock:
        # 再チェック(ロック取得までの間に他スレッドが切り替えた可能性を考慮)。
        current_2512 = is_2512_model(state.get_runtime_config().t2i_model)
        if requested_2512 == current_2512:
            return
        if requested_2512:
            print("[families.qwen_image] switching T2I model: qwen-image -> 2512")
        else:
            print("[families.qwen_image] switching T2I model: 2512 -> qwen-image")
        state.get_runtime_config().t2i_model = "2512" if requested_2512 else "qwen-image"
        need_switch = True

    # lifecycle.unload() は内部で state.lock を取得するため、上の with ブロックの外側で呼ぶ
    # (非再入ロックのためネストできない。CLAUDE.md 14番の罠と同じ理由)。
    # target="t2i" は t2i_group と、その transformer を参照する ControlNet グループを解放する
    # (解放順序は lifecycle.unload() 参照)。
    if need_switch:
        lifecycle.unload("t2i")


def get_t2i_pipeline(model: "str | None" = None) -> QwenImagePipeline:
    """DS_T2I_MODEL(または引数 model で明示指定)に応じて 2512 / 無印 Qwen-Image の
    いずれかを返す。model 指定時、現在アクティブなモデルと異なれば自動的に切り替える
    (T2Iグループをunloadしてから再ロードする。タスク1対応)。

    抽出元: pipeline_manager.py get_t2i_pipeline()(行1508-1520)。
    """
    _switch_t2i_model_if_needed(model)
    if not state.t2i_group["loaded"]:
        with state.lock:
            if not state.t2i_group["loaded"]:
                _load_t2i_group_locked()
    return state.t2i_group["t2i_pipe"]


def get_i2i_pipeline(model: "str | None" = None) -> QwenImageImg2ImgPipeline:
    """I2I パイプラインを返す。2026-07-18から I2I も "2512" | "qwen-image" の両方に対応
    (2512 が無印と同じ transformer 共有構造になったため。QwenImageImg2ImgPipeline は
    _load_t2i_group_locked() が T2I と同一 transformer から構築済み)。

    旧実装との差分: 旧NF4(2512-4bit)は Img2Img クラスを持たない自己完結パイプライン
    だったため I2I 非対応で、model="2512-4bit" 指定時は ValueError(400)を返し、
    さらに t2i_2512_group 常駐時のOOM回避・t2i_model 同期の特殊処理を持っていた。
    新方式では T2I と同じ _switch_t2i_model_if_needed() による排他切替に一本化され、
    これらの特殊処理は不要になった(グループが1つになったため、モデル不整合による
    二重常駐OOM自体が構造的に起きない)。

    抽出元: pipeline_manager.py get_i2i_pipeline()(行1523-1529)。
    """
    _switch_t2i_model_if_needed(model)
    if not state.t2i_group["loaded"]:
        with state.lock:
            if not state.t2i_group["loaded"]:
                _load_t2i_group_locked()
    return state.t2i_group["i2i_pipe"]
