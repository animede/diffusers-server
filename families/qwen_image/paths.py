# -*- coding: utf-8 -*-
"""
Qwen-Image 系ファミリーのモデルパス・リポジトリ定数群。

抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py 行46-236 の定数定義一式。
ロジック・実測コメント・既定値は無変更(勝手な改良はしない方針)。
COMFYUI_MODELS_DIR のみ core.config / core.resolve に一元化済みのものを import する。
"""
import os

from core.resolve import COMFYUI_MODELS_DIR

# --- T2I: Qwen-Image 本体 transformer(ComfyUI fp8スケール版。プレフィックスなし) ---
T2I_TRANSFORMER_PATH = os.path.join(
    COMFYUI_MODELS_DIR, "diffusion_models", "qwen_image_fp8_e4m3fn.safetensors"
)
T2I_TRANSFORMER_HF_REPO = "Comfy-Org/Qwen-Image_ComfyUI"
T2I_TRANSFORMER_HF_FILE = "split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors"

# --- T2I/I2I: Lightning LoRA(4step / 8step)---
T2I_LORA_4STEP_PATH = os.path.join(
    COMFYUI_MODELS_DIR, "loras", "Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors"
)
T2I_LORA_4STEP_HF_REPO = "lightx2v/Qwen-Image-Lightning"
T2I_LORA_4STEP_HF_FILE = "Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors"

T2I_LORA_8STEP_PATH = os.path.join(
    COMFYUI_MODELS_DIR, "loras", "Qwen-Image-Lightning-8steps-V2.0.safetensors"
)
T2I_LORA_8STEP_HF_REPO = "lightx2v/Qwen-Image-Lightning"
T2I_LORA_8STEP_HF_FILE = "Qwen-Image-Lightning-8steps-V2.0.safetensors"

# --- T2I: Qwen-Image-2512(公式 Qwen/Qwen-Image-2512、bf16、Apache 2.0。DS_T2I_MODEL の既定)---
# 2026-07-18変更: 旧「2512-4bit」(ovedrive/Qwen-Image-2512-4bit、コミュニティ製 bnb NF4、
# CC-BY-NC-SA-4.0 非商用)を廃止し、公式 Qwen/Qwen-Image-2512 の bf16 transformer に
# 2512用 Lightning LoRA を自前 fuse -> fp8 layerwise cast する方式(無印 qwen-image の
# fp8-lightning 分岐と同一レシピ)へ置き換えた。廃止理由: NF4 + adapter 方式は
# bf16 + 自前fuse 方式に品質で明確に劣ることが判明したため(CLAUDE.md 参照)。
#
# リポジトリ構成の互換性(2026-07-18、HF Hub の files_metadata(LFS sha256)を
# Qwen/Qwen-Image と直接比較して確認済み):
#   - text_encoder(4シャード)・vae・tokenizer の safetensors は sha256 完全一致
#     (バイト単位で同一チェックポイント)。config.json 類も値は同一
#     (_diffusers_version 等の表記差のみ)。
#     => 既存の共有コンポーネント(state.shared)をそのまま再利用でき、二重DL/二重ロード不要。
#   - scheduler/scheduler_config.json も全キー同値(use_dynamic_shifting=true, shift=1.0,
#     base_shift=0.5, max_shift=0.9 等)。shift の扱いも無印と完全に同じでよい。
#   - transformer/config.json は同一アーキテクチャ(QwenImageTransformer2DModel、60層・
#     24ヘッド・128次元)。2512側は pooled_projection_dim キーが無いだけ(未使用フィールド)。
#   - transformer は bf16 9シャード、計約40GB(初回のみDL)。
T2I_2512_REPO = "Qwen/Qwen-Image-2512"

# 2512用 Lightning LoRA(base Qwen-Imageの Qwen-Image-Lightning とは別リポジトリ)
T2I_2512_LORA_4STEP_PATH = os.path.join(
    COMFYUI_MODELS_DIR, "loras", "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"
)
T2I_2512_LORA_4STEP_HF_REPO = "lightx2v/Qwen-Image-2512-Lightning"
T2I_2512_LORA_4STEP_HF_FILE = "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"

T2I_2512_LORA_8STEP_PATH = os.path.join(
    COMFYUI_MODELS_DIR, "loras", "Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors"
)
T2I_2512_LORA_8STEP_HF_REPO = "lightx2v/Qwen-Image-2512-Lightning"
T2I_2512_LORA_8STEP_HF_FILE = "Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors"

# "2512" が正式値(fp8-lightning fuse)。旧値 "2512-4bit"/"2512_4bit" は後方互換のため
# "2512" として扱う(旧NF4版は廃止済み。README のモデル一覧参照)。
T2I_2512_VALUES = {"2512", "2512-4bit", "2512_4bit"}


def is_2512_model(value: str) -> bool:
    """DS_T2I_MODEL(またはリクエストの model)が 2512(Qwen/Qwen-Image-2512、
    fp8-lightning fuse)かどうか。旧値 "2512-4bit" も後方互換で True を返す。"""
    return (value or "qwen-image").strip().lower() in T2I_2512_VALUES


# --- ControlNet(無印 Qwen-Image ベース。InstantX 配布)---
# QwenImageControlNetModel は5〜6層程度の小さなサイドネットワーク(3〜4GB)で、
# 20B本体のtransformerとは別モジュール。QwenImageTransformer2DModel.forward() は
# quantization方式に関わらず(bf16/fp8/GGUFいずれも)共通で controlnet_block_samples
# 引数を受け付ける実装になっているため、理論上どの量子化transformerとも組み合わせられる
# (実機検証結果は README/CLAUDE.md 参照)。base Qwen-Image の共有コンポーネント
# (vae/text_encoder)と T2Iグループのtransformerをそのまま再利用し、ControlNetモジュール
# だけを追加ロードする。ControlNet は無印 Qwen-Image ベースで学習されているため、
# T2Iモデルが 2512 でも常に無印 Qwen-Image の transformer を使う(t2i_group を
# qwen-image でロードし直す)。
CONTROLNET_UNION_REPO = "InstantX/Qwen-Image-ControlNet-Union"
CONTROLNET_INPAINT_REPO = "InstantX/Qwen-Image-ControlNet-Inpainting"

# --- Edit: Qwen-Image-Edit-2511 transformer(charsheet と同一パス)---
EDIT_TRANSFORMER_PATH = os.path.join(
    COMFYUI_MODELS_DIR, "diffusion_models", "qwen_image_edit_2511_bf16.safetensors"
)
EDIT_TRANSFORMER_HF_REPO = "Comfy-Org/Qwen-Image-Edit_ComfyUI"
EDIT_TRANSFORMER_HF_FILE = "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors"

# フォールバック(2511 bf16 が読めない場合、2509 fp8 を使う。charsheet と同様)
EDIT_TRANSFORMER_FALLBACK_PATH = os.path.join(
    COMFYUI_MODELS_DIR, "diffusion_models", "qwen_image_edit_2509_fp8_e4m3fn.safetensors"
)
EDIT_TRANSFORMER_FALLBACK_HF_REPO = "Comfy-Org/Qwen-Image-Edit_ComfyUI"
EDIT_TRANSFORMER_FALLBACK_HF_FILE = "split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"
EDIT_TRANSFORMER_FALLBACK_PREFIX = "model.diffusion_model."

# --- Edit: Lightning 4steps LoRA(charsheet と同一パス)---
EDIT_LORA_LIGHTNING_PATH = os.path.join(
    COMFYUI_MODELS_DIR, "loras", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
)
EDIT_LORA_LIGHTNING_HF_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
EDIT_LORA_LIGHTNING_HF_FILE = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

# --- Edit: Multiple-angles LoRA(charsheet の fp8-lightning-angles 変種専用) ---
# 2509世代向けにComfyUIで配布されている多アングルLoRA(charsheet旧実装
# /home/animede/charsheet/pipeline.py の LORA_ANGLES_PATH と同一ファイル)。
# キーは既に PEFT 形式(lora_A/lora_B、"transformer_blocks." から始まりプレフィックス無し)
# であることを実機確認済み(core.loaders.fuse_lora_into_transformer のdocstring参照)。
EDIT_LORA_ANGLES_PATH = os.path.join(
    COMFYUI_MODELS_DIR, "loras", "Qwen-Edit-2509-Multiple-angles.safetensors"
)
EDIT_LORA_ANGLES_HF_REPO = "Comfy-Org/Qwen-Image-Edit_ComfyUI"
EDIT_LORA_ANGLES_HF_FILE = "split_files/loras/Qwen-Edit-2509-Multiple-angles.safetensors"

# --- GGUF 量子化 transformer(unsloth 配布。DS_QUANT で opt-in)---
# Q4_K_M は bf16(約40GB)の約1/3(約12〜13GB)。diffusers 0.36.0 で動作確認済み:
#   QwenImageTransformer2DModel.from_single_file(path, quantization_config=GGUFQuantizationConfig(...),
#     config=<config repo>, subfolder="transformer", torch_dtype=torch.bfloat16)
# GGUF のテンソルキー名は diffusers ネイティブの命名と一致している(prefix変換不要、
# single_file_model.py の checkpoint_mapping_fn が恒等写像であることを実機で確認済み)。
EDIT_GGUF_HF_REPO = "unsloth/Qwen-Image-Edit-2511-GGUF"
EDIT_GGUF_FILENAME_TEMPLATE = "qwen-image-edit-2511-{suffix}.gguf"
# "Qwen/Qwen-Image-Edit-2511" は2511専用のHFリポジトリで transformer/config.json を持つ
# (2509 の config を流用していた bf16 経路とは別に、GGUF 経路ではこちらを直接使う)。
EDIT_GGUF_CONFIG_REPO = "Qwen/Qwen-Image-Edit-2511"

T2I_GGUF_HF_REPO = "unsloth/Qwen-Image-GGUF"
T2I_GGUF_FILENAME_TEMPLATE = "qwen-image-{suffix}.gguf"
T2I_GGUF_CONFIG_REPO = "Qwen/Qwen-Image"

# --- Edit / T2I 共通: fp8 Lightning マージ版(DS_QUANT=fp8-lightning)---
# GGUF量子化transformerにはLoRAを適用できない(既知の制約、上記参照)ため、
# 代わりにこちらは「bf16 transformerをロード -> Lightning LoRAをその場でfuse(重みに焼き込み)
# -> enable_layerwise_casting(storage_dtype=fp8_e4m3fn) でストレージのみfp8に圧縮」という
# 手順で実現する(実機検証済み: Editでfuse後 38.06GB -> layerwise casting後 19.05GB)。
# 同じ手法を無印Qwen-Image(T2I/I2I)のtransformerにも適用できる(_load_t2i_group_locked参照)。
FP8_LIGHTNING_QUANT_VALUES = {"fp8-lightning", "fp8_lightning", "fp8lightning"}

# --- Layered: Qwen-Image-Layered(画像 -> 複数RGBAレイヤー分解、QwenImageLayeredPipeline)---
# diffusers 0.36.0(comfy-env)には QwenImageLayeredPipeline が存在しない(git版のみ)。
# このワークスペースの venv には git diffusers をインストール済み(README/CLAUDE.md参照)。
#
# 実機調査で判明した重要事実(huggingface_hub blobs API で sha256 を直接比較して確認済み):
#   - text_encoder(Qwen2.5-VL-7B)は Qwen/Qwen-Image-Layered と Qwen/Qwen-Image で
#     4シャードすべて sha256 が完全一致(バイト単位で同一チェックポイント)。
#     => 既存の共有 text_encoder/tokenizer をそのまま再利用でき、
#        二重ロード(16GB分のVRAM浪費)を避けられる。
#   - vae は専用(input_channels=4、RGBA画素空間を直接エンコード/デコードする構成、
#     latents_mean/std・sha256とも base Qwen-Image の vae と異なる)。専用ロードが必要。
#   - transformer は bf16 で約38.05GB(Edit-2511と同一サイズ、シャード5分割の合計値で確認済み)。
#     ComfyUIローカルの qwen_image_layered_fp8mixed.safetensors はComfyUIの"scaled fp8"形式
#     (CLAUDE.mdの既知の罠と同様、diffusersの単純ロードでは読めない)ため使用せず、
#     HF Hubの bf16 シャードを直接ストリーミングロードする(_load_layered_transformer_streaming)。
#   - Lightning LoRAは ComfyUI ワークフロー(Qwen Image Layered Lightning4)が
#     T2I/I2Iと同一ファイル(Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors、
#     T2I_LORA_4STEP_* 定数で既に定義済み)を使っている(node 93 の lora_name)。
#     transformerのアーキテクチャサイズ(60層・24ヘッド・128次元、joint_attention_dim=3584)は
#     標準Qwen-Imageと同一のため、既存の fuse_lightning_lora_and_cast_to_fp8() をそのまま
#     再利用できる(実機確認済み)。
LAYERED_REPO = "Qwen/Qwen-Image-Layered"

# 既定は fp8-lightning(T2I/Editと同じ自前fuse方式)。VRAM逼迫時は
# DS_LAYERED_QUANT=gguf-q4_k_m 等(unsloth/Qwen-Image-Layered-GGUF、LoRA適用不可、
# 50steps/cfg4.0にフォールバック)に切り替える。
LAYERED_DEFAULT_QUANT = "fp8-lightning"

LAYERED_GGUF_HF_REPO = "unsloth/Qwen-Image-Layered-GGUF"
LAYERED_GGUF_FILENAME_TEMPLATE = "qwen-image-layered-{suffix}.gguf"
LAYERED_GGUF_CONFIG_REPO = LAYERED_REPO

# ComfyUI ワークフロー(Qwen Image Layered Lightning4)と同一のデフォルト値
# (KSampler: steps=4, cfg=2.5 / ModelSamplingAuraFlow: shift=1)。
LAYERED_LIGHTNING_STEPS = 4
LAYERED_LIGHTNING_CFG = 2.5
LAYERED_LIGHTNING_SHIFT = 1.0
# Lightning未適用時(GGUF等)のフォールバック値(モデルカード推奨: 50steps/cfg4.0)。
LAYERED_DEFAULT_STEPS = 50
LAYERED_DEFAULT_CFG = 4.0

LAYERED_DEFAULT_LAYERS = 4
LAYERED_DEFAULT_RESOLUTION = 640
LAYERED_VALID_RESOLUTIONS = (640, 1024)  # QwenImageLayeredPipeline.__call__ の assert と同じ

# --- 共有コンポーネント(vae / text_encoder / tokenizer)の由来リポジトリ ---
BASE_REPO = "Qwen/Qwen-Image"
EDIT_PROCESSOR_REPO = "Qwen/Qwen-Image-Edit-2509"  # 2511 transformer でも processor は 2509 のものを使う

# 画像生成で定番の低品質除外セット(手・指の破綻、低解像度、透かし等)。
# ユーザーはUI上で自由に編集・削除できる(あくまでプリセット既定値)。
# 注意: Lightning LoRA 使用時(true_cfg_scale=1.0)は CFG が無効化されるため
# negative_prompt 自体が効かない(LIGHTNING_TRUE_CFG_SCALE 参照)。
NEGATIVE_PROMPT_DEFAULT = (
    "lowres, bad anatomy, bad hands, missing fingers, extra digits, fewer fingers, "
    "cropped, worst quality, low quality, jpeg artifacts, signature, watermark, "
    "username, blurry"
)

# Lightning LoRA 推奨パラメータ(lightx2v 推奨値)
LIGHTNING_SHIFT = 3.0
LIGHTNING_TRUE_CFG_SCALE = 1.0
DEFAULT_SHIFT = None  # None ならスケジューラの事前学習済み既定値を使う
