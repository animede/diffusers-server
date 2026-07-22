# -*- coding: utf-8 -*-
"""
LTX-2.3 ファミリーのランタイム構成(環境変数、DS_LTX2_* 名前空間)。

このファミリーは新規追加のため旧名エイリアスは持たない(core.config._LEGACY_ALIASES に
追記不要。env_str/env_bool/env_float は DS_LTX2_* が未設定なら default にフォールバックする)。

オフロード設計(タスク指示、48GB VRAM対応):
  - "none": 全コンポーネント GPU 常駐(96GB機向け、実測ピーク ~70GB)。
  - "group"(48GB向け本命): transformer(bf16デクオンタイズ後 ~35.4GB)のみ
    block-level group offload(CLAUDE.md 33番・34番の実証済みパターン踏襲。
    CPU側に静的確保、丸ごとスワップは禁止)。text_encoder(~23GB)/ connectors(~6.3GB)/
    vae類は GPU 常駐のまま(旧charsheet "group" モードと同じ設計思想: 一番大きい
    transformer だけを block 単位でオフロードすれば、48GB専有には十分収まる)。
  - "auto": 空きVRAMで none / group を自動選択
    (DS_LTX2_OFFLOAD_FREE_VRAM_THRESHOLD_GB、既定 75.0)。
  - "group"(既定、2026-07-22変更): "auto"は空きVRAMの瞬間値で判定するため、他ファミリー
    切替直後や高解像度・多フレーム動画で"none"モードの巨大な単発アロケーション
    (CLAUDE.md 45番、原因未特定)によりOOMしやすいと実機報告があった。361フレーム
    (15秒)+2xアップスケールでもピーク27.7GB(fp8 TE併用)で完走することを実機確認
    したため、常に"group"を使う設計へ変更した(CLAUDE.md該当項目参照)。

ロード前RAMガード(CLAUDE.md 34番のパターン踏襲): group モード時は CPU に
transformer(bf16 ~35.4GB)を静的確保するため、空きRAM(MemAvailable)が
DS_LTX2_GROUP_OFFLOAD_MIN_RAM_GB(既定 40.0GB)を下回れば明確な RuntimeError で中止する。
"""
from core.config import env_bool, env_float, env_str

MODEL_ID_CONFIG_REPO = "diffusers/LTX-2.3-Diffusers"  # config/tokenizer取得専用(重みはローカル)

DEFAULT_CKPT_PATH = "/home/animede/ComfyUI/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"
DEFAULT_GEMMA_PATH = "/home/animede/ComfyUI/models/text_encoders/gemma_3_12B_it.safetensors"

# latent upsampler(低解像度生成 -> 潜在空間アップスケール -> 高解像度デコードの2段生成、
# 2026-07-20追加)。ComfyUI配布のローカルファイル(bf16、~995MB)を優先し、config だけ
# HF Hub の Lightricks/LTX-2(latent_upsampler/config.json、266バイトの小さいJSON)から
# 取得する(families/ltx2/convert.py の load_upsampler() 参照。use_rational_resampler=False
# へ上書きしてローカルファイルの構造に合わせる)。
DEFAULT_UPSAMPLER_PATH = (
    "/home/animede/ComfyUI/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
)
DEFAULT_UPSAMPLER_CONFIG_REPO = "Lightricks/LTX-2"

# text_encoder(Gemma 3 12B)量子化オプション(2026-07-20追加、CLAUDE.md 33番・37番・39番の
# 「transformer/text_encoder/connectorsを丸ごとCPUスワップしない」方針を厳守した上での
# VRAM削減策)。
#   "none"      : bf16フル精度。load_text_encoder() でロードした bf16 モデルをそのまま使う。
#   "fp8"(既定、2026-07-22変更): ロード済み bf16 text_encoder に apply_layerwise_casting()
#                 (storage_dtype=float8_e4m3fn, compute_dtype=bfloat16)を適用する
#                 (VRAM ~23GB -> ~12GB見込み、実測はCLAUDE.md 42番参照)。embed_tokens/
#                 lm_head/rotary_emb 等の decoder-onlyには無い命名(diffusers既定skip
#                 パターンはtransformer系の命名規則向けのため、Gemma3の embed_tokens/
#                 lm_head はデフォルトではskipされない)は明示的に追加skipパターンで
#                 保護する。ロード元は "none" と**同一のbf16チェックポイント**を量子化
#                 するだけなので、hidden statesのずれや別チェックポイント由来の品質
#                 懸念がない(下記nf4との違い)。
#   "nf4"       : QAT版ローカルチェックポイント(DS_LTX2_TE_QAT_DIR、Googleの別配布、
#                 通常it版とは異なる重み)を BitsAndBytesConfig(nf4)でロードする
#                 (VRAM ~8GB見込み、fp8よりさらに小さい)。**hidden statesがit版とずれ、
#                 品質A/Bはユーザー未確定**(CLAUDE.md 42番、比較用動画3本あり)のため
#                 既定には採用しない。VRAMをさらに切り詰めたい場合の選択肢として残す。
# 既定を "none" から "fp8" へ変更した経緯(2026-07-22): DS_LTX2_OFFLOAD="auto" が
# 空きVRAMの瞬間値で offload_mode を決めるため、他ファミリーとの切替直後や高解像度・
# 多フレーム動画で "none" モードの巨大な単発アロケーション(CLAUDE.md 45番、原因未特定)
# によりOOMしやすいと実機報告があった。"group"+"nf4" なら361フレーム(15秒)+2xアップ
# スケールでもピーク20.7GBで完走することを実機確認したが、上記の品質懸念からユーザーは
# "group"+"fp8" (同条件でピーク27.7GB、いずれもCLAUDE.md該当項目参照)を選択した。
DEFAULT_TE_QUANT = "fp8"
# QAT版 Gemma 3 12B(Google, unquantized権重を保持した完全HF形式ディレクトリ)。
# resolve.py の ComfyUI ディレクトリ解決の流儀(COMFYUI_MODELS_DIR配下)に合わせる。
DEFAULT_TE_QAT_DIR = "/home/animede/ComfyUI/models/text_encoders/gemma-3-12b-it-qat-q4_0-unquantized"

DEFAULT_OFFLOAD = "group"
DEFAULT_OFFLOAD_FREE_VRAM_THRESHOLD_GB = 75.0
DEFAULT_GROUP_OFFLOAD_MIN_RAM_GB = 40.0

# 蒸留モデル既定値(guidance_scale=1.0固定 = CFGなし、タスク指示どおり)。
DEFAULT_STEPS = 8
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 320
DEFAULT_NUM_FRAMES = 25  # 8n+1
DEFAULT_FPS = 24

# FLF(First-Last-Frame)既定値。ComfyUIワークフロー(video_ltx2_3-22b-flf-bf8.json)の
# LTXVAddGuide ×2(frame_idx=0/-1)ノードの既定 strength=0.7 に準拠。frame_rate はワークフロー
# の LTXVConditioning 既定値25(T2V/I2Vの24とは別、FLFワークフロー固有の値のため個別定数にする)。
DEFAULT_FLF_STRENGTH = 0.7
DEFAULT_FLF_FRAME_RATE = 25

# IA2V(Image+Audio to Video、リップシンク動画。2026-07-19追加、tools/ltx2_ia2v_probe.py の
# 実証済みカスタムループを移植)既定値。frame_rate はFLFと同じ25(ComfyUI IA2Vワークフロー
# 系列も25fps基調のため踏襲)。フレーム数は音声長×fpsから自動計算(8n+1丸め)するため
# T2V/I2V/FLFのような固定既定値は持たない(generate.py の _frames_from_audio_duration() 参照)。
DEFAULT_IA2V_FRAME_RATE = 25
DEFAULT_IA2V_PROMPT = (
    "A person looks directly at the camera and talks, natural lip movements synced to speech, "
    "photorealistic, static camera, indoor lighting"
)
DEFAULT_IA2V_NEGATIVE_PROMPT = "blurry, distorted face, static mouth, low quality"

# audio_out(タスク要件): 出力音声の扱い。
#   T2V/I2V/FLF: "on"(既定、従来どおりvocoder生成音声をmux)/ "off"(無音mp4)
#   IA2V:        "original"(既定、入力wavの原音をそのままmux)/ "vocoder"(再合成音)/
#                "none"(無音mp4)
DEFAULT_AUDIO_OUT = "on"
DEFAULT_IA2V_AUDIO_OUT = "original"

# V2A(Video to Audio、IA2Vの鏡像。2026-07-20追加): 既存動画に合った音声を生成して付与する。
# 映像latentを毎ステップclean(=入力動画)へx0固定し、音声latentだけをdenoiseする
# (run_ia2v() の逆。generate.py の run_v2a() docstring参照)。
DEFAULT_V2A_PROMPT = "ambient sound and effects that match the scene, realistic audio"
DEFAULT_V2A_NEGATIVE_PROMPT = "music, voice, speech, low quality, noise"
DEFAULT_V2A_MAX_NUM_FRAMES = 361  # フレーム数上限(8n+1、run_v2a()/run_iclora()の明示バリデーション用)。
# 2026-07-20に257(≈10秒)から361(=15秒@24fps)へ引き上げ。旧5秒制限の正体はUI側の
# max="121"属性でVRAM制約ではなかったが、実測の結果、長尺のVRAM要件は構成依存:
#   - none + TE bf16(96GB機の既定): 241フレーム(10秒)で既にOOM(activationが乗らない)
#   - none + TE nf4: 361フレームでOOM(attention系で23.8GBの単発アロケーションが発生)
#   - group + TE nf4: 361フレーム成功、ピーク43.1GB・生成36.0s(長尺はこの構成を使う。
#     ステップあたりのブロック転送コストは固定のため、長尺ほど相対効率が良い)

# IC-LoRA(In-Context LoRA、MergeGreen、2026-07-20追加): 動画編集モード。中央フレームを
# 緑 RGB(0,191,0) でマスクした参照動画を LTX2InContextPipeline.reference_conditions へ渡し、
# プロンプトで「シーン内で何が変わるか」を指示すると、緑マスク領域が指示通りに変化した動画を
# 生成する(HF repo siraxe/MergeGreen_IC-lora_ltx2.3、Apache-2.0)。既定のマスク範囲は
# 動画全体の中央1/3(mask_start/mask_end で明示上書き可)。
DEFAULT_ICLORA_PATH = "/home/animede/ComfyUI/models/loras/ltx2/ic_merge/MergeGreen_IC-lora_ltx2.3.safetensors"
DEFAULT_ICLORA_HF_REPO = "siraxe/MergeGreen_IC-lora_ltx2.3"
DEFAULT_ICLORA_HF_FILE = "MergeGreen_IC-lora_ltx2.3.safetensors"
# LoRA strength: HFリポジトリのComfyUIワークフロー(MergeGreen_IC-lora_ltx2.3.json、
# LTXICLoRALoaderModelOnly ノード)の既定値 0.9 をそのまま採用(README「0.9 blends
# better」の記述とも一致)。
DEFAULT_ICLORA_STRENGTH = 0.9
# reference_conditions の強さ(1.0=完全にclean、参照動画にそのまま従う。ワークフローの
# LTXAddVideoICLoRAGuide 既定値 strength=1 に対応)。
DEFAULT_ICLORA_REF_STRENGTH = 1.0
DEFAULT_ICLORA_MASK_COLOR = (0, 191, 0)  # HFリポジトリ README 記載の緑マスク色
DEFAULT_ICLORA_NEGATIVE_PROMPT = "static, no change, low quality, blurry, distorted"


class LTX2RuntimeConfig:
    """DS_LTX2_* 環境変数から読み取るランタイム構成。

    DS_LTX2_CKPT_PATH   : ComfyUI scaled-fp8 チェックポイントの絶対パス
    DS_LTX2_GEMMA_PATH  : Gemma 3 12B bf16 チェックポイントの絶対パス
    DS_LTX2_UPSAMPLER_PATH        : latent upsampler(ComfyUI形式bf16、~995MB)の絶対パス
    DS_LTX2_UPSAMPLER_CONFIG_REPO : latent upsampler の config.json 取得元(HF Hub repo id)
    DS_LTX2_TE_QUANT    : "none" | "fp8"(既定、2026-07-22変更) | "nf4"(text_encoder量子化、
                          上記docstring参照)
    DS_LTX2_TE_QAT_DIR  : "nf4" 時にロードする QAT版 Gemma 3 12B ディレクトリの絶対パス
    DS_LTX2_OFFLOAD     : "auto" | "none" | "group"(既定、2026-07-22変更)
    DS_LTX2_OFFLOAD_FREE_VRAM_THRESHOLD_GB : auto判定の閾値(既定75.0、これ以上空きがあればnone)
    DS_LTX2_GROUP_OFFLOAD_MIN_RAM_GB : groupロード前のRAMガード閾値(既定40.0)
    DS_LTX2_GROUP_OFFLOAD_BLOCKS : group offload の num_blocks_per_group(既定1)
    DS_LTX2_ICLORA_PATH     : IC-LoRA(MergeGreen)safetensorsの絶対パス
    DS_LTX2_ICLORA_STRENGTH : IC-LoRA の adapter weight(既定0.9)
    DS_DEVICE           : "cuda"(既定、core.config 共通)
    """

    def __init__(self):
        self.ckpt_path = env_str("DS_LTX2_CKPT_PATH", DEFAULT_CKPT_PATH)
        self.gemma_path = env_str("DS_LTX2_GEMMA_PATH", DEFAULT_GEMMA_PATH)
        self.upsampler_path = env_str("DS_LTX2_UPSAMPLER_PATH", DEFAULT_UPSAMPLER_PATH)
        self.upsampler_config_repo = env_str("DS_LTX2_UPSAMPLER_CONFIG_REPO", DEFAULT_UPSAMPLER_CONFIG_REPO)
        self.iclora_path = env_str("DS_LTX2_ICLORA_PATH", DEFAULT_ICLORA_PATH)
        self.iclora_strength = env_float("DS_LTX2_ICLORA_STRENGTH", DEFAULT_ICLORA_STRENGTH)
        te_quant = env_str("DS_LTX2_TE_QUANT", DEFAULT_TE_QUANT).strip().lower()
        if te_quant not in ("none", "fp8", "nf4"):
            print(f"[families.ltx2.runtime] unknown DS_LTX2_TE_QUANT={te_quant!r}; falling back to 'none'")
            te_quant = "none"
        self.te_quant = te_quant
        self.te_qat_dir = env_str("DS_LTX2_TE_QAT_DIR", DEFAULT_TE_QAT_DIR)
        self.offload = env_str("DS_LTX2_OFFLOAD", DEFAULT_OFFLOAD).strip().lower()
        self.offload_free_vram_threshold_gb = env_float(
            "DS_LTX2_OFFLOAD_FREE_VRAM_THRESHOLD_GB", DEFAULT_OFFLOAD_FREE_VRAM_THRESHOLD_GB
        )
        self.group_offload_min_ram_gb = env_float(
            "DS_LTX2_GROUP_OFFLOAD_MIN_RAM_GB", DEFAULT_GROUP_OFFLOAD_MIN_RAM_GB
        )
        self.group_offload_blocks = int(env_str("DS_LTX2_GROUP_OFFLOAD_BLOCKS", "1"))
        self.device = env_str("DS_DEVICE", "cuda")
        self.attention_backend = env_str("DS_ATTN", "default").strip().lower()
        self.compile = env_bool("DS_COMPILE", False)

    def __repr__(self):
        return (
            f"LTX2RuntimeConfig(offload={self.offload!r}, "
            f"offload_free_vram_threshold_gb={self.offload_free_vram_threshold_gb}, "
            f"group_offload_min_ram_gb={self.group_offload_min_ram_gb}, "
            f"group_offload_blocks={self.group_offload_blocks}, te_quant={self.te_quant!r}, "
            f"device={self.device!r}, "
            f"attention_backend={self.attention_backend!r}, compile={self.compile})"
        )


def resolve_offload_mode(config: "LTX2RuntimeConfig", free_vram_gb: float) -> str:
    """"auto" を実際のモード("none"/"group")へ解決する。"none"/"group" 明示指定はそのまま。"""
    mode = config.offload
    if mode in ("none", "group"):
        return mode
    if mode != "auto":
        print(f"[families.ltx2.runtime] unknown DS_LTX2_OFFLOAD={mode!r}; falling back to auto")
    return "none" if free_vram_gb >= config.offload_free_vram_threshold_gb else "group"
