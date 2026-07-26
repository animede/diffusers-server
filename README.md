# DiffusersWEBUI

A single FastAPI server + web UI that runs multiple state-of-the-art image and video
diffusion model families (Qwen-Image family, FLUX.2, Z-Image-Turbo, LTX-2.3 video+audio,
JoyAI-Edit-Plus) side by side, with automatic VRAM management, quantization, and a shared
comparison UI — built to run comfortably on a single 48GB-class GPU.

(リポジトリ/ディレクトリ名: `diffusers-server`)

## 概要

複数のモデルファミリー(Qwen-Image 系 / FLUX.2 / Z-Image-Turbo / LTX-2.3 動画+音声 /
JoyAI-Edit-Plus)を単一の FastAPI サーバ + 単一の Web UI で動かす統合画像・動画生成サーバ。
VRAM に同時常駐できないモデル同士は自動でアンロード(排他制御)しながら切り替えるため、
1枚の GPU で複数系統のモデルを使い分けられる。

### 主な機能

- **T2I(Text-to-Image)** — Qwen-Image / Qwen-Image-2512 / FLUX.2-dev / Z-Image-Turbo
- **I2I(Image-to-Image)** — 上記モデル共通
- **Edit(参照画像編集)** — Qwen-Image-Edit-Plus(2511)/ JoyAI-Edit-Plus(複数参照画像合成)
- **ControlNet**(Canny 等)、**Inpaint**(マスク指定領域の再生成)
- **Layered**(RGBA レイヤー分解生成)
- **背景削除**(`rembg` ベース、独立タブ + 各結果パネルへのインスタントボタン)
- **キャラクターシート生成**(1枚の画像から8方向のキャラクター画像を自動生成)
- **動画生成(LTX-2.3)** — Text-to-Video / Image-to-Video / First-Last-Frame 補間 /
  リップシンク(Image+Audio-to-Video)/ 任意キーフレーム条件付け / Video-to-Audio /
  IC-LoRA による動画編集 / 潜在空間アップスケール(2x)
- **LLM プロンプト支援**(任意の OpenAI 互換ローカル LLM サーバと連携した英語強化・和文英訳)
- **比較ギャラリー**(生成結果を並べて比較・ダウンロード)

## 動作要件

- **Python 3.12**
- **CUDA 対応 GPU、VRAM 48GB 推奨**(一部機能はより少ない VRAM でも動作しますが、複数
  モデルを頻繁に切り替える場合や高解像度・長尺動画生成では 48GB 級を推奨)
- **RAM 64GB 推奨**(group offload 系のオフロードモードがホスト RAM に重みを保持するため)
- **ffmpeg**(任意。動画生成時の音声 mux に使用。無い場合は映像のみ mp4 + 音声 wav の
  別ファイルにフォールバック)

### 機能別のおおよその VRAM 目安(実測ベース、48GB GPU 環境)

| 機能 | 目安ピークVRAM | 備考 |
|---|---|---|
| T2I / I2I(Qwen-Image, fp8-lightning) | 約35GB | Lightning LoRA fuse + fp8 layerwise casting |
| Edit(Qwen-Image-Edit-Plus, fp8-lightning) | 約35〜43GB | 解像度依存。1024²は text_encoder の一時 CPU 退避で対応 |
| ControlNet / Inpaint | 約42〜43GB | |
| Layered(RGBAレイヤー分解) | 約42GB | |
| FLUX.2-dev T2I/I2I | 約18〜34GB | オフロード設定依存 |
| Z-Image-Turbo(T2I/I2I/Inpaint) | 約22GB | 単一モデル参照共有、bf16全常駐 |
| LTX-2.3 動画生成(オフロード無し) | 約70〜75GB | 96GB級GPU向け |
| LTX-2.3 動画生成(48GB向けgroupオフロード) | 約35〜43GB | transformer のみ block-level group offload。長尺(15秒級)も対応 |
| JoyAI-Edit-Plus(複数参照編集) | 約33〜34GB | transformer⇔text_encoder 相互オフロードで48GB専有でも動作 |

実際の値は解像度・ステップ数・オフロード設定によって変動します。詳細な実測値は
`API_SPEC.md` および各エンドポイントのレスポンスに含まれる `peak_vram_gb` を参照してください。

## セットアップ

### 1. Python 仮想環境の作成

```bash
python3.12 -m venv venv
source venv/bin/activate
```

### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

**重要**: `diffusers` は PyPI のリリース版ではなく **git 版**が必要です(本リポジトリは
`0.40.0.dev0` で動作確認済み)。理由: Z-Image / LTX-2.3 / Layered など一部パイプラインが
リリース版にまだ含まれていない実装に依存しているため。

```bash
pip install "git+https://github.com/huggingface/diffusers"
```

git 版は日々更新されるため、動作確認済みのコミットと異なる場合があります。問題が
発生した場合は `requirements.txt` に記載のバージョンや、diffusers のコミット履歴から
近い時期のコミットを試してください。

`transformers` は 5.x 系列を使用します(`AutoProcessor` が一部モデルで正しく解決できない
既知の問題があるため、`Qwen2VLProcessor` 等クラスを明示指定するコードになっています)。

CUDA 対応の PyTorch(cu128 系でビルド済みのものを推奨)を GPU 環境に合わせて別途
インストールしてください:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. モデルの取得

モデルの重みは本リポジトリに含まれません。初回リクエスト時に Hugging Face Hub から
自動ダウンロードされ、通常の HF キャッシュ(`~/.cache/huggingface`)に保存されます。

既に [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 環境をお持ちの場合、
`DS_COMFYUI_DIR` 環境変数(既定 `~/ComfyUI`)配下の `models/diffusion_models/` や
`models/loras/` 等を優先的に参照し、無ければ Hugging Face Hub からダウンロードします。
LTX-2.3(動画生成)は ComfyUI 形式のチェックポイントを前提とするため、
`DS_LTX2_CKPT_PATH` 等でファイルパスを指定する必要があります(詳細は環境変数表参照)。

### 4. Mage-Flow セットアップ(任意、別プロセス・専用venv)

Mage-Flow(Microsoft、軽量4.1B T2I/Edit、MIT)は torch 2.13 / transformers 5.5 /
flash-attn 2.8.3 を要求し、本体venvとバージョン衝突するため**完全隔離の専用venv**で
動かします(`--system-site-packages` は使わない。comfy-env にもインストールしない)。

```bash
# 1) Mage リポジトリの取得
mkdir -p third_party && git clone --depth 1 https://github.com/microsoft/Mage.git third_party/Mage

# 2) 専用venv(完全隔離)
python3.12 -m venv venv-mageflow

# 3) torch は cu130 系(sm_120/Blackwell のカーネルは cu129 以降にしか入っていない。
#    cu126/cu128 wheel は sm_120 非対応なので不可)
venv-mageflow/bin/pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.13.0 torchvision==0.28.0

# 4) 残りの依存 + ラッパーサービス用パッケージ
venv-mageflow/bin/pip install diffusers==0.38.0 transformers==5.5.0 \
    "accelerate>=1.0.0" "safetensors>=0.8.0" einops pydantic pillow loguru \
    fastapi uvicorn python-multipart requests ninja

# 5) flash-attn 2.8.3 をソースビルド(必須。varlen packing が唯一のattention経路)。
#    nvcc は torch の CUDA メジャーと一致させる(cu130 → /usr/local/cuda-13.0)。
#    ★並列数を必ず制限すること(無制限だと 20並列nvcc がホストRAMを食い潰し
#      システム全体を巻き込むOOMになった実績あり。CLAUDE.md 50番)
systemd-run --user --scope -p MemoryMax=45G -p MemorySwapMax=0 \
    env PATH=/usr/local/cuda-13.0/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0 \
    MAX_JOBS=4 NVCC_THREADS=2 TORCH_CUDA_ARCH_LIST="12.0" \
    venv-mageflow/bin/pip install --no-build-isolation flash-attn==2.8.3

# 6) mage_flow パッケージ(依存は上で導入済みなので --no-deps)
venv-mageflow/bin/pip install -e third_party/Mage/mage_flow --no-deps
```

モデル(HF Hub、各リポジトリ自己完結型 diffusers-style)は初回リクエスト時に自動
ダウンロードされます(T2I/Edit 各 base/rl/turbo の6リポジトリ。既定の `rl` は
`microsoft/Mage-Flow` + `microsoft/Mage-Flow-Edit`)。

## 起動方法

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8601
```

起動後、ブラウザで `http://localhost:8601/` を開くと Web UI が表示されます。

Mage-Flow を使う場合はラッパーサービスも起動します(別プロセス、既定ポート8602):

```bash
./run_mageflow.sh                      # ポートは DS_MAGEFLOW_PORT で変更可
# 本体側はラッパーへ DS_MAGEFLOW_URL(既定 http://127.0.0.1:8602)で接続する。
# ポートを変えた場合は本体起動時に DS_MAGEFLOW_URL を合わせること。
```

## UI 構成

タブ構成: T2I / I2I / Edit / ControlNet / Inpaint / Layered / 背景削除 / キャラシート /
動画(LTX-2.3)/ Mage-Flow。

- **T2I / I2I タブ**: モデル選択で Qwen-Image-2512 / Qwen-Image(無印) / FLUX.2-dev /
  Z-Image-Turbo を切り替え
- **Edit タブ**: Qwen-Image-Edit-Plus(2511)/ JoyAI-Edit-Plus(複数参照画像合成)を切替
- **Inpaint タブ**: Qwen ControlNet / Z-Image-Turbo を切替
- **動画タブ**: T2V / I2V / FLF(First-Last-Frame)等のサブタブ
- **Mage-Flow タブ**: T2I / Edit のサブタブ(base/rl/turbo バリアント選択、
  「排他制御」チェックボックス既定ON。別プロセスのラッパーサービス経由、
  未起動時は起動コマンド入りのエラーメッセージを表示)

各生成結果パネル・比較ギャラリーカードには、ダウンロードボタンとインスタント背景削除
ボタンが付いています。

## API

主要なエンドポイント:

| エンドポイント | 内容 |
|---|---|
| `POST /api/t2i` / `/api/i2i` | Qwen-Image系 T2I / I2I(`model` パラメータでモデル切替) |
| `POST /api/edit` | Qwen-Image-Edit-Plus によるマルチ参照画像編集 |
| `POST /api/controlnet` / `/api/inpaint` | ControlNet(Canny等)/ Inpainting |
| `POST /api/layered` | RGBAレイヤー分解生成 |
| `POST /api/remove_bg` | 背景除去(rembg) |
| `POST /api/flux2/t2i` / `/api/flux2/i2i` | FLUX.2-dev |
| `POST /api/zimage/t2i` / `/api/zimage/i2i` / `/api/zimage/inpaint` | Z-Image-Turbo |
| `POST /api/ltx2/t2v` / `/i2v` / `/flf` / `/ia2v` / `/keyframes` / `/v2a` / `/iclora` | LTX-2.3 動画生成(各種条件付け・音声・編集モード) |
| `POST /api/joyai/edit` | JoyAI-Edit-Plus によるマルチ参照画像編集 |
| `POST /api/mageflow/t2i` / `/api/mageflow/edit` | Mage-Flow T2I / Edit(別プロセスへのプロキシ、`exclusive` で排他選択) |
| `GET /api/mageflow/status` / `POST /api/mageflow/unload` | Mage-Flow ラッパーの状態確認 / 解放 |
| `POST /api/charsheet/generate` 他 | キャラクターシート生成ジョブ(8方向) |
| `POST /api/scene_angles/generate` 他 | シーンアングル生成ジョブ(1枚のシーン画像→カメラ8アングル、charsheetと同一パイプライン。2026-07-24追加) |
| `POST /api/tpose/generate` 他 | Tポーズ4ビュー生成ジョブ(1枚の画像→正面/背面/左前45度/右前45度。image-3dのマルチビュー入力・rig-serviceのTポーズ前提リグ向け。ビュー個別DL/ZIP・背景透過版の併産に対応。2026-07-26追加) |
| `POST /api/outpaint` | アウトペイント(画角拡張。既存インペイント流用、中央部は元画像ピクセル保証。2026-07-24追加) |
| `POST /api/prompt/enhance` / `/api/prompt/translate` | LLM プロンプト支援(要別途LLMサーバ) |
| `GET /api/status` | 全ファミリーのロード状態・VRAM |
| `POST /api/unload` | モデルの明示的アンロード |

生成系エンドポイントは GPU を同時1件のみ使用するグローバルロックで排他され、実行中は
`409` を返します。レスポンスには所要時間・ピークVRAM・使用パラメータ・seed 等の統一
メタデータが含まれます。

**詳細なパラメータ・レスポンス例・curl サンプルは `API_SPEC.md` を参照してください。**

## 環境変数

環境変数名は `DS_*` に統一されています(主要なもののみ抜粋。全項目は `API_SPEC.md` /
`core/config.py` 参照)。

| 変数名 | 既定値 | 説明 |
|---|---|---|
| `DS_COMFYUI_DIR` | `~/ComfyUI` | モデル重みの優先探索先(ComfyUI モデルディレクトリ) |
| `DS_OFFLOAD` | ファミリーごとに異なる(`auto`/`none`等) | `none`(全GPU常駐)/ `model`(model_cpu_offload)/ `group`(block-level group offload)/ `group_lowvram` |
| `DS_ATTN` | `default` | attention backend(`default`(SDPA)/ `xformers` 等)。Blackwell(sm_120)では `sage` 系は非対応 |
| `DS_COMPILE` | `0` | `1` で `torch.compile` を有効化 |
| `DS_QUANT` | `fp8-lightning` | Qwen系の量子化方式。`fp8-lightning` / `gguf-q4_k_m` 等 / `none`(bf16) |
| `DS_T2I_MODEL` | `2512` | T2I既定モデル(`2512` / `qwen-image`) |
| `DS_QWEN_TILED_VAE` | `1`(2026-07-24追加) | Qwen-Image系の共有VAE(t2i/i2i/edit/edit_angles系/controlnet/controlnet_inpaint)のencode/decodeを常時tiled化。`/api/outpaint`(1280×720キャンバス等)の大きな画像をVAEエンコードする際のOOM対策(CLAUDE.md 56番)。`0`で旧動作。Layered VAEは専用チェックポイント(8番)のため対象外 |
| `DS_VRAM_FREE_THRESHOLD_GB` / `DS_VRAM_LOW_THRESHOLD_GB` | GPU の空きVRAMに応じて調整 | オフロードモードの自動判定しきい値。お使いのGPUのVRAM容量に合わせて調整してください |
| `DS_FLUX2_PRECISION` | `bnb-4bit` | FLUX.2 の量子化精度(`bnb-4bit` / `bf16`) |
| `DS_ZIMAGE_PRECISION` | `bf16` | Z-Image の精度(`bf16` / `bnb-4bit`) |
| `DS_EDIT_TE_OFFLOAD` | `auto` | Edit系の text_encoder CPU退避(高解像度時のOOM対策) |
| `DS_LLM_URL` | `http://127.0.0.1:64652` | プロンプト支援機能が呼ぶ、OpenAI互換 `/v1/chat/completions` を持つローカルLLMサーバのURL(任意機能、無くても他機能に影響なし) |
| `DS_CHARSHEET_METHOD` | `bf16-group` | キャラクターシート生成の実装方式切替 |
| `DS_LTX2_CKPT_PATH` / `DS_LTX2_GEMMA_PATH` | ComfyUIモデルディレクトリ配下 | LTX-2.3 のチェックポイント・text_encoderパス |
| `DS_LTX2_OFFLOAD` | `group`(2026-07-22変更) | LTX-2.3 のオフロードモード(`none` / `group` / `auto`)。`auto`は空きVRAMの瞬間値で判定するため他ファミリー切替直後にOOMしやすく非推奨(CLAUDE.md 49番)。従来noneモードは長尺(241f超)や768×448級でVAEデコードOOMがあったが、tiled化(下記`DS_LTX2_TILED_DECODE`)により解消: noneで361f(15秒)・upscaleとも生成可、短尺はgroup比10倍超高速(96GB専有時の選択肢。既定はgroupのまま) |
| `DS_LTX2_TE_QUANT` | `fp8`(2026-07-22変更) | LTX-2.3 text_encoder の量子化(`none` / `fp8` / `nf4`)。`nf4`はGoogle製QAT版の別チェックポイントで品質A/B未確定のため既定に非採用(CLAUDE.md 49番) |
| `DS_LTX2_TILED_DECODE` | `1`(2026-07-23追加) | LTX-2.3 の全VAEデコード経路(全モード、upscale有無問わず)を常時tiled化。noneモード長尺OOMの正体だった一括VAEデコード(23.8GiB単発要求)を解消(CLAUDE.md 52番)。`0`で旧動作 |
| `DS_MAGEFLOW_URL` | `http://127.0.0.1:8602` | 本体サーバが Mage-Flow ラッパーサービスへ接続するURL(ポートはハードコードしない。ラッパー未起動時は `/api/mageflow/*` 生成系が502) |
| `DS_MAGEFLOW_PORT` / `DS_MAGEFLOW_HOST` | `8602` / `127.0.0.1` | `run_mageflow.sh` がラッパーサービスを起動するポート/ホスト |
| `DS_TERMINAL_PROGRESS` | `0`(2026-07-24追加) | `1` でサーバ起動ターミナル(stderr)へ生成中の進捗バーを表示(`\r`上書き、完了時は確定行)。charsheet/scene_angles は「direction i/n」も表示。ON時、diffusers パイプライン自前の tqdm(denoiseの`25%\|██▌\|`表示)は `set_progress_bar_config(disable=True)` で抑制(HFダウンロードのtqdmは対象外)。uvicornログとの行混在あり。詳細は CLAUDE.md 55番参照 |

VRAM しきい値はお使いの GPU の空き VRAM に合わせて調整してください。値を大きくしすぎると
本来オフロード不要な構成までオフロードされ低速になり、小さくしすぎると OOM のリスクが
上がります。

**LTX-2.3 `group` オフロードのホスト RAM ガードについて**: `group` モード(既定)は
transformer(bf16 約35.4GB)を一時的にホスト RAM 上に構築してから GPU への block-level
転送に登録する設計のため、**パイプラインの初回ロード時**(サーバ起動後の最初のLTX-2.3
生成リクエスト、または `/api/unload` や他ファミリーへの切替でLTX-2.3が一度アンロード
された後の次回ロード時)に、空きホスト RAM(`/proc/meminfo` の `MemAvailable`)が
`DS_LTX2_GROUP_OFFLOAD_MIN_RAM_GB`(既定 40.0GB)を下回っていると、システムフリーズを
避けるためロード自体を明確なエラーメッセージ付きで中止します(旧実装がこのガードを
持たず実際にフリーズ・強制終了を招いた経緯は `CLAUDE.md` 17番参照)。**このガードは
動画の解像度・フレーム数・アップスケール有無とは無関係**です(一度ロードされた
パイプラインはそれ以降のリクエストで再チェックされません)。ブラウザ・IDE 等
他アプリのメモリ使用量が大きいマシンでは、LTX-2.3 生成の**最初の1回目**がこのエラーで
失敗することがあります。その場合は他アプリを閉じるかホストRAMの空きを確保してから
再試行してください(`free -h` の `available` 列で確認可能)。閾値を下げる
(`DS_LTX2_GROUP_OFFLOAD_MIN_RAM_GB`)ことも可能ですが、フリーズのリスクとのトレード
オフのため推奨しません。

## 使用モデルとライセンス

**モデルの重みは本リポジトリに含まれません。** 各モデルは初回利用時に Hugging Face Hub
等から取得され、それぞれ配布元が定めるライセンス・利用規約に従います。以下は本サーバが
呼び出すモデルの一覧です。**正確な最新のライセンス条項は必ず各モデルカードを参照して
ください**(下表は本ドキュメント作成時点の確認結果であり、将来変更される可能性があります)。

| モデル | 配布元 | ライセンス(要各モデルカード確認) |
|---|---|---|
| Qwen-Image | `Qwen/Qwen-Image` | Apache License 2.0 |
| Qwen-Image-2512 | `Qwen/Qwen-Image-2512` | Apache License 2.0 |
| Qwen-Image-Edit-Plus(2511) | `Qwen/Qwen-Image-Edit-2511` 系 | Apache License 2.0(各モデルカードを参照) |
| Qwen-Image-Lightning LoRA | `lightx2v/Qwen-Image-Lightning`、`lightx2v/Qwen-Image-2512-Lightning` | Apache License 2.0 |
| Qwen-Image-ControlNet-Union | `InstantX/Qwen-Image-ControlNet-Union` | Apache License 2.0 |
| FLUX.2-dev | `black-forest-labs/FLUX.2-dev` | FLUX Non-Commercial License(Black Forest Labs。**商用利用不可**、要モデルカード確認・利用規約への同意が必要) |
| Z-Image-Turbo | `Tongyi-MAI/Z-Image-Turbo` | Apache License 2.0 |
| LTX-2.3 | `Lightricks/LTX-2` | LTX-2 Community License Agreement(要モデルカード確認、独自ライセンス) |
| Gemma 3(LTX-2.3 の text encoder) | Google | Gemma 利用規約(要モデルカード確認) |
| JoyAI-Image-Edit-Plus | `jdopensource/JoyAI-Image-Edit-Plus-Diffusers` | Apache License 2.0 |
| MergeGreen IC-LoRA(LTX-2.3用) | `siraxe/MergeGreen_IC-lora_ltx2.3` | Apache License 2.0 |
| Mage-Flow(T2I: Base/RL/Turbo) | `microsoft/Mage-Flow-Base` / `microsoft/Mage-Flow` / `microsoft/Mage-Flow-Turbo` | MIT(コード・重みとも。要モデルカード確認) |
| Mage-Flow-Edit(Base/RL/Turbo) | `microsoft/Mage-Flow-Edit-Base` / `microsoft/Mage-Flow-Edit` / `microsoft/Mage-Flow-Edit-Turbo` | MIT(同上) |

**本リポジトリ自体(コード)は Apache License 2.0 で公開しています**が、上記モデルの
重みを利用して生成したコンテンツやモデルの再配布については、各モデルのライセンス条項が
別途適用されます。特に **FLUX.2-dev は非商用ライセンス**である点にご注意ください。
商用利用を検討する場合は、Qwen-Image 系・Z-Image-Turbo・JoyAI-Edit-Plus 等の Apache 2.0
モデルの使用を推奨します。

## 既知の制約

- 生成系エンドポイントは GPU 排他制御のため同時に1件しか実行できません(実行中は
  409 を返します)。
- 一部のモデルファミリー(Qwen-Image系 / FLUX.2 / Z-Image-Turbo / LTX-2.3 /
  JoyAI-Edit-Plus)は VRAM 同時常駐不可とみなし、ファミリー切替時に自動アンロードします。
  切替には数十秒程度かかる場合があります。
- GGUF 量子化された transformer には LoRA を適用できません(`diffusers`/`peft` の
  現行実装の制約)。Lightning LoRA を使う場合は fp8-lightning 方式を利用してください。
- LTX-2.3(動画生成)はローカルにチェックポイントファイルが配置されている前提です
  (Hugging Face Hub からの自動ダウンロードは行いません。config/tokenizer のみ Hub から
  取得します)。
- 48GB クラスの VRAM でも、オフロード設定・解像度・生成モードの組み合わせによっては
  OOM が発生する場合があります。環境変数のオフロード関連しきい値を調整してください。
- FLF(First-Last-Frame)補間の滑らかさは入力2枚の意味的な近さに強く依存します。

## ドキュメント

- `API_SPEC.md` — 全エンドポイントの詳細なパラメータ・レスポンス例・curl サンプル
- `LICENSE` — 本リポジトリのライセンス(Apache License 2.0)
