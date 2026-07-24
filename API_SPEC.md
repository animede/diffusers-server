# API_SPEC.md — DiffusersWEBUI API仕様書

このドキュメントは `diffusers-server`(DiffusersWEBUI)が提供する HTTP API の詳細仕様である。
稼働中サーバの `GET /openapi.json`(FastAPI自動生成)と各エンドポイントの実装
(`app.py` / `families/*/generate.py` / `apps/charsheet/*`)を突き合わせて作成した。
README.md の「API 一覧」は概要のみを記載し、詳細は本ドキュメントを参照する構成に統一する。

- 対象バージョン: git `main` 時点(2026-07-21、コミット `0a434c1` 以降)
- ベースURL: `http://localhost:8601`(既定ポート、`--port` で変更可)
- 認証: なし(ローカル専用サーバ、外部公開を想定しない)

## 目次

1. [概要](#概要)
2. [Qwen-Image系](#qwen-image系)
3. [FLUX.2](#flux2)
4. [Z-Image-Turbo](#z-image-turbo)
5. [LTX-2.3(動画+音声生成)](#ltx-23動画音声生成)
6. [JoyAI-Image-Edit-Plus](#joyai-image-edit-plus)
7. [Mage-Flow(別プロセスプロキシ)](#mage-flow別プロセスプロキシ)
8. [charsheet](#charsheet)
9. [ユーティリティ](#ユーティリティ)
10. [管理系](#管理系)
11. [エラー仕様](#エラー仕様)
12. [環境変数の影響](#環境変数の影響)

---

## 概要

### 共通の排他制御

生成系エンドポイント(画像/動画を実際に生成するもの)は、全ファミリー共通のGPUロック
(`core.gpu.generation_lock`、プロセス内で同時1件のみ)で排他される。別の生成が実行中の
場合は非ブロッキングで即座に **409** を返す(ポーリングやリトライは呼び出し側の責任)。

charsheetのジョブ生成(`/api/charsheet/generate` とその派生)は上記GPUロックに加えて
「charsheetジョブ自体の同時1件制御」(`current_job_id`)も持つため、charsheetジョブ実行中は
他のcharsheet操作(`/refine` `/remove_bg` `/undo`)も409になる。

LLMプロンプト支援(`/api/prompt/enhance` `/api/prompt/translate`)・`/api/canny`・
`/api/remove_bg`・`/api/status`・`/api/progress` はGPUを使わない(または排他ロックを取らない)
ため、画像生成中でも並行して呼べる。

### 生成レスポンスの統一メタデータ

画像/動画を生成する全エンドポイントは、JSONレスポンスに以下のフィールドを(ファミリー・
モードごとに差はあるが)共通して含む。

| フィールド | 型 | 説明 |
|---|---|---|
| `mode` | string | 内部モード名(例: `t2i`, `edit`, `flux2_t2i`, `ltx2_t2v`, `joyai_edit`) |
| `elapsed_s` | float | 生成処理の所要時間(秒)。モデルロードが発生した場合はロード時間を含む |
| `peak_vram_gb` | float \| null | 生成中のピークVRAM使用量(GB)。`torch.cuda.max_memory_allocated()` 基準 |
| `seed` | int \| null | 実際に使用したseed(未指定/`-1`指定時はランダムseedが採番されここに入る) |
| `image_url` / `video_url` | string | `outputs/` 配下の生成物への相対URL(例: `/outputs/t2i_20260721_120000_abcdef12.png`) |
| （各パラメータ） | - | リクエストで受け取ったパラメータのエコーバック(丸め・自動調整後の実効値) |

画像系は `image_url`、動画系(LTX-2.3)は `video_url` / `video_only_url` / `audio_url` /
`has_audio` を持つ。Layered は複数レイヤーのため `layer_image_urls`(配列)と
`composite_image_url` / `image_url`(合成プレビュー)を持つ。詳細は各エンドポイントの節を参照。

### outputs/ の静的配信

生成物は `outputs/` ディレクトリに保存され、`/outputs/{filename}` として静的配信される
(`app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR))`)。レスポンスの `image_url` /
`video_url` 等はこの形式の相対パスであり、ベースURLと結合してそのままブラウザ表示・
ダウンロードに使える。

### 進捗ポーリング(`GET /api/progress`)

全ファミリー共通ロックにより生成は同時1件のため、進捗状態もグローバルに1つ持てば足りる。
生成リクエストを投げた直後から、`GET /api/progress` を500ms間隔程度でポーリングすることで
進捗(ロード中/denoiseステップ/デコード中)を取得できる。認証・排他不要で、生成中でも
呼び放題(GPUを使わない読み取り専用)。レスポンス形式は[管理系](#get-apiprogress)を参照。

### アップロード画像のEXIF処理

全ての画像アップロードエンドポイントは `ImageOps.exif_transpose()` でEXIF Orientationを
実ピクセルに適用してから処理する(スマホ撮影画像等の90度回転バグ対策、共通ヘルパー
`_open_upload_image()` / charsheet側は同等処理を個別実装)。呼び出し側でEXIF回転を
気にする必要はない。

---

## Qwen-Image系

共通のネガティブプロンプト既定値: `NEGATIVE_PROMPT_DEFAULT`(T2I/I2I/Edit共通)
```
lowres, bad anatomy, bad hands, missing fingers, extra digits, fewer fingers, cropped,
worst quality, low quality, jpeg artifacts, signature, watermark, username, blurry
```
ControlNet/Inpaintの既定ネガティブプロンプトは `" "`(半角スペース1文字)。

### POST /api/t2i

T2I(Text-to-Image)。Content-Type: `application/json`。

**パラメータ**

| 名前 | 型 | 必須/任意 | 既定値 | 説明・制約 |
|---|---|---|---|---|
| `prompt` | string | 必須 | - | 生成プロンプト |
| `negative_prompt` | string | 任意 | `NEGATIVE_PROMPT_DEFAULT` | ネガティブプロンプト |
| `width` | int | 任意 | `1328` | 16の倍数に自動丸め |
| `height` | int | 任意 | `1328` | 16の倍数に自動丸め |
| `steps` | int | 任意 | `30` | denoiseステップ数。Lightning有効時は4/8以外を指定すると4へ強制 |
| `cfg` | float | 任意 | `4.0` | true_cfg_scale。Lightning有効時は1.0に強制(CFG無効) |
| `seed` | int | 任意 | `-1` | `-1`はランダム。実使用seedはレスポンス`seed`で確認 |
| `lightning` | bool | 任意 | `false` | Lightning LoRA(4/8steps高速化)を有効化するか |
| `shift` | float \| null | 任意 | `null` | scheduler shift値。未指定時は自動 |
| `model` | string \| null | 任意 | `null`(現在ロード中のモデルを維持) | `"2512"`(既定相当、Apache 2.0)\| `"qwen-image"`(無印)。旧値`"2512-4bit"`は後方互換で`"2512"`扱い |

**レスポンス例**

```json
{
  "mode": "t2i",
  "prompt": "a cat on a chair",
  "negative_prompt": "lowres, ...",
  "width": 1328,
  "height": 1328,
  "steps": 4,
  "cfg": 1.0,
  "shift": 3.0,
  "seed": 42,
  "lightning": true,
  "elapsed_s": 11.2,
  "peak_vram_gb": 34.83,
  "image_url": "/outputs/t2i_20260721_120000_abcdef12.png",
  "lightning_requested": true,
  "lightning_unavailable_reason": null,
  "model": "2512",
  "quant": "fp8-lightning"
}
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/t2i \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a cat on a chair","width":1024,"height":1024,"lightning":true,"seed":42}'
```

**特記事項**

- `model` を現在ロード中のモデルと異なる値に指定すると、T2Iグループを自動unloadしてから
  切り替える(fuse込みで数十秒〜数分かかる)。
- fp8-lightning構成ではLightning LoRAは重みにfuse済みのため常時適用され、`lightning`
  チェックボックスはsteps/cfgプリセット切替として機能する。
- 48GB専有環境では既定解像度1328²前後でVRAM逼迫時、text_encoderのCPU退避
  (`DS_EDIT_TE_OFFLOAD`)が自動的に働く(`should_offload_edit_text_encoder()`)。

---

### POST /api/i2i

I2I(Image-to-Image)。Content-Type: `multipart/form-data`。

**パラメータ**

| 名前 | 型 | 必須/任意 | 既定値 | 説明・制約 |
|---|---|---|---|---|
| `image` | file | 必須 | - | 入力画像 |
| `prompt` | string | 必須 | - | プロンプト |
| `negative_prompt` | string | 任意 | `NEGATIVE_PROMPT_DEFAULT` | - |
| `strength` | float | 任意 | `0.6` | denoise強度(0〜1)。小さいほど原画像を維持 |
| `steps` | int | 任意 | `30` | - |
| `cfg` | float | 任意 | `4.0` | - |
| `seed` | int | 任意 | `-1` | - |
| `lightning` | bool | 任意 | `false` | - |
| `shift` | float \| null | 任意 | `null` | - |
| `model` | string \| null | 任意 | `null` | `"2512"` \| `"qwen-image"`(2026-07-18からI2Iも両対応) |

**レスポンス例**

```json
{
  "mode": "i2i",
  "prompt": "make it winter",
  "strength": 0.6,
  "steps": 30,
  "cfg": 4.0,
  "seed": 123,
  "lightning": false,
  "elapsed_s": 18.3,
  "peak_vram_gb": 34.82,
  "image_url": "/outputs/i2i_20260721_120500_11223344.png",
  "input_size": [1024, 1024],
  "te_offload": true,
  "quant": "fp8-lightning",
  "lightning_requested": false,
  "lightning_unavailable_reason": null
}
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/i2i \
  -F "image=@input.png" -F "prompt=make it winter" -F "strength=0.6"
```

**特記事項**: `te_offload` は実際にtext_encoderをCPU退避したかどうか(`DS_EDIT_TE_OFFLOAD`、
既定`auto`は入力画像解像度640²以上で有効化)。この修正により、クリーンな状態からの単独呼び出しでも
VAE decode時のCUDA OOMが解消されている(旧既知バグ、README「既知の制約」参照)。

---

### POST /api/edit

Edit(参照画像1〜3枚を用いた編集・合成、`QwenImageEditPlusPipeline`)。
Content-Type: `multipart/form-data`。

**パラメータ**

| 名前 | 型 | 必須/任意 | 既定値 | 説明・制約 |
|---|---|---|---|---|
| `images` | file[] | 必須 | - | 参照画像1〜3枚(`MAX_EDIT_IMAGES`超過は400) |
| `prompt` | string | 必須 | - | 編集指示 |
| `negative_prompt` | string | 任意 | `NEGATIVE_PROMPT_DEFAULT` | - |
| `steps` | int | 任意 | `30` | Lightning有効時は0〜12の範囲外なら4へ補正 |
| `cfg` | float | 任意 | `4.0` | Lightning有効時は1.0に強制 |
| `seed` | int | 任意 | `-1` | - |
| `lightning` | bool | 任意 | `true` | Editは既定でLightning有効(T2Iとは既定が異なる) |
| `shift` | float \| null | 任意 | `null` | - |
| `width` | int \| null | 任意 | `null`(参照画像から自動推定、概ね1024²相当) | 48GB環境では明示指定推奨(下記参照) |
| `height` | int \| null | 任意 | `null` | 同上 |

**レスポンス例**

```json
{
  "mode": "edit",
  "prompt": "change the shirt color to blue",
  "steps": 4,
  "cfg": 1.0,
  "shift": 3.0,
  "width": 640,
  "height": 640,
  "seed": 12345,
  "lightning": true,
  "elapsed_s": 10.97,
  "peak_vram_gb": 34.98,
  "image_url": "/outputs/edit_20260721_121000_aabbccdd.png",
  "num_reference_images": 1,
  "edit_transformer_fallback": false,
  "quant": "fp8-lightning",
  "lightning_requested": true,
  "lightning_unavailable_reason": null,
  "te_offload": true
}
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/edit \
  -F "images=@char.png" -F "prompt=change the shirt color to blue" \
  -F "height=640" -F "width=640"
```

**特記事項**

- **`width`/`height` 未指定 → 約1024²自動推定**。48GB専有環境ではVAE decode時の一時ピークで
  以前OOMしていたが、text_encoder CPU退避(`DS_EDIT_TE_OFFLOAD`、既定`auto`)実装により解消済み
  (実測: 未指定時 `te_offload=true`・ピークVRAM 34.98GB)。逼迫時は `height=640&width=640` 等の
  縮小指定も引き続き可能(実測39.0GB)。
- `edit_transformer_fallback`: GGUF量子化フォールバックが発生したかどうか。
- 参照画像は内部で `QwenImageEditPlusPipeline` に必ずリスト形式で渡される(単体でも1要素リスト)。

---

### POST /api/controlnet

ControlNet(Union、既定 canny)。Content-Type: `multipart/form-data`。

**パラメータ**

| 名前 | 型 | 必須/任意 | 既定値 | 説明・制約 |
|---|---|---|---|---|
| `control_image` | file | 必須 | - | 制御画像 |
| `prompt` | string | 必須 | - | - |
| `negative_prompt` | string | 任意 | `" "` | - |
| `control_type` | string | 任意 | `"canny"` | 制御タイプ(ControlNet Union対応種別) |
| `controlnet_conditioning_scale` | float | 任意 | `1.0` | 制御強度 |
| `steps` | int | 任意 | `30` | - |
| `cfg` | float | 任意 | `4.0` | - |
| `seed` | int | 任意 | `-1` | - |
| `width` | int \| null | 任意 | `null`(制御画像サイズに自動追従、16の倍数丸め) | - |
| `height` | int \| null | 任意 | `null` | - |

**レスポンス例**

```json
{
  "mode": "controlnet",
  "prompt": "a photo of a room",
  "negative_prompt": " ",
  "control_type": "canny",
  "controlnet_conditioning_scale": 1.0,
  "width": 624,
  "height": 944,
  "steps": 30,
  "cfg": 4.0,
  "seed": 42,
  "elapsed_s": 2.0,
  "peak_vram_gb": 42.9,
  "image_url": "/outputs/controlnet_20260721_121500_9988aabb.png"
}
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/controlnet \
  -F "control_image=@edges.png" -F "prompt=a photo of a room" -F "control_type=canny"
```

**特記事項**: ネイティブ解像度(自動追従)ではVAE decode時の一時ピークでOOMするリスクが
未対応のまま残っている(Editほど優先度が高くない)。逼迫時は `width`/`height` を縮小指定する。
ControlNetはGGUF量子化transformerとも組み合わせて動作する。

---

### POST /api/inpaint

ControlNet Inpainting。Content-Type: `multipart/form-data`。

**パラメータ**

| 名前 | 型 | 必須/任意 | 既定値 | 説明・制約 |
|---|---|---|---|---|
| `image` | file | 必須 | - | 元画像 |
| `mask_image` | file | 必須 | - | マスク画像(白=再生成領域) |
| `prompt` | string | 必須 | - | - |
| `negative_prompt` | string | 任意 | `" "` | - |
| `controlnet_conditioning_scale` | float | 任意 | `1.0` | - |
| `steps` | int | 任意 | `30` | - |
| `cfg` | float | 任意 | `4.0` | - |
| `seed` | int | 任意 | `-1` | - |

**レスポンス例**

```json
{
  "mode": "inpaint",
  "prompt": "add a hat",
  "negative_prompt": " ",
  "controlnet_conditioning_scale": 1.0,
  "width": 1024,
  "height": 1024,
  "steps": 30,
  "cfg": 4.0,
  "seed": 7,
  "elapsed_s": 5.1,
  "peak_vram_gb": 41.2,
  "image_url": "/outputs/inpaint_20260721_122000_deadbeef.png"
}
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/inpaint \
  -F "image=@photo.png" -F "mask_image=@mask.png" -F "prompt=add a hat"
```

---

### POST /api/canny

補助機能(GPU不使用、排他不要)。アップロード画像からCannyエッジ画像を生成する。
Content-Type: `multipart/form-data`。

**パラメータ**

| 名前 | 型 | 必須/任意 | 既定値 | 説明 |
|---|---|---|---|---|
| `image` | file | 必須 | - | 入力画像 |
| `low_threshold` | int | 任意 | `100` | cv2.Canny の下限しきい値 |
| `high_threshold` | int | 任意 | `200` | cv2.Canny の上限しきい値 |

**レスポンス例**

```json
{ "image_url": "/outputs/canny_20260721_122500_11112222.png" }
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/canny -F "image=@photo.png" -F "low_threshold=100" -F "high_threshold=200"
```

**特記事項**: `cv2`(opencv)が利用不可の場合 `501` を返す。

---

### POST /api/layered

Layered(RGBAレイヤー分解)。Content-Type: `multipart/form-data`。

**パラメータ**

| 名前 | 型 | 必須/任意 | 既定値 | 説明・制約 |
|---|---|---|---|---|
| `image` | file | 必須 | - | RGBA画像(内部で`.convert("RGBA")`) |
| `prompt` | string | 任意 | `""` | - |
| `negative_prompt` | string | 任意 | `""` | - |
| `layers` | int | 任意 | `4` | 1〜16の範囲(範囲外は400) |
| `resolution` | int | 任意 | `640` | `640` または `1024` のみ(それ以外は400) |
| `steps` | int \| null | 任意 | `null`(Lightning適用有無で自動決定) | - |
| `cfg` | float \| null | 任意 | `null` | - |
| `shift` | float \| null | 任意 | `null` | - |
| `seed` | int | 任意 | `-1` | - |
| `cfg_normalize` | bool | 任意 | `false` | - |
| `use_en_prompt` | bool | 任意 | `true` | - |

**レスポンス例**

```json
{
  "mode": "layered",
  "prompt": "",
  "layers": 4,
  "resolution": 640,
  "steps": 4,
  "cfg": 1.0,
  "shift": 3.0,
  "cfg_normalize": false,
  "use_en_prompt": true,
  "seed": 12345,
  "quant": "fp8-lightning",
  "lightning_merged": true,
  "elapsed_s": 9.0,
  "peak_vram_gb": 42.3,
  "layer_image_urls": [
    "/outputs/layered_20260721_123000_abcd1234_layer0.png",
    "/outputs/layered_20260721_123000_abcd1234_layer1.png",
    "/outputs/layered_20260721_123000_abcd1234_layer2.png",
    "/outputs/layered_20260721_123000_abcd1234_layer3.png"
  ],
  "composite_image_url": "/outputs/layered_20260721_123000_abcd1234_composite.png",
  "image_url": "/outputs/layered_20260721_123000_abcd1234_composite.png"
}
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/layered -F "image=@char_rgba.png" -F "layers=4" -F "resolution=640"
```

**特記事項**: `image_url` は `composite_image_url` と同一値(他ファミリーとのレスポンス
キー互換のため複製)。`layers=4, resolution=640` 前後の既定運用は48GB専有環境でCUDA OOMが
未解決のまま残っている(`layers=1`のみ動作確認済み、継続調査中)。

---

## FLUX.2

FLUX.2-dev(bnb-4bit)単一モデル構成。旧第2モデル ecocoro は2026-07-19に廃止。
qwen_image系とVRAM同時常駐不可のため相互排他(生成時に自動unload)。

### POST /api/flux2/t2i

Content-Type: `application/json`。

| 名前 | 型 | 必須/任意 | 既定値 | 説明 |
|---|---|---|---|---|
| `prompt` | string | 必須 | - | - |
| `steps` | int | 任意 | `28` | - |
| `guidance_scale` | float | 任意 | `4.0` | - |
| `width` | int | 任意 | `1024` | - |
| `height` | int | 任意 | `1024` | - |
| `seed` | int | 任意 | `-1` | - |
| `model` | string \| null | 任意 | `null` | `"dev"` のみ有効。`"ecocoro"` は400(廃止済み) |

**レスポンス例**

```json
{
  "mode": "flux2_t2i",
  "prompt": "a mountain landscape",
  "width": 1024,
  "height": 1024,
  "steps": 28,
  "cfg": 4.0,
  "guidance_scale": 4.0,
  "seed": 1,
  "elapsed_s": 56.66,
  "peak_vram_gb": 33.84,
  "image_url": "/outputs/flux2_t2i_20260721_123500_11119999.png",
  "model": "flux2-dev",
  "quant": "bnb-4bit",
  "offload_mode": "model_cpu_offload"
}
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/flux2/t2i \
  -H "Content-Type: application/json" -d '{"prompt":"a mountain landscape","steps":28}'
```

---

### POST /api/flux2/i2i

Content-Type: `multipart/form-data`。

| 名前 | 型 | 必須/任意 | 既定値 | 説明・制約 |
|---|---|---|---|---|
| `images` | file[] | 必須 | - | 参照画像1〜10枚(`MAX_REF_IMAGES`=10超過は400) |
| `prompt` | string | 必須 | - | - |
| `steps` | int | 任意 | `28` | - |
| `guidance_scale` | float | 任意 | `4.0` | - |
| `width` | int | 任意 | `1024` | - |
| `height` | int | 任意 | `1024` | - |
| `seed` | int | 任意 | `-1` | - |
| `model` | string \| null | 任意 | `null` | `"dev"` のみ有効 |

**レスポンス例**

```json
{
  "mode": "flux2_i2i",
  "prompt": "change the jacket to red",
  "width": 1024,
  "height": 1024,
  "steps": 28,
  "cfg": 4.0,
  "guidance_scale": 4.0,
  "seed": 5,
  "elapsed_s": 128.6,
  "peak_vram_gb": 19.5,
  "image_url": "/outputs/flux2_i2i_20260721_124000_22223333.png",
  "num_reference_images": 1,
  "model": "flux2-dev",
  "quant": "bnb-4bit",
  "offload_mode": "model_cpu_offload"
}
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/flux2/i2i -F "images=@ref.png" -F "prompt=change the jacket to red"
```

**特記事項**: `model="ecocoro"` を指定すると400（「ecocoroは廃止されました」）。

---

## Z-Image-Turbo

`Tongyi-MAI/Z-Image-Turbo`。蒸留モデルのため既定 `steps=8` / `guidance_scale=0.0`(CFGなし)。
`guidance_scale>1` にするとnegative_promptが効くが、denoiseが2パスになり約2倍遅くなる。
qwen_image・flux2とVRAM同時常駐不可のため3方向相互排他。

### POST /api/zimage/t2i

Content-Type: `application/json`。

| 名前 | 型 | 必須/任意 | 既定値 | 説明 |
|---|---|---|---|---|
| `prompt` | string | 必須 | - | - |
| `negative_prompt` | string | 任意 | `""` | - |
| `steps` | int | 任意 | `8` | - |
| `guidance_scale` | float | 任意 | `0.0` | `>1`でnegative_promptが有効、denoise約2倍遅くなる |
| `width` | int | 任意 | `1024` | - |
| `height` | int | 任意 | `1024` | - |
| `seed` | int | 任意 | `-1` | - |

**レスポンス例**

```json
{
  "mode": "zimage_t2i",
  "prompt": "a red bicycle",
  "negative_prompt": "",
  "width": 1024,
  "height": 1024,
  "steps": 8,
  "cfg": 0.0,
  "guidance_scale": 0.0,
  "seed": 12345,
  "elapsed_s": 4.0,
  "peak_vram_gb": 21.67,
  "image_url": "/outputs/zimage_t2i_20260721_124500_55556666.png",
  "model": "z-image-turbo",
  "quant": "bf16",
  "offload_mode": "none"
}
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/zimage/t2i -H "Content-Type: application/json" -d '{"prompt":"a red bicycle"}'
```

---

### POST /api/zimage/i2i

Content-Type: `multipart/form-data`。

| 名前 | 型 | 必須/任意 | 既定値 | 説明 |
|---|---|---|---|---|
| `image` | file | 必須 | - | - |
| `prompt` | string | 必須 | - | - |
| `negative_prompt` | string | 任意 | `""` | - |
| `strength` | float | 任意 | `0.6` | - |
| `steps` | int | 任意 | `8` | - |
| `guidance_scale` | float | 任意 | `0.0` | - |
| `width` | int | 任意 | `1024` | - |
| `height` | int | 任意 | `1024` | - |
| `seed` | int | 任意 | `-1` | - |

レスポンスは `/api/zimage/t2i` と同一キー構成 + `strength`。`mode` は `"zimage_i2i"`。

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/zimage/i2i -F "image=@input.png" -F "prompt=make it sunset" -F "strength=0.6"
```

---

### POST /api/zimage/inpaint

Content-Type: `multipart/form-data`。「Edit」の代替(公式Z-Image-Editチェックポイント未公開のため
マスクインペイントとして提供)。白マスク画素 = 再生成領域。

| 名前 | 型 | 必須/任意 | 既定値 | 説明 |
|---|---|---|---|---|
| `image` | file | 必須 | - | - |
| `mask_image` | file | 必須 | - | 白=再生成領域 |
| `prompt` | string | 必須 | - | - |
| `negative_prompt` | string | 任意 | `""` | - |
| `strength` | float | 任意 | `1.0` | i2iと既定値が異なる点に注意 |
| `steps` | int | 任意 | `8` | - |
| `guidance_scale` | float | 任意 | `0.0` | - |
| `width` | int | 任意 | `1024` | - |
| `height` | int | 任意 | `1024` | - |
| `seed` | int | 任意 | `-1` | - |

`mode` は `"zimage_inpaint"`。他は `/api/zimage/i2i` と同様のキー構成。

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/zimage/inpaint \
  -F "image=@photo.png" -F "mask_image=@mask.png" -F "prompt=add sunglasses"
```

**特記事項**: T2I/I2I/Inpaintは単一baseパイプラインのcomponents参照共有のため、
どれか1つロードすれば追加VRAM消費なしで他モードも使える(実測ピークいずれも21.7GB)。

---

## LTX-2.3(動画+音声生成)

ComfyUI形式 scaled-fp8 単一チェックポイント(蒸留版、28GB)+ ローカルGemma 3 12B
text_encoder。qwen_image・flux2・z_imageとVRAM同時常駐不可のため4方向相互排他。

蒸留モデルの既定値は `steps=8` / `guidance_scale=1.0`(CFG判定は`guidance_scale>1`のため
`1.0`は「無効」を意味する。`negative_prompt`は実質効かない）。`num_frames`は8n+1に丸められる。
`width`/`height`は32の倍数を推奨(内部実装がその前提)。

### POST /api/ltx2/t2v

Content-Type: `application/json`。

| 名前 | 型 | 必須/任意 | 既定値 | 説明・制約 |
|---|---|---|---|---|
| `prompt` | string | 必須 | - | - |
| `negative_prompt` | string | 任意 | `""` | 蒸留モデルでは実質無効 |
| `width` | int | 任意 | `512` | 32の倍数推奨 |
| `height` | int | 任意 | `320` | 32の倍数推奨 |
| `num_frames` | int | 任意 | `25` | 8n+1丸め |
| `fps` | int | 任意 | `24` | - |
| `steps` | int | 任意 | `8` | - |
| `guidance_scale` | float | 任意 | `1.0` | `>1`でCFG有効(蒸留モデルでは通常不要) |
| `seed` | int | 任意 | `-1` | - |
| `audio_out` | string | 任意 | `"on"` | `"on"`(生成音声mux) \| `"off"`(無音mp4) |
| `upscale` | int | 任意 | `0` | `0`(無効) \| `1`(latent upsamplerで2x空間アップスケール) |

**レスポンス例**

```json
{
  "mode": "ltx2_t2v",
  "prompt": "a fox walking through a snowy forest",
  "negative_prompt": "",
  "width": 512,
  "height": 320,
  "num_frames": 25,
  "fps": 24,
  "steps": 8,
  "guidance_scale": 1.0,
  "seed": 12345,
  "elapsed_s": 2.0,
  "peak_vram_gb": 69.98,
  "video_url": "/outputs/ltx2_t2v_20260721_125000_9a8b7c6d.mp4",
  "video_only_url": "/outputs/ltx2_t2v_20260721_125000_9a8b7c6d.mp4",
  "audio_url": "/outputs/ltx2_t2v_20260721_125000_9a8b7c6d.wav",
  "has_audio": true,
  "audio_out": "on",
  "upscale": false,
  "model": "ltx-2.3-22b-distilled-fp8",
  "offload_mode": "none"
}
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/ltx2/t2v \
  -H "Content-Type: application/json" \
  -d '{"prompt":"a fox walking through a snowy forest","num_frames":25}'
```

**特記事項**

- `video_url` はmux成功時はmux済み(音声込み)mp4、失敗/`audio_out=off`時は映像のみmp4と同じ値。
  `video_only_url` は常に映像のみのmp4。
- `upscale=1` 指定時、レスポンスの `width`/`height` は要求値の2倍(実際の出力解像度)。
- 48GB環境では `DS_LTX2_OFFLOAD=group` でtransformerのみblock-level group offloadする
  (実測ピーク約35GB、none比で生成が大幅に遅くなるが出力はピクセル完全一致)。

---

### POST /api/ltx2/i2v

Content-Type: `multipart/form-data`。

| 名前 | 型 | 必須/任意 | 既定値 | 説明 |
|---|---|---|---|---|
| `image` | file | 必須 | - | - |
| `prompt` | string | 必須 | - | - |
| `negative_prompt` | string | 任意 | `""` | - |
| `width` | int | 任意 | `512` | - |
| `height` | int | 任意 | `320` | - |
| `num_frames` | int | 任意 | `25` | 8n+1丸め |
| `fps` | int | 任意 | `24` | - |
| `steps` | int | 任意 | `8` | - |
| `guidance_scale` | float | 任意 | `1.0` | - |
| `seed` | int | 任意 | `-1` | - |
| `audio_out` | string | 任意 | `"on"` | `"on"` \| `"off"` |
| `upscale` | int | 任意 | `0` | `0` \| `1` |

レスポンスは `/api/ltx2/t2v` と同一キー構成、`mode` は `"ltx2_i2v"`。

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/ltx2/i2v \
  -F "image=@first.png" -F "prompt=the character starts walking" -F "num_frames=25"
```

---

### POST /api/ltx2/flf

FLF(First-Last-Frame)。最初と最後のフレーム画像を指定し、間を補間する動画を生成する。
Content-Type: `multipart/form-data`。

| 名前 | 型 | 必須/任意 | 既定値 | 説明 |
|---|---|---|---|---|
| `first_image` | file | 必須 | - | 先頭フレーム画像 |
| `last_image` | file | 必須 | - | 末尾フレーム画像 |
| `prompt` | string | 必須 | - | - |
| `negative_prompt` | string | 任意 | `""` | - |
| `width` | int | 任意 | `512` | - |
| `height` | int | 任意 | `320` | - |
| `num_frames` | int | 任意 | `25` | 8n+1丸め |
| `fps` | float | 任意 | `25` | - |
| `steps` | int | 任意 | `8` | - |
| `guidance_scale` | float | 任意 | `1.0` | - |
| `strength` | float | 任意 | `0.7` | 条件付け強度(両フレーム共通) |
| `seed` | int | 任意 | `-1` | - |
| `audio_out` | string | 任意 | `"on"` | `"on"` \| `"off"` |
| `upscale` | int | 任意 | `0` | `0`(無効) \| `1`(latent upsamplerで2x空間アップスケール、2026-07-23追加) |

**レスポンス例**: `/api/ltx2/t2v` と同様のキー構成 + `strength`。`mode` は `"ltx2_flf"`。
`upscale=1` 指定時、レスポンスの `width`/`height` は要求値の2倍(実際の出力解像度)になる
(i2v/t2vと同じ挙動)。高解像度が必要な場合は、大きな `width`/`height` を直接指定するの
ではなく `upscale=1` を使うこと(直接denoiseはattentionメモリがフレーム数のほぼ2乗で
増大し、1280×704×121フレームで87GB超のCUDA OOMを実測。CLAUDE.md 51番)。

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/ltx2/flf \
  -F "first_image=@a.png" -F "last_image=@b.png" \
  -F "prompt=smooth transition" -F "strength=0.7"
```

**特記事項**: 画像は「長辺基準リサイズ+中央クロップ」で自動処理される(事前リサイズ不要)。
補間の滑らかさは入力2枚の意味的な近さ・フレーム数・プロンプトに強く依存する
(意味的に離れた2枚では中間フレームで急激に遷移する傾向を実機確認済み)。

---

### POST /api/ltx2/keyframes

任意キーフレーム条件付け(FLFの一般化)。N枚の画像を任意フレーム位置・任意strengthで
条件付ける。Content-Type: `multipart/form-data`。

| 名前 | 型 | 必須/任意 | 既定値 | 説明・制約 |
|---|---|---|---|---|
| `images` | file[] | 必須 | - | N枚 |
| `indices` | string | 必須 | - | カンマ区切り整数(負値可、例`"0,45,-1"`)。`images`と同数必須 |
| `strengths` | string | 任意 | `""` | カンマ区切りfloat。省略時は`0.7`を全画像に適用、1個指定時も全画像へブロードキャスト |
| `prompt` | string | 必須 | - | - |
| `negative_prompt` | string | 任意 | `""` | - |
| `width` | int | 任意 | `512` | - |
| `height` | int | 任意 | `320` | - |
| `num_frames` | int | 任意 | `25` | 8n+1丸め |
| `fps` | float | 任意 | `25` | - |
| `steps` | int | 任意 | `8` | - |
| `guidance_scale` | float | 任意 | `1.0` | - |
| `seed` | int | 任意 | `-1` | - |
| `audio_out` | string | 任意 | `"on"` | - |

**レスポンス例**

```json
{
  "mode": "ltx2_keyframes",
  "indices": [0, 12, -1],
  "strengths": [0.7, 0.7, 0.7],
  "num_keyframes": 3,
  "video_url": "/outputs/ltx2_keyframes_20260721_130000_deadbeef.mp4",
  "video_only_url": "/outputs/ltx2_keyframes_20260721_130000_deadbeef.mp4",
  "audio_url": "/outputs/ltx2_keyframes_20260721_130000_deadbeef.wav",
  "has_audio": true
}
```
（他キーは`/api/ltx2/t2v`と同様）

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/ltx2/keyframes \
  -F "images=@a.png" -F "images=@b.png" -F "images=@c.png" \
  -F "indices=0,12,-1" -F "prompt=morphing sequence"
```

**特記事項**: `indices`/`strengths`の解析エラー(不正な数値文字列)は400。

---

### POST /api/ltx2/ia2v

IA2V(Image + Audio to Video、参照画像+wav音声からリップシンク動画を生成)。
Content-Type: `multipart/form-data`。

| 名前 | 型 | 必須/任意 | 既定値 | 説明・制約 |
|---|---|---|---|---|
| `image` | file | 必須 | - | - |
| `audio` | file | 必須 | - | wav等の音声ファイル |
| `prompt` | string | 任意 | `""`(未指定時 `DEFAULT_IA2V_PROMPT`) | - |
| `negative_prompt` | string | 任意 | `""` | - |
| `width` | int | 任意 | `512` | - |
| `height` | int | 任意 | `320` | - |
| `num_frames` | int | 任意 | `0`(未指定→音声長×fpsから自動計算、8n+1丸め) | - |
| `fps` | float | 任意 | `25` | - |
| `steps` | int | 任意 | `8` | - |
| `guidance_scale` | float | 任意 | `1.0` | - |
| `seed` | int | 任意 | `-1` | - |
| `audio_out` | string | 任意 | `"original"` | `"original"`(入力wav原音mux)\| `"vocoder"`(再合成音)\| `"none"`(無音mp4) |

**レスポンス例**

```json
{
  "mode": "ltx2_ia2v",
  "prompt": "a person speaking",
  "audio_guidance_scale": 1.0,
  "gen_elapsed_s": 33.3,
  "elapsed_s": 35.1,
  "peak_vram_gb": 35.48,
  "video_url": "/outputs/ltx2_ia2v_20260721_130500_11112222.mp4",
  "video_only_url": "/outputs/ltx2_ia2v_20260721_130500_11112222.mp4",
  "audio_url": "/outputs/ltx2_ia2v_20260721_130500_11112222_orig.wav",
  "has_audio": true,
  "audio_out": "original",
  "audio_duration_s": 3.46
}
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/ltx2/ia2v -F "image=@face.png" -F "audio=@speech.wav"
```

**特記事項**: `audio_out`の値域が他モード(`"on"`/`"off"`)と異なる点に注意。
`gen_elapsed_s`はdenoiseループのみの所要時間（IA2V/V2A固有のキー）。
groupモード時は時間方向tiled decodeが自動適用される。

---

### POST /api/ltx2/v2a

V2A(Video to Audio、既存動画に音声を生成して付与)。Content-Type: `multipart/form-data`。

| 名前 | 型 | 必須/任意 | 既定値 | 説明・制約 |
|---|---|---|---|---|
| `video` | file | 必須 | - | 入力動画 |
| `prompt` | string | 任意 | `""`(未指定時 `DEFAULT_V2A_PROMPT`) | - |
| `negative_prompt` | string | 任意 | `""`(未指定時 `DEFAULT_V2A_NEGATIVE_PROMPT`) | - |
| `width` | int | 任意 | `512` | - |
| `height` | int | 任意 | `320` | - |
| `num_frames` | int | 任意 | `0`(未指定→動画全長を8n+1へ丸め、上限`DEFAULT_V2A_MAX_NUM_FRAMES`=361) | - |
| `steps` | int | 任意 | `8` | - |
| `guidance_scale` | float | 任意 | `1.0` | - |
| `seed` | int | 任意 | `-1` | - |

**レスポンス例**

```json
{
  "mode": "ltx2_v2a",
  "fps": 24.0,
  "num_frames": 89,
  "source_total_frames": 89,
  "video_duration_s": 3.7,
  "video_url": "/outputs/ltx2_v2a_20260721_131000_33334444.mp4",
  "video_only_url": "/outputs/ltx2_v2a_20260721_131000_33334444.mp4",
  "audio_url": "/outputs/ltx2_v2a_20260721_131000_33334444.wav",
  "has_audio": true
}
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/ltx2/v2a -F "video=@silent.mp4" -F "prompt=footsteps on snow"
```

**特記事項**: `fps`は入力動画から自動検出した値。`audio_out`パラメータは存在しない
(常にボコーダ生成音を付与)。入力動画の元音声(あれば)は無視され差し替えられる。
groupモードとnoneモードで生成音声はbit一致しない(音質水準は同等、仕様)。

---

### POST /api/ltx2/iclora

IC-LoRA(MergeGreen、参照動画の中央部を緑マスクし指示に沿って動画を編集)。
Content-Type: `multipart/form-data`。

| 名前 | 型 | 必須/任意 | 既定値 | 説明・制約 |
|---|---|---|---|---|
| `video` | file | 必須 | - | 参照動画 |
| `prompt` | string | 必須 | - | 「シーン内で何が変わるか」を指示(必須、未指定はエラー) |
| `negative_prompt` | string | 任意 | `""`(未指定時 `DEFAULT_ICLORA_NEGATIVE_PROMPT`) | - |
| `mask_mode` | string | 任意 | `"middle"` | `"middle"`(サーバ側で中央1/3を緑マスク)\| `"prefilled"`(既に緑マスク済み動画) |
| `mask_start` | int \| null | 任意 | `null`(`mask_mode=middle`時、未指定は動画全体の中央1/3の開始フレーム) | フレームインデックス |
| `mask_end` | int \| null | 任意 | `null` | 同上、終了フレーム |
| `width` | int | 任意 | `512` | - |
| `height` | int | 任意 | `320` | - |
| `num_frames` | int | 任意 | `0`(未指定→動画全長を8n+1へ丸め、上限361) | - |
| `fps` | float | 任意 | `25` | - |
| `steps` | int | 任意 | `8` | - |
| `guidance_scale` | float | 任意 | `1.0` | - |
| `ref_strength` | float | 任意 | `1.0` | 参照条件付け強度 |
| `seed` | int | 任意 | `-1` | - |

**レスポンス例**

```json
{
  "mode": "ltx2_iclora",
  "prompt": "the snowy winter forest transforms into a lush green summer forest",
  "mask_mode": "middle",
  "mask_start": 8,
  "mask_end": 17,
  "ref_strength": 1.0,
  "video_url": "/outputs/ltx2_iclora_20260721_131500_55556666.mp4",
  "video_only_url": "/outputs/ltx2_iclora_20260721_131500_55556666.mp4",
  "has_audio": true,
  "source_total_frames": 25
}
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/ltx2/iclora \
  -F "video=@scene.mp4" -F "prompt=the snow melts into a green forest"
```

**特記事項**: LoRAは `siraxe/MergeGreen_IC-lora_ltx2.3`(Apache-2.0、weight 0.9固定）。
このLoRAはdev(フル)モデル向けに学習されたものを蒸留fp8モデルへ転用しているため、
マスク境界外への色滲み等の品質の粗さが確認されている(モデルミスマッチが主因と推定)。
`mask_mode`が不正な値だと400、`prompt`未指定も400。

---

## JoyAI-Image-Edit-Plus

`jdopensource/JoyAI-Image-Edit-Plus-Diffusers`(Apache 2.0)。複数参照画像編集・合成。
qwen_image/flux2/z_image/ltx2とVRAM同時常駐不可のため5方向相互排他。

### POST /api/joyai/edit

Content-Type: `multipart/form-data`。

| 名前 | 型 | 必須/任意 | 既定値 | 説明・制約 |
|---|---|---|---|---|
| `images` | file[] | 必須 | - | 参照画像1〜3枚(`MAX_REF_IMAGES`超過は400) |
| `prompt` | string | 必須 | - | - |
| `negative_prompt` | string | 任意 | `""` | - |
| `steps` | int | 任意 | `30` | - |
| `guidance_scale` | float | 任意 | `4.0` | - |
| `seed` | int | 任意 | `-1` | - |
| `width` | int \| null | 任意 | `null`(最後の参照画像サイズから自動推定) | - |
| `height` | int \| null | 任意 | `null` | - |

**レスポンス例**

```json
{
  "mode": "joyai_edit",
  "prompt": "make them hold an apple and smile",
  "negative_prompt": "",
  "num_ref_images": 2,
  "width": 1024,
  "height": 1024,
  "steps": 30,
  "cfg": 4.0,
  "guidance_scale": 4.0,
  "seed": 12345,
  "elapsed_s": 157.1,
  "peak_vram_gb": 34.1,
  "image_url": "/outputs/joyai_edit_20260721_132000_77778888.png",
  "model": "joyai-edit-plus",
  "te_offload": true
}
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/joyai/edit \
  -F "images=@person.png" -F "images=@apple.png" -F "prompt=make them hold an apple and smile"
```

**特記事項**: `width`/`height`の`image_url`結果の実寸(生成結果の実サイズ)がレスポンスの
`width`/`height`に入る(リクエスト値そのままではない)。48GB専有環境ではtransformer
(~32.5GB)とtext_encoder(~17.5GB)を相互排他スワップする設計のため、96GB機でも48GB専有と
同じ速度・VRAM使用量になる(`DS_JOYAI_TE_OFFLOAD=auto`が常時有効なため）。

---

## Mage-Flow(別プロセスプロキシ)

Microsoft の軽量画像生成/編集モデル(4.1B bf16、MIT)。torch 2.13 / transformers 5.5 /
flash-attn 2.8.3 を要求し本体venvとバージョン衝突するため、**専用venv
(`venv-mageflow/`)の別プロセス**(`mageflow_service/app_mageflow.py`、
`./run_mageflow.sh` で起動、既定ポート8602)で動作する。本体の `/api/mageflow/*` は
`DS_MAGEFLOW_URL`(既定 `http://127.0.0.1:8602`)へのHTTPプロキシ。
**ラッパー未起動時は生成系が502**(起動コマンド入りの日本語メッセージ)を返す。
詳細は CLAUDE.md 50番。

バリアント(`model` パラメータ、いずれも `steps`/`cfg` 未指定時の既定値が変わる):

| `model` | T2I リポジトリ | Edit リポジトリ | 既定steps(T2I/Edit) | 既定cfg |
|---|---|---|---|---|
| `base` | microsoft/Mage-Flow-Base | microsoft/Mage-Flow-Edit-Base | 30 / 30 | 5.0 |
| `rl`(既定) | microsoft/Mage-Flow | microsoft/Mage-Flow-Edit | 20 / 30 | 5.0 |
| `turbo` | microsoft/Mage-Flow-Turbo | microsoft/Mage-Flow-Edit-Turbo | 4 / 4 | 1.0 |

ラッパーは**同時1バリアントのみ常駐**(別バリアント要求時は自動unload+再ロード)。
全プロンプトは内蔵コンテンツゲートで検査され(無効化不可)、拒否時はプレースホルダ
画像が返る。正常出力には Gaussian-Shading 透かしが常に埋め込まれる。

**排他制御(`exclusive` パラメータ、既定 `true`)**: `true` なら本体の
`core.gpu.generation_lock` を非ブロッキング取得してから転送する(取得失敗は409。
他ファミリーの生成と相互に排他)。`false` ならロックを取らず即転送し、他ファミリーの
生成と並行実行できる(4.1B・ピーク18-20GB。空きVRAMが不足する組み合わせでは
どちらかがOOMするリスクはユーザー持ち)。`registry.load()` は呼ばない
(モデルはプロセス外のため FamilyRegistry の排他unload対象ではない)。

### POST /api/mageflow/t2i

Content-Type: `application/json`。

| 名前 | 型 | 必須/任意 | 既定値 | 説明・制約 |
|---|---|---|---|---|
| `prompt` | string | 必須 | - | - |
| `negative_prompt` | string \| null | 任意 | `null` | cfg>1 のとき有効 |
| `width` | int | 任意 | `1024` | 16の倍数へ丸め、512〜2048にクランプ |
| `height` | int | 任意 | `1024` | 同上 |
| `steps` | int \| null | 任意 | `null`(バリアント既定) | - |
| `cfg` | float \| null | 任意 | `null`(バリアント既定) | - |
| `seed` | int | 任意 | `-1`(ランダム) | - |
| `model` | string | 任意 | `"rl"` | `base` / `rl` / `turbo` |
| `exclusive` | bool | 任意 | `true` | 上記「排他制御」参照 |

**レスポンス例**

```json
{
  "mode": "mageflow_t2i",
  "prompt": "a cat holding a sign that says hello",
  "width": 1024, "height": 1024,
  "steps": 20, "cfg": 5.0, "seed": 12345,
  "model": "rl", "repo": "microsoft/Mage-Flow",
  "elapsed_s": 5.2, "load_time_s": 20.0, "peak_vram_gb": 18.5,
  "image_url": "/outputs/mageflow_t2i_20260722_120000_aabbccdd.png",
  "exclusive": true
}
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/mageflow/t2i \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat holding a sign that says hello", "model": "rl"}'
```

### POST /api/mageflow/edit

Content-Type: `multipart/form-data`。

| 名前 | 型 | 必須/任意 | 既定値 | 説明・制約 |
|---|---|---|---|---|
| `image` | file[] | 必須 | - | 参照画像1〜3枚(学習時上限。超過は400) |
| `prompt` | string | 必須 | - | 編集指示 |
| `negative_prompt` | string | 任意 | `""` | - |
| `steps` | int \| null | 任意 | `null`(バリアント既定) | - |
| `cfg` | float \| null | 任意 | `null`(バリアント既定) | - |
| `seed` | int | 任意 | `-1`(ランダム) | - |
| `max_size` | int | 任意 | `1024` | 出力長辺(width/height未指定時のみ有効) |
| `width` | int \| null | 任意 | `null` | height と両方指定時のみ明示解像度 |
| `height` | int \| null | 任意 | `null` | 同上 |
| `model` | string | 任意 | `"rl"` | `base` / `rl` / `turbo` |
| `exclusive` | bool | 任意 | `true` | 上記「排他制御」参照 |

**レスポンス例**(`width`/`height` は出力画像の実寸)

```json
{
  "mode": "mageflow_edit",
  "prompt": "把背景改为城市街道",
  "num_ref_images": 1,
  "width": 1024, "height": 768, "max_size": 1024,
  "steps": 30, "cfg": 5.0, "seed": 42,
  "model": "rl", "repo": "microsoft/Mage-Flow-Edit",
  "elapsed_s": 9.8, "load_time_s": 21.5, "peak_vram_gb": 19.9,
  "image_url": "/outputs/mageflow_edit_20260722_120100_eeff0011.png",
  "exclusive": true
}
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/mageflow/edit \
  -F "image=@momo.png" -F "prompt=change the background to a city street"
```

### GET /api/mageflow/status

ラッパーサービスの状態を返す。**未起動時は502ではなく200で
`{"service": "mageflow", "available": false, "detail": "..."}`** を返す
(UIステータスバーが8秒ごとにポーリングするため)。起動時はラッパーの
`/status` をそのまま透過する:

```json
{
  "service": "mageflow", "loaded": true,
  "kind": "t2i", "variant": "rl", "repo": "microsoft/Mage-Flow",
  "load_time_s": 20.0, "busy": false,
  "vram": {"allocated_gb": 9.5, "max_allocated_gb": 18.5, "free_gb": 40.0, "total_gb": 95.0}
}
```

### POST /api/mageflow/unload

ラッパーにロード済みのパイプラインを解放させる(`{"freed": [...]}`)。
未起動時は502。本体の `/api/unload` とは独立(本体側のモデルには影響しない)。

---

## charsheet

1枚の画像から8方向キャラクターシートを生成するアプリケーション。`/api/charsheet/` 配下。
方向キー(8方向): `front`, `back`, `left`, `right`, `front_left_45`, `front_right_45`,
`back_left_45`, `back_right_45`。

charsheetジョブはバックグラウンドスレッドで非同期実行され、`current_job_id`により
同時1ジョブに制限される(別ジョブ実行中の操作は409）。GPU排他は他ファミリーと共通の
`core.gpu.generation_lock`を使う。

### POST /api/charsheet/split

複数キャラクターを検出・分離する(GPU不使用)。Content-Type: `multipart/form-data`。

| 名前 | 型 | 必須/任意 | 説明 |
|---|---|---|---|
| `image` | file | 必須 | 複数キャラクターを含む可能性のある画像 |

**レスポンス例**

```json
{
  "split_id": "a1b2c3d4e5f6",
  "count": 2,
  "figures": [
    {"index": 0, "url": "/api/charsheet/splits/a1b2c3d4e5f6/figure_0.png", "width": 512, "height": 768},
    {"index": 1, "url": "/api/charsheet/splits/a1b2c3d4e5f6/figure_1.png", "width": 480, "height": 720}
  ]
}
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/charsheet/split -F "image=@group_photo.png"
```

---

### GET /api/charsheet/splits/{split_id}/{filename}

分割済み画像の取得。`filename`は`"source.png"`または`"figure_{i}.png"`のみ許可
(パストラバーサル対策、不正なら404)。レスポンス: `image/png`のFileResponse。

---

### POST /api/charsheet/generate

8方向生成ジョブを開始する。Content-Type: `multipart/form-data`。

| 名前 | 型 | 必須/任意 | 既定値 | 説明・制約 |
|---|---|---|---|---|
| `image` | file | 任意 | `null` | 単体アップロード。`split_id`+`figure_index`との排他的併用(どちらか必須、両方無しは400) |
| `seed` | int | 任意 | `0` | - |
| `split_id` | string | 任意 | `null` | `/split`で得たID |
| `figure_index` | int | 任意 | `null` | `/split`で得たインデックス |

**レスポンス例**

```json
{ "job_id": "1a2b3c4d5e6f" }
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/charsheet/generate -F "image=@char.png" -F "seed=12345"
```

**特記事項**: 別ジョブ実行中は409。8方向は `DS_CHARSHEET_METHOD`(既定`bf16-group`)で
選択したEdit変種で連続生成される(1回ロードして使い回し)。既定解像度1024×1024
(`CHARSHEET_EDIT_SIZE`で変更可)。

---

### GET /api/charsheet/jobs/{job_id}

ジョブ状態を取得する。ディスクからの復元(`_restore_job_from_disk`)にも対応。

**レスポンス例**

```json
{
  "job_id": "1a2b3c4d5e6f",
  "status": "done",
  "progress": 8,
  "total": 8,
  "seed": 12345,
  "views": [
    {
      "key": "front", "label_ja": "前", "label_en": "Front",
      "status": "done",
      "url": "/api/charsheet/jobs/1a2b3c4d5e6f/images/front.png",
      "has_prev": false
    }
  ],
  "sheet_url": "/api/charsheet/jobs/1a2b3c4d5e6f/sheet.png",
  "zip_url": "/api/charsheet/jobs/1a2b3c4d5e6f/download.zip",
  "error": null,
  "refine_error": null,
  "created_at": "2026-07-21T13:20:00.000000",
  "load_info": {
    "loaded": true,
    "quant": null,
    "fallback": false,
    "lightning_merged": true,
    "load_time_s": 33.6,
    "angles_lora": true,
    "method": "bf16-group"
  }
}
```

**ジョブ全体の `status` が取りうる値**: `queued` / `running` / `error` / `done` /
`refining` / `removing_bg`

**`views[].status` が取りうる値**: `queued` / `running` / `error` / `done` /
`refining` / `removing_bg`

見つからない場合404。

---

### POST /api/charsheet/jobs/{job_id}/refine

個別ビューの修正指示(Edit 1回適用)。Content-Type: `application/json`。

| 名前 | 型 | 必須/任意 | 説明 |
|---|---|---|---|
| `key` | string | 必須 | 方向キー(不正値は400) |
| `instruction` | string | 必須 | 修正指示(空は400) |
| `seed` | int | 任意(既定0) | - |

**レスポンス例**

```json
{ "job_id": "1a2b3c4d5e6f", "key": "front", "status": "refining" }
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/charsheet/jobs/1a2b3c4d5e6f/refine \
  -H "Content-Type: application/json" -d '{"key":"front","instruction":"change shoes to red","seed":1}'
```

**特記事項**: 対象画像未生成は409、他ジョブ実行中は409。実行前に現画像を`{key}_prev.png`へ
バックアップし(undo用、1世代のみ)、修正指示末尾に "Keep everything else exactly the
same." が自動付加される。

---

### POST /api/charsheet/jobs/{job_id}/remove_bg

背景除去(rembg / isnet-general-use)。Content-Type: `application/json`。GPU不使用だが
ジョブの同時1件制約は共有する。

| 名前 | 型 | 必須/任意 | 説明 |
|---|---|---|---|
| `key` | string | 必須 | 方向キー、または `"all"`(全方向) |

**レスポンス例**

```json
{ "job_id": "1a2b3c4d5e6f", "keys": ["front", "back", "left", "right"], "status": "removing_bg" }
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/charsheet/jobs/1a2b3c4d5e6f/remove_bg \
  -H "Content-Type: application/json" -d '{"key":"all"}'
```

**特記事項**: 対象画像が1つも生成されていなければ409。実行前に各画像を`{key}_prev.png`へ
バックアップする。

---

### POST /api/charsheet/jobs/{job_id}/undo

修正の取り消し(トグル、`{key}_prev.png`と入れ替え)。Content-Type: `application/json`。同期処理。

| 名前 | 型 | 必須/任意 | 説明 |
|---|---|---|---|
| `key` | string | 必須 | 方向キー(不正値は400) |

**レスポンス**: ジョブ全体のdictをそのまま返す(`GET /api/charsheet/jobs/{job_id}`と同形式)。

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/charsheet/jobs/1a2b3c4d5e6f/undo \
  -H "Content-Type: application/json" -d '{"key":"front"}'
```

**特記事項**: 復元できるバックアップが無ければ404、現在の画像が存在しなければ409。
トグル動作のため、undoを2回実行すると元の状態に戻る。

---

### GET /api/charsheet/jobs/{job_id}/images/{key}.png

各方向の生成画像(`image/png`)。無ければ404。

### GET /api/charsheet/jobs/{job_id}/input.png

前処理済みアップロード画像(`image/png`)。無ければ404。

### GET /api/charsheet/jobs/{job_id}/sheet.png

4×2グリッド合成画像(`image/png`)。無ければ404。

### GET /api/charsheet/jobs/{job_id}/download.zip

8方向 + sheet.pngの一括ZIP。`Content-Type: application/zip`、
`filename=character_sheet_{job_id}.zip`。無ければ404。

---

## ユーティリティ

### POST /api/remove_bg

背景削除(rembg / isnet-general-use、GPU不使用・排他不要)。Content-Type: `multipart/form-data`。
charsheetの`remove_bg`と同一実装(`apps/charsheet/bg.py`)を再利用。

| 名前 | 型 | 必須/任意 | 説明 |
|---|---|---|---|
| `image` | file | 必須 | 入力画像(内部で`RGBA`変換) |

**レスポンス例**

```json
{ "mode": "remove_bg", "image_url": "/outputs/removebg_20260721_133000_99998888.png", "elapsed_s": 0.8 }
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/remove_bg -F "image=@photo.png"
```

初回呼び出し時、モデル(~179MB)を`~/.u2net`へ自動ダウンロードする。

---

### POST /api/prompt/enhance

プロンプトの英語強化(GPU不使用・排他不要)。Content-Type: `application/json`。

| 名前 | 型 | 必須/任意 | 説明 |
|---|---|---|---|
| `text` | string | 必須 | 強化対象のプロンプト |

**レスポンス例**

```json
{ "result": "A photorealistic cat sitting elegantly on a wooden chair, soft lighting, ...", "elapsed_s": 1.2 }
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/prompt/enhance -H "Content-Type: application/json" -d '{"text":"a cat"}'
```

**特記事項**: `DS_LLM_URL`(既定`http://127.0.0.1:64652`)のOpenAI互換LLMサーバに接続する。
未接続時は502。画像生成用ロックは一切使わない。

---

### POST /api/prompt/translate

日本語プロンプトの英訳(GPU不使用・排他不要)。Content-Type: `application/json`。

| 名前 | 型 | 必須/任意 | 説明 |
|---|---|---|---|
| `text` | string | 必須 | 翻訳対象の日本語テキスト |

**レスポンス例**

```json
{ "result": "a cat sitting on a wooden chair", "elapsed_s": 0.9 }
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/prompt/translate -H "Content-Type: application/json" -d '{"text":"椅子に座る猫"}'
```

**特記事項**: `/api/prompt/enhance`と同様、LLM未接続時は502。

---

## 管理系

### GET /api/status

全ファミリーのロード状態・量子化・VRAMを返す(GET、パラメータなし)。

**レスポンス構造(抜粋、実測)**

```json
{
  "offload_mode": "none",
  "runtime_config": "RuntimeConfig(offload='auto', ..., t2i_model='2512', ...)",
  "shared_loaded": true,
  "shared_load_time_s": 12.92,
  "t2i_model": "2512",
  "t2i_loaded": true,
  "t2i_load_time_s": 38.15,
  "t2i_quant": "fp8-lightning",
  "t2i_lora_available": true,
  "t2i_lightning_merged": true,
  "edit_loaded": false,
  "edit_angles_loaded": false,
  "edit_angles_bf16_loaded": false,
  "edit_angles_bf16group_loaded": false,
  "controlnet_union_loaded": false,
  "controlnet_inpaint_loaded": false,
  "layered_loaded": false,
  "gpu_busy": false,
  "vram": {
    "allocated_gb": 19.31,
    "max_allocated_gb": 34.84,
    "reserved_gb": 37.62,
    "free_gb": 53.06,
    "total_gb": 94.97
  },
  "flux2": { "active_model": "dev", "loaded": false, "vram": { "...": "..." } },
  "z_image": { "loaded": false, "i2i_loaded": false, "inpaint_loaded": false },
  "ltx2": { "loaded": false, "i2v_loaded": false, "flf_loaded": false, "iclora_loaded": false, "upsampler_loaded": false },
  "joyai": { "loaded": false, "te_offload": null, "patched": false },
  "last_generation": { "mode": "t2i", "elapsed_s": 47.25, "peak_vram_gb": 34.84, "at": "2026-07-21T21:34:35" }
}
```

**注意**: 旧クライアント互換のため、`qwen_image`ファミリーの状態キー(`t2i_loaded`等)は
トップレベルに展開される。`flux2` / `z_image` / `ltx2` / `joyai` は各々ネストされたオブジェクト
として返る。`vram`はファミリーごとにも重複して含まれる(全て同じプロセス全体のVRAM値)。

**curlサンプル**

```bash
curl http://localhost:8601/api/status
```

---

### POST /api/unload

指定ファミリー(またはグループ)のモデルを明示的に解放する。Content-Type: `application/json`。

| 名前 | 型 | 必須/任意 | 既定値 | 説明 |
|---|---|---|---|---|
| `target` | string | 任意 | `"all"` | `"t2i"` \| `"edit"` \| `"edit_angles"` \| `"controlnet"` \| `"layered"` \| `"flux2"` \| `"zimage"` \| `"ltx2"` \| `"joyai"` \| `"all"` |

**レスポンス例**

```json
{ "qwen_image": {"...": "..."}, "flux2": {"...": "..."}, "z_image": {"...": "..."}, "ltx2": {"...": "..."}, "joyai": {"...": "..."} }
```

`target="all"`は登録済み全ファミリーを一括解放する。`target`が未対応の値だと400
(「target は t2i / edit / controlnet / layered / flux2 / zimage / ltx2 / joyai / all のいずれかです」)。

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/unload -H "Content-Type: application/json" -d '{"target":"all"}'
```

---

### GET /api/progress

生成中の進捗状態を返す(グローバル1本、非GPU・排他不要)。生成リクエストと同時に
500ms間隔程度でポーリングしてよい。

**レスポンス構造**

| フィールド | 型 | 説明 |
|---|---|---|
| `active` | bool | 生成/ロード中か |
| `mode` | string \| null | 実行中モード(例: `t2i`, `edit`, `flux2_t2i`, `ltx2_t2v`, `charsheet`, `joyai_edit`) |
| `phase` | string \| null | `"loading"`(モデルロード中、step/total_steps不定) \| `"generating"`(denoiseループ) \| `"decoding"`(VAE decode等の後処理) |
| `step` | int | 現在ステップ(1-origin) |
| `total_steps` | int | 総ステップ数 |
| `started_at` | float \| null | 内部タイムスタンプ |
| `elapsed_s` | float \| null | 経過秒数(動的計算) |
| `extra` | object \| null | 追加情報。charsheetジョブ中は`{job_id, direction, direction_label, direction_index, direction_total}`等 |

**レスポンス例(非実行中)**

```json
{ "active": false, "mode": "t2i", "phase": null, "step": 0, "total_steps": 0, "started_at": null, "extra": null, "elapsed_s": null }
```

**レスポンス例(charsheet実行中)**

```json
{
  "active": true,
  "mode": "charsheet",
  "phase": "generating",
  "step": 2,
  "total_steps": 4,
  "elapsed_s": 3.1,
  "extra": { "job_id": "1a2b3c4d5e6f", "direction": "back", "direction_label": "後", "direction_index": 2, "direction_total": 8 }
}
```

**curlサンプル**

```bash
curl http://localhost:8601/api/progress
```

---

### GET /

評価UI(`static/index.html`)を返す。

---

## エラー仕様

| ステータスコード | 意味 | 代表的な `detail` メッセージ |
|---|---|---|
| `400` | バリデーションエラー・モデル切替エラー | `"参照画像を1〜3枚アップロードしてください。"` / `"参照画像は最大{n}枚までです。"` / `"resolution は [640, 1024] のいずれかである必要があります"` / `"画像の読み込みに失敗しました: {exc}"` / `"ecocoroは廃止されました"`(ValueErrorを400へ変換) / `"target は t2i / edit / ... のいずれかです"` |
| `404` | リソースが見つからない | `"ジョブが見つかりません"` / `"画像が見つかりません"` / `"シートが見つかりません"` / `"ZIP が見つかりません"` / `"不正な split_id です"` / `"復元できるバックアップがありません"` |
| `409` | 実行中の別処理と競合 | `"別の生成が実行中です。しばらく待ってから再試行してください。"`(GPUロック競合) / `"別のジョブが実行中です。しばらく待ってから再試行してください。"`(charsheetジョブ競合) / `"この方向の画像はまだ生成されていません"` / `"対象の画像がまだ生成されていません"` / `"現在の画像が存在しません"` |
| `500` | 生成処理中の予期しない失敗 | `"{mode}生成に失敗しました: {exc}"`(トレースバックはサーバログに出力、レスポンスには含まれない) / `"背景削除に失敗しました: {exc}"` / `"キャラクター検出に失敗しました: {exc}"` |
| `501` | 依存ライブラリ不足 | `"opencv(cv2)が利用できません: {exc}"`(`/api/canny`でcv2未インストール時) |
| `502` | LLMサーバ未接続 | LLM接続エラーメッセージ(`DS_LLM_URL`を確認するよう促す日本語文言、`core.llm.LLMConnectionError`由来) |
| `422` | FastAPI標準のリクエストボディ検証エラー | Pydanticモデルの型不一致等(`HTTPValidationError`スキーマ、`detail`は`ValidationError`の配列) |

**エラー処理の一般則(`app.py` `_generate_or_409()`)**:
- `ValueError`を送出するファミリー実装は自動的に**400**に変換される。
- `HTTPException`はそのまま再送出される。
- その他の例外は**500**に変換され、トレースバックがサーバの標準出力に出力される
  (`traceback.print_exc()`、レスポンスの`detail`には例外メッセージのみ含まれる)。
- いずれの場合も`finally`ブロックで進捗状態のリセット(`progress_mod.finish()`)と
  GPUロックの解放(`gpu.generation_lock.release()`)が保証される。

---

## 環境変数の影響

以下はAPI応答・挙動に直接影響する主要な環境変数。全一覧・既定値の根拠は
`README.md`「環境変数一覧」を参照。

| 環境変数 | 影響するAPI | 概要 |
|---|---|---|
| `DS_T2I_MODEL` | `/api/t2i`, `/api/i2i` | 起動時のT2I既定モデル(`model`パラメータ未指定時に使われる)。`"2512"`(既定)\| `"qwen-image"` |
| `DS_QUANT` | `/api/t2i`, `/api/edit`, `/api/controlnet` 等 | Qwen系の量子化方式。レスポンスの`quant`フィールドに反映 |
| `DS_EDIT_TE_OFFLOAD` | `/api/edit`, `/api/i2i`, `/api/t2i`, charsheet | text_encoder CPU退避の`auto`/`on`/`off`。レスポンスの`te_offload`に反映 |
| `DS_CHARSHEET_METHOD` | `/api/charsheet/generate` とその派生 | 8方向生成の方式(`bf16-group`既定 / `bf16-adapters` / `fp8-fuse` / `prompt-only`)。ジョブの`load_info.method`に反映 |
| `DS_OFFLOAD` | 全ファミリーの生成速度・VRAM | オフロードモード(`none`/`model`/`group`/`group_lowvram`)。レスポンスの`offload_mode`に反映 |
| `DS_LTX2_OFFLOAD` | `/api/ltx2/*` | `none`(96GB機向け)/ `group`(既定、2026-07-22変更)/ `auto`(非推奨)。生成速度・ピークVRAMに直接影響。CLAUDE.md 49番参照 |
| `DS_LTX2_TE_QUANT` | `/api/ltx2/*` | text_encoder(Gemma 3 12B)の量子化(`none`/`fp8`(既定、2026-07-22変更)/`nf4`)。VRAM削減に影響。`nf4`は別チェックポイントで品質A/B未確定 |
| `DS_LTX2_TILED_DECODE` | `/api/ltx2/*` | `1`(既定、2026-07-23追加): 全モードのVAEデコードを常時tiled化。noneモードでの長尺(361f)・高解像度の一括デコードOOMを解消(CLAUDE.md 52番)。`0`で旧動作 |
| `DS_JOYAI_TE_OFFLOAD` | `/api/joyai/edit` | transformer⇔text_encoder相互排他スワップの有効化(既定`auto`で実質常時有効) |
| `DS_ZIMAGE_PRECISION` | `/api/zimage/*` | `bf16`(既定)\| `bnb-4bit` |
| `DS_FLUX2_PRECISION` | `/api/flux2/*` | `bnb-4bit`(既定)\| `bf16` |
| `DS_LLM_URL` | `/api/prompt/enhance`, `/api/prompt/translate` | LLMサーバの接続先。未接続だと502 |
| `DS_COMFYUI_DIR` | ほぼ全ての生成系 | モデル重みの優先探索先(ComfyUIディレクトリ) |

詳細な既定値・後方互換の旧環境変数名・調整根拠は `README.md` の「環境変数一覧」表を参照。
