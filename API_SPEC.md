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

### POST /api/outpaint

アウトペイント(画角拡張、2026-07-24追加)。入力画像をアスペクト維持でターゲット
キャンバス中央に配置し、余白帯を既存インペイント機構で「同じシーンの続き」として
生成する。最後に元画像の中央部(フェザー帯より内側)をピクセルそのまま再合成する
(中央無傷の保証)。Content-Type: `multipart/form-data`。

**実行前の安全策(2026-07-24追加、CLAUDE.md 56番)**: 生成本体に入る前に、必ず
`target=all`相当で登録済み全ファミリーを解放してから対象パイプラインをロードし直す
(48GB専有機での実機OOM対策)。他ファミリーが元々未ロードなら実害はほぼ無いが、
何かロード済みの場合は解放+再ロード分(実測: 空きVRAM十分な状態からでも
毎回+約50秒)のオーバーヘッドが常に発生する。

| 名前 | 型 | 必須/任意 | 既定値 | 説明・制約 |
|---|---|---|---|---|
| `image` | file | 必須 | - | 入力画像(スクエア/縦長→横長、横長→縦長の両方向対応) |
| `width` | int | 任意 | `1280` | ターゲット幅 |
| `height` | int | 任意 | `720` | ターゲット高さ |
| `prompt` | string | 任意 | `""` | 誘導プロンプト(空可。空なら周囲文脈のみで生成) |
| `negative_prompt` | string | 任意 | `" "` | - |
| `seed` | int | 任意 | `-1` | - |
| `feather` | int | 任意 | `64` | 元画像側への食い込み幅(px)。継ぎ目の滑らかさに影響 |
| `engine` | string | 任意 | `"qwen"` | `"qwen"`(ControlNet Inpainting、A/B検証で明確に優位)\| `"zimage"`(文脈無視の傾向あり非推奨) |
| `steps` | int | 任意 | `30` | qwenエンジン時のみ有効 |
| `cfg` | float | 任意 | `4.0` | qwenエンジン時のみ有効 |

**レスポンス例**

```json
{
  "mode": "outpaint",
  "engine": "qwen",
  "feather": 64,
  "width": 1280,
  "height": 720,
  "paste_box": [400, 0, 880, 720],
  "seed": 42,
  "elapsed_s": 22.1,
  "peak_vram_gb": 39.4,
  "image_url": "/outputs/outpaint_20260724_145837_b3f25c7b.png",
  "raw_inpaint_url": "/outputs/inpaint_20260724_145836_xxxxxxxx.png"
}
```

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/outpaint \
  -F "image=@square.png" -F "width=1280" -F "height=720" -F "seed=42"
```

**特記事項**: `paste_box` は元画像のフィット配置矩形(left, top, right, bottom)。
`raw_inpaint_url` は中央再合成前の生インペイント結果(デバッグ用)。フェザー帯より
内側は元画像とピクセル完全一致することを実機検証済み(CLAUDE.md 54番)。
`peak_vram_gb` は共有VAEのtiled化(`DS_QWEN_TILED_VAE`既定`1`、CLAUDE.md 56番)後の
実測値(旧実測46.3GBから約7GB削減、48GB専有機でのOOM対策)。

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

## scene_angles

1枚のシーン画像から「カメラ指示プロンプト8種のEdit」で同一シーンの8アングル画像を
生成するアプリケーション(2026-07-24追加)。`/api/scene_angles/` 配下。
ComfyUIワークフロー `templates-1_click_multiple_scene_angles-v1.0_api.json` のdiffusers版。
パイプラインは charsheet と同一(edit_angles系グループ、`DS_CHARSHEET_METHOD` /
`CHARSHEET_EDIT_SIZE` に従う)。ジョブ式(バックグラウンド実行、scene_anglesジョブ同士は
同時1件。charsheetジョブとはGPUロックの取得待ちで自然に直列化される)。

アングルID(8種): `close_up`, `wide_angle`, `right_90`, `right_45`, `aerial`,
`low_angle`, `left_90`, `left_45`

### POST /api/scene_angles/generate

Content-Type: `multipart/form-data`。

| 名前 | 型 | 必須/任意 | 既定値 | 説明 |
|---|---|---|---|---|
| `image` | file | 必須 | - | 入力シーン画像(内部で~1MP相当へスケール) |
| `seed` | int | 任意 | `0` | - |
| `angles` | string | 任意 | `""`(=8種全部) | カンマ区切りのアングルID。未知IDは400。生成順はID指定順でなく定義順に正規化 |

**レスポンス例**: `{ "job_id": "3619953819b2" }`

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/scene_angles/generate \
  -F "image=@scene.png" -F "seed=42" -F "angles=aerial,close_up"
```

### GET /api/scene_angles/jobs/{job_id}

ジョブ状態(charsheetの `GET /jobs/{job_id}` と同形式のサブセット。
`views[].key` はアングルID、`sheet_url`/`zip_url`/`refine_error` は無い)。

```json
{
  "job_id": "3619953819b2",
  "status": "done",
  "progress": 8,
  "total": 8,
  "seed": 42,
  "views": [
    {"key": "close_up", "label_ja": "クローズアップ", "label_en": "Close-up",
     "status": "done", "url": "/api/scene_angles/jobs/3619953819b2/images/close_up.png"}
  ],
  "error": null,
  "created_at": "2026-07-24T09:51:24.000000",
  "load_info": {"loaded": true, "method": "bf16-group", "angles_lora": true}
}
```

`status`: `queued` / `running` / `done` / `error`。ディスク復元(charsheetの
`_restore_job_from_disk` 相当)は未実装(サーバ再起動後は既存ジョブの状態参照不可、
画像ファイル自体は `outputs/scene_angles/{job_id}/` に残る)。

### GET /api/scene_angles/jobs/{job_id}/images/{key}.png

各アングルの生成画像(`image/png`)。無ければ404。

### GET /api/scene_angles/jobs/{job_id}/input.png

前処理済み入力画像(`image/png`)。

### GET /api/scene_angles/angles

利用可能なアングルIDの一覧(GPU不使用): `{"angles": [{"key","label_ja","label_en"}, ...]}`

**特記事項**: プロンプトは Qwen-Edit-2509-Multiple-angles LoRA のトリガー文
(ComfyUIワークフローのノード66〜73と同一文言)。実機の目視では close_up / wide_angle /
aerial / low_angle / 45度系は高品質で成立するが、**90度回転系はロール回転
(画像の横倒し)として誤解釈されることがある**(2511ベース+2509用LoRAの既知の癖)。

---

## tpose(Tポーズ4ビュー)

1枚のキャラクター画像から **Tポーズ(両腕を水平に広げた姿勢)の4ビュー**
(正面 / 背面 / 左前45度 / 右前45度)を生成するアプリケーション(2026-07-26追加)。
`/api/tpose/` 配下。image-3d のマルチビュー入力(Hunyuan3D-2mv)と rig-service の
自動リグ/VRM化(Tポーズ前提)向け。

- パイプラインは **通常 Edit(`mode="edit"`、fp8-lightning、4steps・cfg1.0・既定1024²)**。
  charsheet/scene_angles と違い **Multiple-angles LoRA は使わない**(Tポーズでは通常Editの
  方が同一性・速度とも優位: 実測 5〜11秒/枚 vs 42〜46秒/枚、angles LoRAは背面で頭部が
  黒髪へ変質した)。解像度は `DS_TPOSE_SIZE` で変更可。
- **2段生成**: front を最初に生成し、他ビューは「生成した front + 元画像」を参照して
  連鎖生成する(前後で帽子・しっぽ等の造形を揃えるため)。
- ジョブ式(バックグラウンド実行、tposeジョブ同士は同時1件。他アプリとはGPUロックの
  取得待ちで自然に直列化)。

ビューID(5種): `front`, `back`, `left`, `right`(以上 `for_3d: true`)、
`front_left_45`(参考出力、`for_3d: false`)。**45度は左のみ**(モデルが45度の左右を
区別できず同じ絵が出るため、`front_right_45` は2026-07-29に削除した)

### POST /api/tpose/generate

Content-Type: `multipart/form-data`。

| 名前 | 型 | 必須/任意 | 既定値 | 説明 |
|---|---|---|---|---|
| `image` | file | 必須 | - | 入力キャラクター画像(全身/胸像どちらでも可。内部で~1MP相当へスケール) |
| `tail_ref` | file | 任意 | なし | しっぽ形状の参照画像。front以外で2枚目の参照に使う。**非推奨**(顔・毛色まで参照画像側へ引きずられる) |
| `seed` | int | 任意 | `0` | 0=ビューごとにランダム |
| `views` | string | 任意 | `""`(=4種全部) | カンマ区切りのビューID。未知IDは400。生成順は定義順に正規化(frontが先) |
| `subject` | string | 任意 | `"auto"` | 被写体タイプ。`auto`=中立(毛皮/肉球/髪のどの語彙も使わない)/ `animal`=動物・ぬいぐるみ(fur/paws/paw pads)/ `human`=人物・リアルな人形(hair/hands/fingers)。他の値は400 |
| `palms` | string | 任意 | `"forward"` | `forward` は「手のひらを正面へ向け、その内側全面が見えるように」まで指示する(補強句がないと seed 次第で手が下向きになる実測があったため)。| `forward`(手のひらをカメラへ=リグ用Tポーズの標準)/ `natural`(指示しない)。他の値は400 |
| `fur_color` | string | 任意 | `""` | 毛色の色名(例 `cream white`)。空なら **`subject` が `animal` に解決されるときだけ**入力画像から自動推定する(人物・中立では推定しない: 服や肌が低彩度な人物キャラでは `cream white` 等に誤推定し、背面の後頭部が白くなる実バグがあった)(rembgで被写体マスクを取り、彩度の低い画素の中央値を色名へ写す)。推定結果はジョブJSONの `fur_color_detected` に入る |
| `claws` | string | 任意 | `"none"` | 爪。`none`=爪なし(ぬいぐるみでは自然)/ `auto`=参照画像に任せる / 自由記述(例 `short white claws`)。`subject` が `animal` に解決されるときのみ有効 |
| `paw_pads` | string | 任意 | `"auto"` | 肉球の色などの自由記述(例 `pink`)。`subject` が `animal` に解決されるときだけプロンプトへ入る。`none`=肉球に言及しない(`subject=auto` なら `human` 扱い)、色の明示指定は `subject=auto` でも `animal` 扱いになる |
| `tail` | string | 任意 | `""` | しっぽ形状の自由記述(例 `a long fluffy tail with a black tip, hanging down`)。`none`=しっぽなし、空/`auto`=指定なし |
| `body` | string | 任意 | `""` | 体型の自由記述(例 `short stubby legs and a large head`)。**脚が伸びる劣化への主要な対処**。1段目(元画像からポーズを変える段)のプロンプトにのみ入る |
| `costume` | string | 任意 | `""` | **背面から見た衣装**の自由記述(背面ビューのプロンプト末尾にのみ入る)。前開きのベスト・カーディガンが「背中にも前開きで描かれる」問題への対処。**丈・範囲まで書くこと**(例 `the short cream lace bolero ends at the waist and its back is one continuous piece of lace, the white pencil skirt below it is unchanged`。「背中を一枚で覆う」だけだと膝丈のワンピースへ伸びた) |
| `extra_prompt` | string | 任意 | `""` | プロンプト末尾への追記 |
| `recolor` | string | 任意 | `""` | 生成後に色を調整する**2パス目のEdit指示**(空なら実行しない)。例 `Make the fur a warmer cream tone with richer shading`。全ビューへ同じ指示を適用し、正面は調整後の画像を後続ビューの参照に使う(色をビュー間で揃えるため)。生成回数が2倍になる |
| `bg_method` | string | 任意 | `"anime"` | 背景除去の方式(`anime`=アニメ・キャラクター向け / `rembg`=汎用)。Tポーズは被写体がキャラクターのため既定を `anime` にしている(淡い色の毛の取りこぼしが実測 27,232px → 13,733px) |
| `remove_bg` | bool | 任意 | `false` | 背景除去(rembg / isnet-general-use)。各ビューの背景透過版 `<key>_nobg.png`(RGBA)を併産する(白背景版はそのまま残る)。生成後・GPUロック解放後にCPUで処理するためGPU待ちなし(1枚1秒前後) |

**レスポンス例**: `{ "job_id": "d3e3ee58a29c" }`

**curlサンプル**

```bash
curl -X POST http://localhost:8601/api/tpose/generate \
  -F "image=@character.png" -F "seed=42" -F "paw_pads=pink" \
  -F "tail=a long fluffy tail with a black tip, hanging down"
```

### GET /api/tpose/jobs/{job_id}

```json
{
  "job_id": "d3e3ee58a29c",
  "status": "done",
  "progress": 4,
  "total": 4,
  "seed": 42,
  "params": {"palms": "forward", "paw_pads": "pink", "tail": "...",
             "extra_prompt": "", "tail_ref": false, "size": 1024},
  "views": [
    {"key": "front", "label_ja": "正面", "label_en": "Front", "for_3d": true,
     "status": "done",
     "url": "/api/tpose/jobs/d3e3ee58a29c/images/front.png",
     "download_url": "/api/tpose/jobs/d3e3ee58a29c/download/front.png",
     "prompt": "The character stands upright facing directly toward the camera, ..."}
  ],
  "zip_url": "/api/tpose/jobs/d3e3ee58a29c/download.zip",
  "error": null,
  "created_at": "2026-07-26T22:31:00.000000",
  "load_info": {"loaded": true, "method": "edit", "angles_lora": false}
}
```

`for_3d`: Hunyuan3D-2mv のビュースロット(front/left/back/right)へそのまま渡してよいか。
45度ビューは `false`(参考出力。left/rightスロットへ入れるとカメラ事前分布を誤らせる)。
各ビューには `remove_bg=true` のとき `nobg_url` / `nobg_download_url` も入る。
`has_prev` は追加Edit(下記 `/edit`)で退避された1世代前の画像があるか(undo可能か)。
ビューの `status` は `queued` / `running` / `recoloring`(2パス目実行中) / `done` / `error`。
ジョブの `status`: `queued` / `running` / `removing_bg` / `done` / `error`。ディスク復元は未実装
(サーバ再起動後は状態参照不可、画像は `outputs/tpose/{job_id}/` に残る)。

### GET /api/tpose/jobs/{job_id}/images/{key}.png

各ビューの生成画像(`image/png`、inline表示用)。無ければ404。
`{key}` に `_nobg` を付けると背景透過版(`remove_bg=true` 時のみ存在)。

### GET /api/tpose/jobs/{job_id}/download/{key}.png

各ビューの**個別ダウンロード**(`Content-Disposition: attachment;
filename="tpose_{key}_{job_id}.png"`)。`{key}` に `_nobg` を付けると背景透過版。

### GET /api/tpose/jobs/{job_id}/download.zip

生成済み全ビュー(+ `remove_bg=true` なら `<key>_nobg.png` も)+ `input.png` の
ZIP一括ダウンロード(`filename="tpose_{job_id}.zip"`)。ジョブ完了時に生成される。

### GET /api/tpose/jobs/{job_id}/input.png

前処理済み入力画像(`image/png`)。

### POST /api/tpose/jobs/{job_id}/edit

生成済みビューへ**追加のEdit**をかける(何度でも呼べる汎用編集。色調整に限らず
「帽子を外す」等にも使える)。生成時の `recolor` と同じ2パス目の仕組みを独立させたもの。
Content-Type: `multipart/form-data`。

| 名前 | 型 | 必須/任意 | 既定値 | 説明 |
|---|---|---|---|---|
| `prompt` | string | 必須 | - | 修正指示。空・空白のみは400 |
| `views` | string | 任意 | `""`(=完了済み全ビュー) | カンマ区切りのビューID。このジョブに無いIDは400 |
| `seed` | int | 任意 | `0` | 0=ランダム |
| `keep_pose` | bool | 任意 | `true` | trueなら「ポーズ・画角・背景と、それ以外の細部は変えない」を**前置き**し、指示文を末尾に置く(2026-07-28修正。旧実装は1パス目用の構図指示 `plain white background, full body visible from head to toe` と `design ... exactly the same` を**末尾**に置いており、部分編集が壊れて髪色を変えると髪型まで変わっていた) |
| `use_reference` | bool | 任意 | `false` | trueなら**元画像を2枚目の参照として渡す**(「元の髪型に戻す」等で元の見た目を参照できる)。ただし**元画像のポーズ・背景まで引き戻す事故**がある(実測で3seed中1つがTポーズを失い自然な立ち姿へ戻り、別seedでは背景が市松模様になった)ため既定OFF |

**レスポンス**: `{"job_id": "...", "views": ["front","back"], "status": "editing"}`。
進行状況は `GET /jobs/{job_id}`(ジョブ `status="editing"`、各ビュー `status="editing"`)で
ポーリングする。直前の画像は `<key>_prev.png` へ1世代退避され、**透過版を持つビューは
Edit後に作り直す**。ZIPも再生成される。編集の失敗はジョブの `edit_error` に入る。
別の生成/編集が実行中なら409。

### POST /api/tpose/jobs/{job_id}/upscale

生成済みビューを **2048へアップスケール**する(2026-07-28追加)。
Content-Type: `multipart/form-data`。

| 名前 | 型 | 必須/任意 | 既定値 | 説明 |
|---|---|---|---|---|
| `views` | string | 任意 | `""`(=完了済み全ビュー) | カンマ区切りのビューID |
| `target` | int | 任意 | `2048` | 出力の長辺(1024〜4096。範囲外は400) |

**Real-ESRGAN x2(RRDBNet、`core/upscale.py`、spandrel 経由)による決定論的な拡大**で、
拡散モデルでの再生成ではないため**内容は書き換わらない**(髪型・衣装がドリフトしない)。
1024版はそのまま残り `<key>_2048.png` が追加される。透過版を持つビューは
**2048で切り抜きを作り直す**。ZIP・個別ダウンロード・画像配信(`{key}_2048` /
`{key}_2048_nobg`)にも対応。元画像を `/edit`・`/undo` で変えると古い2048版は自動破棄される。
実測: 1024→2048 が **2.0秒/枚・ピークVRAM 4.09GB**。重みは `ai-forever/Real-ESRGAN` の
`RealESRGAN_x2.pth`(64MB、`DS_UPSCALE_MODEL` でローカルパスに差し替え可)。

**レスポンス**: `{"job_id": "...", "views": ["front","back"], "status": "upscaling", "target": 2048}`。
ジョブ `status="upscaling"`、失敗は `upscale_error` に入る。別の生成/編集が実行中なら409。

### POST /api/tpose/jobs/{job_id}/undo

直前のEditを取り消す(`<key>_prev.png` から復元、1世代のみ)。Form: `views`
(任意、省略=退避のある全ビュー)。復元対象が無ければ409。透過版とZIPも作り直す。
**レスポンス**: `{"job_id": "...", "restored": ["back"]}`

### GET /api/tpose/views

利用可能なビューID・しっぽ/体型プリセット・palms/subjectモードの一覧(GPU不使用):
`{"views": [{"key","label_ja","label_en","for_3d"}, ...], "tail_presets": [...],
"body_presets": [...], "palms_modes": ["forward","natural"],
"subject_modes": ["auto","animal","human"], "claws_modes": ["none","auto"]}`

**特記事項(実機検証で確定した制約)**

- **真横(`left` / `right`)は2026-07-29に追加**(image-3d が side view に対応したため)。
  **このビューだけ腕の姿勢が違う**: Tポーズのまま(腕を左右へ広げたまま)の真横は、
  手前の腕がカメラをまっすぐ指す極端な短縮になり**描けない**(胴体は綺麗に回り込むのに
  腕だけが長い管・棒になって画面外へ伸びる。人物のように腕が長いほど顕著で、
  angles LoRA でも同じ)。腕を下ろすと綺麗になるが胴体の側面が隠れるため、
  **真横では腕を前方へ水平に出す**(胴体の側面シルエットが隠れず、極端な短縮も不要)。
  3D再構成へ渡す際は**真横だけ腕の向きが他ビューと異なる**点に注意すること。
  **image-3d へ渡すのは front / back / left / right の4枚**
  (45度ビューは `for_3d: false` なので left/right スロットへ入れないこと)。
- **しっぽ形状は入力画像から推定できない**。未指定だとビューごとに別形状が創作される
  (実測でポンポン/なし/長い尻尾にばらけた)ため、入力画像にしっぽが写っていない場合は
  `tail` で明示すること。しっぽの記述が短すぎると色・大きさがドリフトする
  (`a long fluffy black-tipped tail` では全体が黒い巨大なしっぽになった)ため、
  プリセット相当の具体的な記述(`a long fluffy tail with a black tip, hanging down`)を推奨。
- **体型がドリフトする(脚が長くなる)**: ポーズ変更時にモデルが人型寄りの比率へ引っ張られる。
  肩ライン比(肩の高さ/全高)の実測で 0.45 相当が 0.366 まで伸びた。汎用的な
  「体型を維持せよ」文やキャンバス比の変更では改善せず、**`body` で体型を具体的に
  言語化するのが唯一効いた対処**(0.401〜0.431へ回復)。詳細は
  `apps/tpose/prompts.py` の「脚が伸びる問題」コメント参照。
- 胸像入力からでも全身Tポーズを生成できるが、写っていない部分(脚部の衣装・靴等)は
  モデルが創作する。
- **爪は既定で出さない(`claws="none"`)**: 背面プロンプトに `with their claws` と
  書いていたため黒い爪が目立つ出力になっていた(削除済み)。加えて爪の抑制は
  **肯定形でしか効かない**: `"...without claws"`(否定形)では爪が残り、
  `"the paw tips are soft, round and smooth"`(肯定形)で消える(実測)。
  爪を出したい場合は `claws=auto` または自由記述を指定する。
- **生成画像は参照元より明るい(実測)**: 参照元(輝度中央値112 / 白に近い画素1.9%)に対し
  生成は正面155/2.2%・背面184/10.1%。これが背景除去で淡い部位が消える一因。1パス目の
  プロンプトで色調を保持させる案は**爪抑制文を末尾から押し出して爪が戻るため採用していない**
  (`extra_prompt` も爪抑制文より前に挿入される)。切り抜き側では対処済みで、
  **色そのものを変えたい場合は `recolor`(2パス目のEdit)を使う**: 実測で
  「Make the fur a warmer cream tone with richer shading and slightly deeper contrast」
  により4ビューとも 輝度中央値 88〜97(参照元99に接近)・白に近い画素 1.7〜2.0%
  (最大10%から改善)・切り抜き取りこぼし 368〜2,166px になった。ただし
  **強い指示は同一性も動かす**(毛色が黄褐色へ寄り、耳の斑や爪の描写も変化した)ため
  文言の強さはユーザー側で調整すること。所要時間は4ビュー+背景削除で157秒(2パスなし64秒)。
- **爪を消すと後足の指の分離まで失われる問題への対処**: 上記の抑制文だけだと足が
  ミトン状になる(足領域の内部勾配 1151px→965px)。「同じ毛色で」のような**相対表現では
  効かず**、`each toe is <色名> fur right to the tip` のように**具体的な色名**を書くと
  「指は分離・爪は無い」を両立できた(実測: 爪 889px / 指 1401px)。色名は `fur_color`
  未指定時に入力画像から自動推定する。背面ビューは踵側が見えるため指が写らないのが正常。
- **背景透過版(`remove_bg=true`)の後処理**: rembgのアルファをそのまま使うと
  (a) 内部に穴が空く(帯の間が alpha≈160 になり暗い色が透ける)、(b) RGBが濁る、
  (c) **淡い色の部位が消える**(背面ビューで白い毛の手が実測40,072px=被写体の約10%
  落ちた)という3つの問題が出た。背景が**生成された白一色**であることを利用して
  アルファを組み立て直している: 画像の境界から連結した白だけを背景とし(被写体に
  囲まれた白飛び画素は背景にしない)、rembgが落とした明るい画素(輝度>=245)を
  「淡い毛」として回収(影は輝度が低いので巻き込まない。実測 影≈148 / 淡い毛≈253)、
  rembgの二値化閾値も 128→64 に下げる(淡い部位でrembgは薄いゴーストしか返さないため)。
  最終マスクは最大連結成分+穴埋めで確定し、約1pxのぼかしで縁を整える
  (**rembgの半透明マットは使わない**)。実測: 背面の取りこぼし 17,338px → 774px(-96%)、
  代償は輪郭の純白の縁 約2,353px(約1px幅)。細い毛の房(しっぽ)は従来同様に保持される。
- **被写体タイプの指定を誤ると語彙が悪影響する**: `subject=animal` は fur/paws/paw pads を
  プロンプトへ入れるため、リアルな人形・人物に使うと背面が動物化し手に肉球が付く。
  既定 `auto` は中立語彙で両方に安全(実測: リアルな人形で背面が髪・手の甲になり肉球なし、
  ぬいぐるみでも背面の毛並みを維持)。ただし `animal` を明示した方が背面の
  「手の甲と爪・肉球は見えない」まで正しく描かれる。
- 前後のシルエット一致度は良好(実測: 幅/高比 1.009 vs 1.007、腕ラインの相対高
  0.435 vs 0.405、bbox高 970 vs 971 px)。

---

## ユーティリティ

### POST /api/remove_bg

背景削除(GPU不使用・排他不要)。Content-Type: `multipart/form-data`。
実装は `core/bg.py`(charsheet / Tポーズと共通)。

| 名前 | 型 | 必須/任意 | 説明 |
|---|---|---|---|
| `image` | file | 必須 | 入力画像(内部で`RGBA`変換) |
| `method` | string | 任意(既定 `rembg`) | `rembg`=汎用(rembg / isnet-general-use)/ `anime`=アニメ・キャラクター向け(SkyTNT/anime-segmentation の ISNet、`skytnt/anime-seg` の `isnetis.onnx`、Apache-2.0)。未知の値は既定へフォールバック。レスポンスに実際に使った `method` を含む |

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

**部分解放時の共有コンポーネント解放(2026-07-24変更、CLAUDE.md 56番)**:
`target="t2i"` 等 qwen_image 内の個別グループを指定した場合、解放後に
qwen_imageファミリー内の他のどのグループ(t2i/edit/edit_angles/edit_angles_bf16/
edit_angles_bf16group/layered)もロードされていなければ、共有コンポーネント
(vae/text_encoder/tokenizer、~15.7GB)も併せて解放する(旧実装は`target="all"`
指定時のみ共有を解放しており、個別指定では常駐したまま残っていた)。

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
| `DS_TERMINAL_PROGRESS` | 全生成系(API応答自体は無変更) | `0`(既定)\| `1`(2026-07-24追加)。`1`でサーバ起動ターミナル(stderr)へ進捗バーを描画するのみで、レスポンスJSON・挙動・速度には影響しない(CLAUDE.md 55番) |

詳細な既定値・後方互換の旧環境変数名・調整根拠は `README.md` の「環境変数一覧」表を参照。
