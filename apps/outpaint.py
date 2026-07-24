# -*- coding: utf-8 -*-
"""
アウトペイント(既存インペイント機構の流用による画角拡張)のヘルパー(2026-07-24追加、
CLAUDE.md 54番)。

原理:
  1. ターゲットキャンバス中央に入力画像をアスペクト維持でフィット配置
     (scale = min(tw/sw, th/sh)。縦長→横長なら左右に帯、横長→縦長なら上下に帯)
  2. 帯領域をエッジ延長(端の行/列を引き伸ばし)で初期充填
  3. 帯=白(255)+フェザー(元画像側へ feather px 食い込む白→黒勾配)のマスクを生成
  4. 既存インペイント(/api/inpaint 等)で帯+フェザー部を生成
  5. 最後に元画像の中央部をピクセルそのまま再合成(中央無傷を保証)

A/B検証(2026-07-24)の結論: エンジンは Qwen ControlNet Inpainting(fp8-lightning
構成でも cfg4.0 の通常denoise)が明確に優位。Z-Image-Turbo は元シーンの文脈を無視して
無関係な背景を描き境界が断絶するため不採用(選択肢としては残す)。フェザーは64pxを既定
(0でもほぼ破綻しないが、64が最も継ぎ目が滑らか)。
"""
from PIL import Image


def build_outpaint_canvas(
    src: Image.Image,
    target_width: int,
    target_height: int,
    feather: int,
) -> "tuple[Image.Image, Image.Image, tuple[int, int, int, int]]":
    """キャンバス(エッジ延長充填済み)・マスク(白=生成領域)・中央配置box を作る。

    戻り値: (canvas RGB, mask L, paste_box)
        paste_box: フィット配置した元画像の (left, top, right, bottom)。
                   生成後の中央再合成に使う。
    """
    src = src.convert("RGB")
    tw, th = int(target_width), int(target_height)
    if tw <= 0 or th <= 0:
        raise ValueError("width/height は正の整数が必要です")

    scale = min(tw / src.width, th / src.height)
    new_w = max(1, round(src.width * scale))
    new_h = max(1, round(src.height * scale))
    fitted = src.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (tw - new_w) // 2
    top = (th - new_h) // 2

    canvas = Image.new("RGB", (tw, th))
    # 初期充填: 帯方向のエッジ行/列を引き伸ばす(denoise の初期値として自然な色分布を与える)
    if left > 0:  # 左右に帯
        left_col = fitted.crop((0, 0, 1, new_h)).resize((left, new_h))
        right_w = tw - (left + new_w)
        canvas.paste(left_col, (0, top))
        if right_w > 0:
            right_col = fitted.crop((new_w - 1, 0, new_w, new_h)).resize((right_w, new_h))
            canvas.paste(right_col, (left + new_w, top))
    if top > 0:  # 上下に帯
        top_row = fitted.crop((0, 0, new_w, 1)).resize((new_w, top))
        bottom_h = th - (top + new_h)
        canvas.paste(top_row, (left, 0))
        if bottom_h > 0:
            bottom_row = fitted.crop((0, new_h - 1, new_w, new_h)).resize((new_w, bottom_h))
            canvas.paste(bottom_row, (left, top + new_h))
    canvas.paste(fitted, (left, top))

    # マスク: 帯=255、元画像内は帯に隣接するエッジから feather px の白→黒勾配、中央=0
    feather = max(0, int(feather))
    mask = Image.new("L", (tw, th), 255)
    inner = Image.new("L", (new_w, new_h), 0)
    if feather > 0:
        px = inner.load()
        has_lr = left > 0 or (tw - left - new_w) > 0
        has_tb = top > 0 or (th - top - new_h) > 0
        for y in range(new_h):
            for x in range(new_w):
                d_candidates = []
                if has_lr:
                    d_candidates.append(min(x, new_w - 1 - x))
                if has_tb:
                    d_candidates.append(min(y, new_h - 1 - y))
                if not d_candidates:
                    continue
                d = min(d_candidates)
                if d < feather:
                    px[x, y] = int(255 * (1 - d / feather))
    mask.paste(inner, (left, top))

    return canvas, mask, (left, top, left + new_w, top + new_h)


def recomposite_center(
    generated: Image.Image,
    src: Image.Image,
    paste_box: "tuple[int, int, int, int]",
    feather: int,
    target_width: int,
    target_height: int,
) -> Image.Image:
    """生成結果の中央へ元画像(フィット済み)を貼り戻す(中央無傷の保証)。

    インペイントはマスク=0領域も微小に変化させることがある(VAEの往復誤差等)ため、
    仕上げとして必ず実行する。貼り戻しはフェザー帯でアルファブレンドする:
      - フェザーより内側(帯から feather px 以上離れた中央部)= 元画像ピクセル100%
        (「中央部はピクセルそのまま」の保証範囲)
      - フェザー帯 = 元画像→生成結果への線形ブレンド(矩形エッジで硬い継ぎ目を
        作らないため。生成時のマスク勾配と同じ距離関数を使う)
    """
    left, top, right, bottom = paste_box
    new_w, new_h = right - left, bottom - top
    fitted = src.convert("RGB").resize((new_w, new_h), Image.Resampling.LANCZOS)
    result = generated.convert("RGB").copy()

    feather = max(0, int(feather))
    if feather <= 0:
        result.paste(fitted, (left, top))
        return result

    # アルファ: 中央=255(元画像)、帯に隣接するエッジへ向かって0へ漸減
    # (build_outpaint_canvas() のマスク勾配の反転と同じ距離関数)
    alpha = Image.new("L", (new_w, new_h), 255)
    px = alpha.load()
    has_lr = left > 0 or (target_width - right) > 0
    has_tb = top > 0 or (target_height - bottom) > 0
    for y in range(new_h):
        for x in range(new_w):
            d_candidates = []
            if has_lr:
                d_candidates.append(min(x, new_w - 1 - x))
            if has_tb:
                d_candidates.append(min(y, new_h - 1 - y))
            if not d_candidates:
                continue
            d = min(d_candidates)
            if d < feather:
                px[x, y] = int(255 * (d / feather))
    result.paste(fitted, (left, top), alpha)
    return result
