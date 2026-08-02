"""背景削除済みRGBA画像に残った白背景を追加で透明化する。

自動処理は既存の透明領域に接している近似白だけを対象にする。髪などで完全に
囲まれた領域は、UIでクリックされた座標を起点に連結領域を選択して除去する。
"""
from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image


def _flood(mask: np.ndarray, seeds: list[tuple[int, int]]) -> np.ndarray:
    """4近傍で seeds から到達できる mask=True の領域を返す。"""
    height, width = mask.shape
    selected = np.zeros_like(mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for x, y in seeds:
        if 0 <= x < width and 0 <= y < height and mask[y, x] and not selected[y, x]:
            selected[y, x] = True
            queue.append((x, y))
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if (0 <= nx < width and 0 <= ny < height
                    and mask[ny, nx] and not selected[ny, nx]):
                selected[ny, nx] = True
                queue.append((nx, ny))
    return selected


def refine_white_alpha(
    image: Image.Image,
    *,
    white_threshold: int = 250,
    auto: bool = True,
    points: list[tuple[int, int]] | None = None,
    color_tolerance: int = 12,
    feather: float = 0.8,
) -> tuple[Image.Image, int]:
    """近似白の自動選択とクリック色の連結選択を透明化する。"""
    if not 0 <= white_threshold <= 255:
        raise ValueError("white_threshold は 0〜255 で指定してください")
    if not 0 <= color_tolerance <= 255:
        raise ValueError("color_tolerance は 0〜255 で指定してください")
    if not 0 <= feather <= 8:
        raise ValueError("feather は 0〜8 で指定してください")

    rgba = np.asarray(image.convert("RGBA")).copy()
    rgb = rgba[..., :3].astype(np.int16)
    alpha = rgba[..., 3]
    visible = alpha > 0
    selected = np.zeros(alpha.shape, dtype=bool)

    if auto:
        near_white = visible & np.all(rgb >= white_threshold, axis=2)
        transparent = ~visible
        adjacent = np.zeros_like(transparent)
        adjacent[1:] |= transparent[:-1]
        adjacent[:-1] |= transparent[1:]
        adjacent[:, 1:] |= transparent[:, :-1]
        adjacent[:, :-1] |= transparent[:, 1:]
        edge = np.zeros_like(transparent)
        edge[[0, -1], :] = True
        edge[:, [0, -1]] = True
        ys, xs = np.nonzero(near_white & (adjacent | edge))
        selected |= _flood(near_white, list(zip(xs.tolist(), ys.tolist())))

    height, width = alpha.shape
    for x, y in points or []:
        if not (0 <= x < width and 0 <= y < height) or not visible[y, x]:
            continue
        target = rgb[y, x]
        similar = visible & np.all(np.abs(rgb - target) <= color_tolerance, axis=2)
        selected |= _flood(similar, [(x, y)])

    removed = int(np.count_nonzero(selected & visible))
    if not removed:
        return Image.fromarray(rgba, "RGBA"), 0

    if feather > 0:
        try:
            from scipy import ndimage
            softened = ndimage.gaussian_filter(
                selected.astype(np.float32), sigma=float(feather))
            rgba[..., 3] = np.minimum(
                alpha.astype(np.float32), (1.0 - softened) * 255.0
            ).clip(0, 255).astype(np.uint8)
            rgba[..., 3][selected & (softened >= 0.95)] = 0
        except ImportError:
            rgba[..., 3][selected] = 0
    else:
        rgba[..., 3][selected] = 0
    return Image.fromarray(rgba, "RGBA"), removed
