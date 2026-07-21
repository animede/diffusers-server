# -*- coding: utf-8 -*-
"""
LLM(llama.cpp server 等、OpenAI互換 /v1/chat/completions)によるプロンプト支援。

- GPU(画像生成用)とは無関係の別プロセス(ユーザーが別途起動する llama.cpp server 等)を
  HTTP経由で呼ぶだけなので、core.gpu.generation_lock は使わない(排他不要、非GPU)。
- 接続先は環境変数 DS_LLM_URL で指定する(既定 http://127.0.0.1:64652)。
  llama.cpp server のポートは環境によって変わりうるため、ユーザーが起動しているポートに
  合わせて DS_LLM_URL を設定すること(例: DS_LLM_URL=http://127.0.0.1:8080)。
- llama.cpp server は OpenAI互換 /v1/chat/completions を持つ前提(model名は無視されるため
  "default" 固定でよい)。
- 接続不可・タイムアウト時は LLMConnectionError を送出し、呼び出し側(app.py)で
  日本語エラーメッセージに変換して 502 を返す。画像生成機能には一切影響しない。
"""
import os

import requests

__all__ = ["LLMConnectionError", "get_llm_url", "chat_completion", "enhance_prompt", "translate_prompt"]

DEFAULT_LLM_URL = "http://127.0.0.1:64652"
LLM_TIMEOUT_S = 60


class LLMConnectionError(Exception):
    """LLMサーバに接続できない、またはLLMサーバがエラーを返した場合。"""


def get_llm_url() -> str:
    """DS_LLM_URL(既定 http://127.0.0.1:64652)。末尾の / は取り除く。"""
    return os.environ.get("DS_LLM_URL", DEFAULT_LLM_URL).rstrip("/")


def chat_completion(system_prompt: str, user_text: str, *, temperature: float = 0.7) -> str:
    """OpenAI互換 /v1/chat/completions を呼び、assistant の応答テキストを返す。

    接続不可・タイムアウト・非2xx応答・レスポンス形式異常はすべて LLMConnectionError に
    まとめる(呼び出し側は個別のHTTPエラー種別を気にせず 502 に変換できる)。
    """
    url = f"{get_llm_url()}/v1/chat/completions"
    payload = {
        "model": "default",  # llama.cpp server はmodel指定を無視して動く
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": temperature,
    }
    try:
        res = requests.post(url, json=payload, timeout=LLM_TIMEOUT_S)
    except requests.exceptions.ConnectionError as exc:
        raise LLMConnectionError(
            f"LLMサーバに接続できません(DS_LLM_URL={get_llm_url()} を確認してください): {exc}"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise LLMConnectionError(
            f"LLMサーバへの接続がタイムアウトしました(DS_LLM_URL={get_llm_url()}): {exc}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise LLMConnectionError(f"LLMサーバへのリクエストに失敗しました: {exc}") from exc

    if res.status_code != 200:
        raise LLMConnectionError(
            f"LLMサーバがエラーを返しました(status={res.status_code}): {res.text[:300]}"
        )

    try:
        data = res.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise LLMConnectionError(f"LLMサーバの応答形式が不正です: {exc}") from exc

    return (content or "").strip()


ENHANCE_SYSTEM_PROMPT = (
    "You are a prompt engineer for text-to-image diffusion models. "
    "Rewrite the user's input into a single, detailed English prompt suitable for an image "
    "generation model. Preserve the user's original intent, subject, and composition — do not "
    "change what is being depicted. Add helpful details such as composition, lighting, mood, "
    "and quality descriptors where appropriate. "
    "Style consistency: if the input mentions an art style (e.g. anime, watercolor, oil "
    "painting), every descriptor you add — including background, lighting, and quality tags — "
    "must match that style, and never mix in contradicting terms (no photorealistic/8k "
    "photo/DSLR for anime inputs). If the input does NOT mention any style, you MUST NOT add "
    "any style word at all (no anime, no photorealistic, no watercolor, no 'trending on' "
    "sites) — describe only subject details, lighting, mood, and generic quality terms. "
    "Do not invent scene elements the user did not mention: if the input does not specify a "
    "background or setting, leave it unspecified and let the image model decide — only enrich "
    "the elements that are already present. If the input is in Japanese "
    "(or any non-English language), translate it into English as part of the rewrite. "
    "Examples:\n"
    "Input: アニメ風の銀髪の少女\n"
    "Output: anime style, a silver-haired girl, cel shading, vivid colors, detailed anime "
    "illustration, high quality\n"
    "Input: 銀髪の少女\n"
    "Output: a silver-haired girl, delicate facial features, flowing silver hair, soft "
    "lighting, highly detailed, high quality\n"
    "Output ONLY the final prompt text itself — no preamble, no explanation, no quotes, "
    "no labels like 'Prompt:'."
)

TRANSLATE_SYSTEM_PROMPT = (
    "You are a translator specialized in text-to-image generation prompts. "
    "Translate the user's Japanese text into natural, fluent English suitable for use as an "
    "image generation prompt. Keep the translation faithful to the original meaning — do not "
    "over-interpret, embellish, or add details that are not present in the source text. "
    "Output ONLY the translated text itself — no preamble, no explanation, no quotes, "
    "no labels like 'Translation:'."
)


def enhance_prompt(text: str) -> str:
    return chat_completion(ENHANCE_SYSTEM_PROMPT, text)


def translate_prompt(text: str) -> str:
    return chat_completion(TRANSLATE_SYSTEM_PROMPT, text)
