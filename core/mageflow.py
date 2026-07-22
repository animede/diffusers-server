# -*- coding: utf-8 -*-
"""
Mage-Flow ラッパーサービス(mageflow_service/、専用venv別プロセス)へのHTTPクライアント。

設計は core/llm.py(LLMプロンプト支援、CLAUDE.md 28番)と同じ流儀:
  - 接続先はポートを一切ハードコードせず、環境変数 DS_MAGEFLOW_URL
    (既定 http://127.0.0.1:8602)で指定する。ラッパーの起動ポートを変えたら
    本体サーバ側も DS_MAGEFLOW_URL を合わせて設定すること。
  - 接続不可・タイムアウトは MageFlowConnectionError にまとめ、呼び出し側
    (app.py の /api/mageflow/*)が 502 + 日本語メッセージ(実際の起動コマンド付き)に
    変換する。
  - ラッパー側が返すHTTPエラー(400/500等)は MageFlowServiceError として
    ステータスコードごと透過し、app.py が同じステータスで返す
    (プロキシとしてラッパーのバリデーション結果を握りつぶさないため)。

排他制御(exclusive パラメータ)は app.py 側の責務(このモジュールは純粋なHTTP転送のみ)。
"""
import os

import requests

__all__ = [
    "MageFlowConnectionError",
    "MageFlowServiceError",
    "get_mageflow_url",
    "launch_hint",
    "forward_t2i",
    "forward_edit",
    "forward_status",
    "forward_unload",
]

DEFAULT_MAGEFLOW_URL = "http://127.0.0.1:8602"
# 生成系のタイムアウト。初回はモデルDL(15-20GB)+ロードが乗るため長め。
# それでもHF Hubからの初回ダウンロード(数分〜数十分)には足りないことがあるので、
# 初回は事前に `venv-mageflow/bin/python -c "from mage_flow import MageFlowPipeline; ..."`
# 等でキャッシュを温めておくのが確実(README参照)。
GENERATE_TIMEOUT_S = 300
CONTROL_TIMEOUT_S = 30


class MageFlowConnectionError(Exception):
    """ラッパーサービスに接続できない(未起動・ポート違い・タイムアウト)。"""


class MageFlowServiceError(Exception):
    """ラッパーサービスがHTTPエラーを返した(ステータスコードを保持して透過する)。"""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def get_mageflow_url() -> str:
    """DS_MAGEFLOW_URL(既定 http://127.0.0.1:8602)。末尾の / は取り除く。"""
    return os.environ.get("DS_MAGEFLOW_URL", DEFAULT_MAGEFLOW_URL).rstrip("/")


def launch_hint() -> str:
    """502メッセージに含める起動コマンドのヒント(実際に動くコマンドを案内する)。"""
    return (
        "Mage-Flowサービスが起動していません。diffusers-server ディレクトリで "
        "./run_mageflow.sh を実行するか、"
        "venv-mageflow/bin/python -m uvicorn mageflow_service.app_mageflow:app "
        "--host 127.0.0.1 --port 8602 で起動してください"
        f"(現在の接続先: {get_mageflow_url()}。ポートを変えた場合は "
        "DS_MAGEFLOW_URL を合わせて設定)"
    )


def _handle_response(res: requests.Response) -> dict:
    if res.status_code != 200:
        try:
            detail = res.json().get("detail", res.text[:300])
        except ValueError:
            detail = res.text[:300]
        raise MageFlowServiceError(res.status_code, str(detail))
    try:
        return res.json()
    except ValueError as exc:
        raise MageFlowConnectionError(f"Mage-Flowサービスの応答形式が不正です: {exc}") from exc


def _post(path: str, *, json_body=None, data=None, files=None, timeout=GENERATE_TIMEOUT_S) -> dict:
    url = f"{get_mageflow_url()}{path}"
    try:
        res = requests.post(url, json=json_body, data=data, files=files, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        raise MageFlowConnectionError(launch_hint()) from exc
    except requests.exceptions.Timeout as exc:
        raise MageFlowConnectionError(
            f"Mage-Flowサービスへのリクエストがタイムアウトしました({timeout}s。"
            "初回はモデルのダウンロード+ロードに時間がかかるため、"
            f"しばらく待ってから再試行してください): {exc}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise MageFlowConnectionError(f"Mage-Flowサービスへのリクエストに失敗しました: {exc}") from exc
    return _handle_response(res)


def forward_t2i(payload: dict) -> dict:
    """POST /t2i へJSON転送する。payload はラッパーの T2IRequest と同じキー。"""
    return _post("/t2i", json_body=payload)


def forward_edit(form_data: dict, files: list) -> dict:
    """POST /edit へmultipart転送する。

    files: [("image", (filename, bytes, content_type)), ...] 形式
    (requests の files 引数にそのまま渡す。複数画像は同名フィールドの繰り返し)。
    """
    return _post("/edit", data=form_data, files=files)


def forward_status() -> dict:
    url = f"{get_mageflow_url()}/status"
    try:
        res = requests.get(url, timeout=CONTROL_TIMEOUT_S)
    except requests.exceptions.RequestException as exc:
        raise MageFlowConnectionError(launch_hint()) from exc
    return _handle_response(res)


def forward_unload() -> dict:
    return _post("/unload", timeout=CONTROL_TIMEOUT_S)
