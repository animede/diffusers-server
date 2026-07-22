#!/bin/bash
# Mage-Flow ラッパーサービス起動スクリプト(CLAUDE.md 50番)。
#
# 専用venv(venv-mageflow/、本体venvとは完全隔離)で mageflow_service/app_mageflow.py を
# 起動する。ポートはハードコードせず DS_MAGEFLOW_PORT(既定 8602)で指定する。
# 本体サーバ(app.py)側はこのサービスへ DS_MAGEFLOW_URL(既定 http://127.0.0.1:8602)で
# 接続するので、ポートを変える場合は両方を合わせること:
#   DS_MAGEFLOW_PORT=9000 ./run_mageflow.sh
#   DS_MAGEFLOW_URL=http://127.0.0.1:9000 venv/bin/python -m uvicorn app:app --port 8601
set -eu

cd "$(dirname "$0")"

PORT="${DS_MAGEFLOW_PORT:-8602}"
HOST="${DS_MAGEFLOW_HOST:-127.0.0.1}"

if [ ! -x venv-mageflow/bin/python ]; then
    echo "エラー: venv-mageflow/ がありません。README.md の「Mage-Flow セットアップ」を参照してください。" >&2
    exit 1
fi

echo "Mage-Flow wrapper service: http://${HOST}:${PORT} (venv-mageflow)"
exec venv-mageflow/bin/python -m uvicorn mageflow_service.app_mageflow:app \
    --host "$HOST" --port "$PORT"
