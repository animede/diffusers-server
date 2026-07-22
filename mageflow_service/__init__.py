# -*- coding: utf-8 -*-
"""Mage-Flow ラッパーサービス(専用venv venv-mageflow/ で動く独立FastAPIプロセス)。

本体サーバ(app.py、venv/)からは import しないこと — torch/transformers の
バージョンが衝突する(CLAUDE.md 50番)。本体からは HTTP(core/mageflow.py)経由で叩く。
"""
