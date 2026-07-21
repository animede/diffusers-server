# -*- coding: utf-8 -*-
"""
モデルファミリーの登録・排他・ロード状態管理。

Phase 0 では families/qwen_image.py / flux2.py の中身は作らない(REBUILD_PLAN §4 Phase 0
の指示どおり)。このモジュールは Phase 1/2 で各ファミリーが実装する共通インターフェースの
抽象基底クラス(ModelFamily)と、ファミリー横断の登録・排他・自動 unload を管理する
FamilyRegistry を提供する。

設計の由来(REBUILD_PLAN §3.1):
  - 「families/* は共通インターフェース(load(mode) / generate(req) / unload() / status())を
    実装。core/registry.py が唯一の窓口で、生成は全ファミリー横断でグローバルロック1本
    (GPU 同時1件)」
  - 「VRAM 予算管理: Qwen 系と FLUX.2 は同時常駐不可とみなし、ファミリー切替時に
    registry が自動 unload(明示 /api/unload も存続)」

元実装(Ｑwenimage-edit-diffusers/pipeline_manager.py)にはファミリー横断の registry は
存在しない(単一ファミリーのみを扱うモジュールのため)。ただし以下のパターンを
そのまま踏襲・一般化した:
  - モジュールレベルの _lock(threading.Lock、非再入)でロード状態を保護するパターン
    (pipeline_manager.py 行663 _lock、および "_locked" 系関数の規約 = 呼び出し側が
    既にロックを保持している前提。CLAUDE.md 13番の「非再入ロックによるデッドロック」の罠)
    -> FamilyRegistry は各ファミリーの load() 呼び出し全体を1本の registry ロックで
    包むことで、ファミリー側の実装者が再入の罠を踏まないようにする。
  - unload(target) が辞書の状態をリセットして gc.collect() + empty_cache() +
    reset_peak_memory_stats() する設計(pipeline_manager.py 行1540-1621)
    -> core.gpu.empty_cache() / reset_peak_stats() を呼ぶ形で一般化。
  - get_status() が全グループの loaded/quant/lora_available 等をまとめて返す設計
    (pipeline_manager.py 行1624-1669)
    -> ModelFamily.status() を各ファミリーが実装し、FamilyRegistry.status_all() が
    ファミリー名をキーにした dict にまとめる。

生成のグローバルロックは core.gpu.generation_lock(全ファミリー共通の1本)をそのまま使う。
registry 自身が持つロック(_registry_lock)はロード/アンロード/ファミリー切替の排他用で、
生成中の排他とは目的が異なる別ロックである(生成ロックとロード状態ロックを分離することで、
「ロード済みファミリーAが生成中でも、まだロードされていないファミリーBの状態問い合わせ
(status)がブロックされない」ようにする)。
"""
import abc
import threading

from core import gpu

__all__ = ["ModelFamily", "FamilyRegistry", "registry"]


class ModelFamily(abc.ABC):
    """モデルファミリーの共通インターフェース。

    Phase 1(qwen_image)/ Phase 2(flux2)はこのクラスを継承し、以下の4メソッドを実装する
    (REBUILD_PLAN §3.1: "load(mode) / generate(req) / unload() / status()")。

    name はファミリーの一意な識別子(例: "qwen_image", "flux2")。FamilyRegistry.register()
    に渡すキーと一致させること。
    """

    name: str = ""

    @abc.abstractmethod
    def load(self, mode: str, **kwargs) -> None:
        """指定モード(例: "t2i" / "edit" / "controlnet" 等、ファミリーごとに定義)の
        コンポーネントを遅延ロードする。既にロード済みなら何もしない(冪等)。

        kwargs はファミリー固有の追加パラメータ(例: qwen_image の "model"、
        flux2 の "model" 等でモデル切替を指定する。タスク1/タスク2で追加)。
        既定では無視してよい(未対応ファミリーは **kwargs を受け取るだけで無視する)。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate(self, request: dict) -> dict:
        """生成を実行する。呼び出し側(registry.generate())が既に generation_lock を
        保持している前提(gpu.generation_scope() 経由)。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def unload(self) -> dict:
        """このファミリーが確保しているコンポーネントを全て解放する。
        解放した対象のリスト等を dict で返す(元実装の unload() の戻り値 {"freed": [...]}
        を踏襲)。
        """
        raise NotImplementedError

    @abc.abstractmethod
    def status(self) -> dict:
        """ロード状態・量子化構成・VRAM 等をまとめた dict を返す(GET /api/status 用)。"""
        raise NotImplementedError

    def is_loaded(self) -> bool:
        """既定実装: status()["loaded"] を見る。ファミリー側で複数グループを持つ場合
        (Qwen 系のように t2i/edit/controlnet/layered が別々に loaded を持つ場合)は
        オーバーライドして「どれか1つでもロード済みなら True」を返すこと。
        """
        return bool(self.status().get("loaded", False))


class FamilyRegistry:
    """ファミリーの登録・排他・ロード状態管理を行う唯一の窓口。

    - register(family): ファミリーを登録する。
    - load(name, mode): 指定ファミリーをロードする。VRAM 同時常駐不可の組(exclusive_groups)
      に属す場合、他の常駐中ファミリーを自動 unload してからロードする
      (REBUILD_PLAN §3.1 の「ファミリー切替時に registry が自動 unload」)。
    - generate(name, request): gpu.generation_scope() 経由で全ファミリー共通の1本の
      生成ロックを取りつつ、指定ファミリーの generate() を呼ぶ。
    - unload(name=None): 指定ファミリー(None なら全ファミリー)を明示 unload する
      (/api/unload 相当、REBUILD_PLAN §3.1 に "明示 /api/unload も存続" とある)。
    - status_all(): 全ファミリーの status() をまとめた dict を返す(GET /api/status 用)。

    自動 unload フック(on_before_load): Phase 0 では実際のファミリー実装が無いため
    呼び出しは発生しない(REBUILD_PLAN §4 Phase 0: "ファミリー切替時の自動 unload フックを
    用意する(実際の呼び出しは Phase 2)")。Phase 2 で families/flux2.py 登録時に
    exclusive_groups=[{"qwen_image", "flux2"}] のように渡せば有効になる。
    """

    def __init__(self):
        self._registry_lock = threading.Lock()
        self._families: "dict[str, ModelFamily]" = {}
        # 同時常駐不可のファミリー名の集合のリスト。例: [{"qwen_image", "flux2"}]
        self._exclusive_groups: "list[set[str]]" = []

    def register(self, family: ModelFamily, exclusive_with: "list[str] | None" = None) -> None:
        """ファミリーを登録する。exclusive_with に他ファミリー名を渡すと、そのファミリーと
        同時常駐不可(切替時に自動 unload される)として扱われる(相互に登録される)。
        """
        with self._registry_lock:
            if not family.name:
                raise ValueError("ModelFamily.name must be a non-empty string")
            self._families[family.name] = family
            if exclusive_with:
                group = {family.name, *exclusive_with}
                self._exclusive_groups.append(group)

    def _families_exclusive_with(self, name: str) -> "set[str]":
        others: "set[str]" = set()
        for group in self._exclusive_groups:
            if name in group:
                others |= (group - {name})
        return others

    def load(self, name: str, mode: str, **kwargs) -> None:
        """指定ファミリーをロードする。呼び出し前に排他グループの相手が常駐していれば
        自動 unload してから load() を呼ぶ(ファミリー切替。REBUILD_PLAN §3.1)。

        kwargs はそのまま family.load(mode, **kwargs) に転送する(タスク1/タスク2:
        リクエストごとのモデル切替パラメータ、例 model="2512" 等)。
        """
        with self._registry_lock:
            family = self._require(name)
            for other_name in self._families_exclusive_with(name):
                other = self._families.get(other_name)
                if other is not None and other.is_loaded():
                    print(
                        f"[core.registry] switching family {name!r}: auto-unloading "
                        f"exclusive family {other_name!r}"
                    )
                    other.unload()
                    gpu.empty_cache()
                    gpu.reset_peak_stats()
            family.load(mode, **kwargs)

    def generate(self, name: str, request: dict) -> dict:
        """全ファミリー共通の生成ロック(core.gpu.generation_lock)を取ってから
        指定ファミリーの generate() を呼ぶ(GPU 同時1件、REBUILD_PLAN §3.1)。
        """
        family = self._require(name)
        with gpu.generation_scope():
            return family.generate(request)

    def get(self, name: str) -> ModelFamily:
        """登録済みファミリーを名前で取得する(app.py 側が load() 後に generate() を
        直接呼びたい場合の公開アクセサ。409即時応答のため registry.generate() の
        ブロッキングロックを経由できない呼び出し元向け)。
        """
        return self._require(name)

    def unload(self, name: "str | None" = None) -> dict:
        """指定ファミリー(None なら登録済み全部)を明示 unload する。"""
        with self._registry_lock:
            if name is not None:
                family = self._require(name)
                result = family.unload()
                gpu.empty_cache()
                gpu.reset_peak_stats()
                return {name: result}
            results = {}
            for fam_name, family in self._families.items():
                results[fam_name] = family.unload()
            gpu.empty_cache()
            gpu.reset_peak_stats()
            return results

    def status_all(self) -> dict:
        with self._registry_lock:
            names = list(self._families.keys())
        status = {name: self._families[name].status() for name in names}
        status["gpu_busy"] = gpu.generation_lock.locked()
        status["vram"] = gpu.vram_snapshot()
        return status

    def _require(self, name: str) -> ModelFamily:
        family = self._families.get(name)
        if family is None:
            raise KeyError(f"unknown model family: {name!r} (registered: {list(self._families)})")
        return family


# プロセス内シングルトン(app.py / families/* から import して使う唯一の窓口)。
registry = FamilyRegistry()
