# -*- coding: utf-8 -*-
"""
Phase 0 スモークテスト。モデルロード・pip install・HF ダウンロードは一切発生させない。

実行方法:
    venv/bin/python tests/smoke_phase0.py

検証項目(REBUILD_PLAN §4 Phase 0 の指示どおり):
  1. core/ 全モジュールの import 成功
  2. config の新旧環境変数の解決(os.environ を一時設定して確認)
  3. resolve のパス解決(実在の /home/animede/ComfyUI/models 配下で確認、HF DL は発生させない)
  4. registry の登録・排他・状態管理(ダミーファミリー2つで、片方 load 時にもう片方の
     unload フックが呼ばれること)
  5. gpu.py の VRAM 計測とロックの基本動作(torch.cuda.is_available() 確認まで。モデルロードなし)
"""
import os
import sys
import traceback

# このファイルは tests/ 直下にあるので、リポジトリルートを sys.path に通す。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_FAILURES = []


def check(label, fn):
    print(f"--- {label} ---")
    try:
        fn()
        print(f"[OK] {label}")
    except Exception:  # noqa: BLE001
        print(f"[FAIL] {label}")
        traceback.print_exc()
        _FAILURES.append(label)
    print()


# ============================================================================
# 1. import 成功
# ============================================================================
def test_imports():
    from core import config, resolve, loaders, optimize, gpu, registry  # noqa: F401

    print("core.config, core.resolve, core.loaders, core.optimize, core.gpu, core.registry の import OK")


# ============================================================================
# 2. config の新旧環境変数解決
# ============================================================================
def test_config_env_resolution():
    # 事前状態を保存して最後に復元する(他のテストや呼び出し元プロセスに影響しないため)。
    keys = ["DS_OFFLOAD", "QWENIMG_OFFLOAD", "DS_ATTN", "FLUX2_ATTN", "DS_COMPILE", "QWENIMG_COMPILE"]
    saved = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)

        # reload しないと _LEGACY_ALIASES 等は re-import されないが、env_str/env_bool は
        # 呼び出し毎に os.environ を読むので reload 不要。
        from core import config

        # 新名が無い場合、旧名(QWENIMG_OFFLOAD)にフォールバックすること。
        os.environ["QWENIMG_OFFLOAD"] = "group_lowvram"
        assert config.get_offload_mode_raw() == "group_lowvram", config.get_offload_mode_raw()

        # 新名(DS_OFFLOAD)があれば旧名より優先すること。
        os.environ["DS_OFFLOAD"] = "none"
        assert config.get_offload_mode_raw() == "none", config.get_offload_mode_raw()

        # FLUX2_ATTN からのフォールバックも確認(DS_ATTN は複数の旧名を持つ)。
        os.environ.pop("DS_OFFLOAD", None)
        os.environ["FLUX2_ATTN"] = "xformers"
        assert config.get_attention_backend() == "xformers", config.get_attention_backend()

        # env_bool: "1"/"true"/"yes"/"on" を True とみなす。
        os.environ["QWENIMG_COMPILE"] = "true"
        assert config.get_compile_enabled() is True

        os.environ["QWENIMG_COMPILE"] = "0"
        assert config.get_compile_enabled() is False

        # デフォルト値(未設定時)の確認。
        os.environ.pop("QWENIMG_COMPILE", None)
        os.environ.pop("DS_COMPILE", None)
        assert config.get_compile_enabled(default=False) is False

        print("新名優先・旧名フォールバック・env_bool 判定 いずれも期待どおり")
    finally:
        for k in keys:
            if saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved[k]


def test_expandable_segments_default():
    from core import config  # noqa: F401  (import 時に _ensure_expandable_segments が既に実行済み)

    value = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    assert "expandable_segments:True" in value, value
    print(f"PYTORCH_CUDA_ALLOC_CONF={value!r}")


# ============================================================================
# 3. resolve のパス解決
# ============================================================================
def test_resolve_local_path_hit():
    from core.resolve import resolve_comfyui_path, resolve_model_path

    comfyui_diffusion_models = resolve_comfyui_path("diffusion_models")
    assert os.path.isdir(comfyui_diffusion_models), comfyui_diffusion_models

    existing_files = [
        f for f in os.listdir(comfyui_diffusion_models)
        if os.path.isfile(os.path.join(comfyui_diffusion_models, f))
    ]
    assert existing_files, f"no files found under {comfyui_diffusion_models}"

    local_path = os.path.join(comfyui_diffusion_models, existing_files[0])
    # ローカルに実在するパスを渡した場合、HF Hub には一切アクセスせずそのまま返ること。
    resolved = resolve_model_path(local_path, repo_id="dummy/should-not-be-fetched", repo_filename="dummy.safetensors")
    assert resolved == local_path, resolved
    print(f"local hit: {local_path}")


def test_resolve_comfyui_models_dir():
    from core.config import COMFYUI_MODELS_DIR

    assert COMFYUI_MODELS_DIR == os.path.expanduser("~/ComfyUI/models"), COMFYUI_MODELS_DIR
    assert os.path.isdir(COMFYUI_MODELS_DIR), COMFYUI_MODELS_DIR
    print(f"COMFYUI_MODELS_DIR={COMFYUI_MODELS_DIR}")


# ============================================================================
# 4. registry の登録・排他・状態管理
# ============================================================================
def test_registry_exclusive_unload():
    from core.registry import FamilyRegistry, ModelFamily

    class DummyFamily(ModelFamily):
        def __init__(self, name):
            self.name = name
            self._loaded = False
            self.unload_calls = 0

        def load(self, mode):
            self._loaded = True

        def generate(self, request):
            return {"family": self.name, "mode_echo": request.get("mode")}

        def unload(self):
            self._loaded = False
            self.unload_calls += 1
            return {"freed": [self.name]}

        def status(self):
            return {"loaded": self._loaded}

    reg = FamilyRegistry()
    fam_a = DummyFamily("dummy_a")
    fam_b = DummyFamily("dummy_b")
    reg.register(fam_a, exclusive_with=["dummy_b"])
    reg.register(fam_b, exclusive_with=["dummy_a"])

    # 両方ロードされていない状態から dummy_a をロード -> dummy_b の unload は呼ばれないこと。
    reg.load("dummy_a", mode="t2i")
    assert fam_a.is_loaded() is True
    assert fam_b.unload_calls == 0

    # dummy_b をロード -> 排他グループの dummy_a が自動 unload されること。
    reg.load("dummy_b", mode="t2i")
    assert fam_b.is_loaded() is True
    assert fam_a.is_loaded() is False, "dummy_a should have been auto-unloaded on family switch"
    assert fam_a.unload_calls == 1, fam_a.unload_calls

    # status_all がファミリー名をキーに loaded 状態をまとめて返すこと。
    status = reg.status_all()
    assert status["dummy_a"]["loaded"] is False
    assert status["dummy_b"]["loaded"] is True
    assert "gpu_busy" in status and "vram" in status

    # generate() が generation_lock を経由して呼ばれ、結果を返すこと。
    result = reg.generate("dummy_b", {"mode": "edit"})
    assert result == {"family": "dummy_b", "mode_echo": "edit"}, result

    # 明示 unload(/api/unload 相当)。
    unload_result = reg.unload("dummy_b")
    assert fam_b.is_loaded() is False
    assert unload_result == {"dummy_b": {"freed": ["dummy_b"]}}, unload_result

    # 未登録ファミリー名は KeyError になること。
    try:
        reg.load("does_not_exist", mode="t2i")
        raise AssertionError("expected KeyError for unknown family")
    except KeyError:
        pass

    print("registry: 登録・排他自動unload・generate・明示unload・未登録エラー いずれも期待どおり")


# ============================================================================
# 5. gpu.py の VRAM 計測とロックの基本動作
# ============================================================================
def test_gpu_basics():
    from core import gpu

    cuda_available = gpu.is_cuda_available()
    print(f"torch.cuda.is_available() = {cuda_available}")

    free_gb = gpu.free_vram_gb()
    print(f"free_vram_gb() = {free_gb:.2f} GB")
    if not cuda_available:
        assert free_gb == 0.0

    snapshot = gpu.vram_snapshot()
    if cuda_available:
        assert snapshot is not None
        for key in ("allocated_gb", "max_allocated_gb", "reserved_gb", "free_gb", "total_gb"):
            assert key in snapshot, snapshot
        print(f"vram_snapshot() = {snapshot}")
    else:
        assert snapshot is None

    # ロックの基本動作(取得・解放。モデルロードは一切しない)。
    assert not gpu.generation_lock.locked()
    with gpu.generation_scope():
        assert gpu.generation_lock.locked()
    assert not gpu.generation_lock.locked()

    # empty_cache / reset_peak_stats が例外を出さず呼べること。
    gpu.empty_cache()
    gpu.reset_peak_stats()

    print("gpu: is_cuda_available/free_vram_gb/vram_snapshot/generation_scope/empty_cache OK")


def main():
    check("1. core/ 全モジュール import", test_imports)
    check("2a. config: 新旧環境変数の解決", test_config_env_resolution)
    check("2b. config: PYTORCH_CUDA_ALLOC_CONF 既定値", test_expandable_segments_default)
    check("3a. resolve: ComfyUI モデルディレクトリ", test_resolve_comfyui_models_dir)
    check("3b. resolve: ローカルパス解決(HF DL 未発生)", test_resolve_local_path_hit)
    check("4. registry: 登録・排他・状態管理", test_registry_exclusive_unload)
    check("5. gpu: VRAM計測・ロック基本動作", test_gpu_basics)

    print("=" * 60)
    if _FAILURES:
        print(f"FAILED: {len(_FAILURES)} check(s) failed: {_FAILURES}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
