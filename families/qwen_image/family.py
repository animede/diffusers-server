# -*- coding: utf-8 -*-
"""
QwenImageFamily: core.registry.ModelFamily の実装(Qwen-Image 系全モード)。

load(mode) / generate(request) / unload() / status() の共通インターフェースで
T2I / I2I / Edit / Edit(angles) / ControlNet(Union・Inpaint) / Layered をまとめる。

mode に対応するロード関数(core.registry.FamilyRegistry.load() 経由で呼ばれる):
  "t2i" / "i2i"        -> t2i.get_t2i_pipeline() / get_i2i_pipeline()
  "edit"               -> edit.get_edit_pipeline()
  "edit_angles"        -> edit_angles.get_edit_angles_pipeline()(charsheet専用、
                          fp8-lightning-angles。Multiple-angles LoRA fuse済み)
  "edit_angles_bf16"   -> edit_angles_bf16.get_edit_angles_bf16_pipeline()(charsheet専用、
                          bf16 transformer をロードして fuseせずベースのみ fp8_e4m3fn
                          ストレージ化し、その上で Lightning/Multiple-angles を adapter
                          として適用する「fp8-base-adapters」方式)
  "edit_angles_bf16group" -> edit_angles_bf16group.get_edit_angles_bf16group_pipeline()
                          (charsheet専用・新既定、CLAUDE.md 33番。旧
                          ~/charsheet 方式の完全再現: bf16 transformer を
                          fuse・fp8化せずロードし、Lightning/Multiple-angles を adapter
                          として適用した上で block-level group offloading で
                          GPU/CPU入れ替えする「bf16-group」方式)
  "controlnet"         -> controlnet.get_controlnet_pipeline()
  "inpaint"            -> controlnet.get_controlnet_inpaint_pipeline()
  "layered"            -> layered.get_layered_pipeline()

generate(request) の request は {"mode": <str>, ...パラメータ, "_images": [...]} 形式
(app.py 側であらかじめ画像を読み込み・前処理してから渡す。families 側は画像デコード等の
HTTP固有処理を持たない)。

edit / edit_angles / edit_angles_bf16 / edit_angles_bf16group の相互排他(タスク指示:
「同時常駐しない。いずれかロード時に他は解放」): 通常の Edit(edit_group)・charsheet 専用の
Edit-angles(edit_angles_group、fp8-lightning-angles fuse)・charsheet 専用の
Edit-angles-bf16(edit_angles_bf16_group、fp8-base-adapters)・charsheet 専用の
Edit-angles-bf16group(edit_angles_bf16group_group、bf16-group、CLAUDE.md 33番、新既定)は、
いずれもロード過程で bf16 transformer(40GB弱)を一時的に必要とし(fp8-base-adaptersも
fp8化前は一旦bf16でロードする。bf16-groupはCPUへロードするためVRAM上の一時ピークは
発生しないが、他グループのVRAM常駐と共存させる理由が無いため同じ排他に含める)、
48GB専有では2つ以上を同時に確保できない。そのため load(mode) 前に他の3グループの
いずれかが常駐していれば先に解放する(t2i/controlnet/layered グループには影響しない。
qwen_image 内の「グループ切替」パターンとして core.registry.FamilyRegistry の
ファミリー横断排他とは別に、このファミリー内部で同様のロジックを実装する)。

重要(実機検証で判明したOOM対策): 旧グループを解放するだけでは不十分で、共有
コンポーネント(vae/text_encoder/tokenizer、~15.7GB)も一旦解放する必要がある。
fp8-lightning系のEdit変種はロード時に bf16 transformer(約38GB)を直接cuda:0へ
ロードしてからLoRAをfuseする(fuse完了までは一時的に38-39GBのVRAMを要する)。
共有コンポーネントが常駐したままだと(15.7GB + 38GB超)が48GB専有環境の空きVRAMを
超えてOOMする(実機確認済み)。そのため lifecycle.unload_shared_if_unused() も
併せて呼ぶ(新グループの読み込み時に共有コンポーネントは再ロードされる。実測 ~25秒)。
"""
from core.registry import ModelFamily

from families.qwen_image import controlnet as controlnet_mod
from families.qwen_image import edit as edit_mod
from families.qwen_image import edit_angles as edit_angles_mod
from families.qwen_image import edit_angles_bf16 as edit_angles_bf16_mod
from families.qwen_image import edit_angles_bf16group as edit_angles_bf16group_mod
from families.qwen_image import generate as generate_mod
from families.qwen_image import layered as layered_mod
from families.qwen_image import lifecycle
from families.qwen_image import state
from families.qwen_image import t2i as t2i_mod
from families.qwen_image.paths import is_2512_model
from families.qwen_image.runtime import is_fp8_lightning_quant


class QwenImageFamily(ModelFamily):
    name = "qwen_image"

    # ------------------------------------------------------------------
    # ファミリー内グループ間の排他解放(2026-07-18追加、charsheet OOM バグ修正)
    #
    # 背景: fp8-lightning 系のロードは「bf16 transformer(約38GB)を直接cuda:0へ
    # ロード → LoRA fuse → fp8 layerwise cast」のため、fuse完了までの一時ピークだけで
    # 38GB超を要する。48GB専有では、別グループ(例: T2Iのfp8済みtransformer ~19GB +
    # 共有コンポーネント ~15.7GB)が常駐したままだと合算でOOMする(実機再現:
    # T2I(2512)生成後に charsheet(edit_angles、bf16 fuse)を呼ぶとOOM)。
    # 従来の相互排他は edit ⇔ edit_angles にしかなかったため、
    # T2I ⇔ edit/edit_angles の双方向にも同じ枠組みを拡張する。
    # 解放順序は lifecycle.unload() に委譲(ControlNet → T2I の順。CLAUDE.md 18番)。
    # 注意: layered も fuse 経路のため同種のリスクを持つが、layered 自体に既知の
    # OOM問題が残っている(CLAUDE.md 29番)ため別課題として扱い、ここでは対象外
    # (残存リスクとして CLAUDE.md に記載)。
    # ------------------------------------------------------------------

    @staticmethod
    def _t2i_load_will_fuse() -> bool:
        """これから行う T2I ロードが bf16 fuse(一時ピーク~38GB)を伴うか。
        2512 は常に fuse、無印は DS_QUANT=fp8-lightning のとき fuse
        (t2i.py の _load_t2i_group_locked() の分岐と同じ判定)。"""
        cfg = state.get_runtime_config()
        return is_2512_model(cfg.t2i_model) or is_fp8_lightning_quant(cfg.quant)

    # Edit系グループ名 <-> state のグループ辞書の対応(排他解放の共通処理で使う)。
    _EDIT_VARIANT_GROUPS = ("edit", "edit_angles", "edit_angles_bf16", "edit_angles_bf16group")

    @staticmethod
    def _edit_variant_group(target: str) -> dict:
        return {
            "edit": state.edit_group,
            "edit_angles": state.edit_angles_group,
            "edit_angles_bf16": state.edit_angles_bf16_group,
            "edit_angles_bf16group": state.edit_angles_bf16group_group,
        }[target]

    def _unload_other_edit_variants(self, keep: str) -> None:
        """keep 以外の Edit系グループ(edit / edit_angles / edit_angles_bf16 /
        edit_angles_bf16group)が常駐していれば解放する(2026-07-19、
        edit_angles_bf16group 追加に伴い4方向の相互排他へ拡張。いずれも 40GB弱の
        transformer を必要とし、48GB専有では2つ以上同時に確保できない)。
        """
        freed = False
        for target in self._EDIT_VARIANT_GROUPS:
            if target == keep:
                continue
            if self._edit_variant_group(target)["loaded"]:
                print(
                    f"[families.qwen_image] switching Edit variant: auto-unloading "
                    f"{target} group before loading {keep}"
                )
                lifecycle.unload(target)
                freed = True
        if freed:
            lifecycle.unload_shared_if_unused()

    def _unload_edit_groups_for_t2i_fuse(self) -> None:
        """T2Iグループのロード(fuse経路)に先立ち、Edit系グループ+共有を解放する。"""
        if state.t2i_group["loaded"]:
            return  # ロード済みなら fuse は走らない(解放不要)
        if not self._t2i_load_will_fuse():
            return
        freed = False
        for target in self._EDIT_VARIANT_GROUPS:
            if self._edit_variant_group(target)["loaded"]:
                print(
                    f"[families.qwen_image] auto-unloading {target} group before T2I fuse load "
                    "(48GB: bf16 fuse一時ピーク~38GBと他グループは共存不可)"
                )
                lifecycle.unload(target)
                freed = True
        if freed:
            lifecycle.unload_shared_if_unused()

    @staticmethod
    def _unload_t2i_group_for_edit_fuse(edit_variant_loaded: bool) -> None:
        """Edit系グループのロード(fuse経路)に先立ち、T2I/ControlNetグループ+共有を
        解放する(charsheet OOM 修正の本体)。edit_variant_loaded はこれからロードする
        Edit変種グループが既にロード済みかどうか(ロード済みなら fuse は走らず解放不要)。
        呼び出し側は「fuseが走る場合のみ」この関数を呼ぶ規約(edit は
        DS_QUANT=fp8-lightning 時のみ、edit_angles は常に fuse 経路)。
        """
        if edit_variant_loaded:
            return
        if (
            state.t2i_group["loaded"]
            or state.controlnet_union_group["loaded"]
            or state.controlnet_inpaint_group["loaded"]
        ):
            print(
                "[families.qwen_image] auto-unloading t2i (+controlnet) group before Edit fuse load "
                "(48GB: bf16 fuse一時ピーク~38GBとT2Iグループ(~19GB)+共有(~15.7GB)は共存不可)"
            )
            lifecycle.unload("t2i")  # ControlNet -> T2I の順で解放される(依存参照の都合、18番)
            lifecycle.unload_shared_if_unused()

    def load(self, mode: str, model: "str | None" = None) -> None:
        """model: T2I/I2Iの場合のみ有効("2512"(既定) | "qwen-image"。旧値 "2512-4bit" は
        後方互換で "2512" 扱い、タスク1)。他モードでは無視する。
        """
        if mode in ("t2i", "i2i"):
            # モデル切替(2512 ⇔ qwen-image)を先に確定させてから、fuse が走るかどうかを
            # 判定する(切替時は _switch_t2i_model_if_needed() が t2i_group を解放するため、
            # その後の t2i_group["loaded"] チェックが正しく「これからfuseが走る」を表す)。
            # get_*_pipeline() 内でも同じ切替が呼ばれるが冪等(同値なら即return)。
            t2i_mod._switch_t2i_model_if_needed(model)
            self._unload_edit_groups_for_t2i_fuse()
            if mode == "t2i":
                t2i_mod.get_t2i_pipeline(model)
            else:
                t2i_mod.get_i2i_pipeline(model)
        elif mode == "edit":
            self._unload_other_edit_variants(keep="edit")
            if is_fp8_lightning_quant(state.get_runtime_config().quant):
                self._unload_t2i_group_for_edit_fuse(state.edit_group["loaded"])
            edit_mod.get_edit_pipeline()
        elif mode == "edit_angles":
            self._unload_other_edit_variants(keep="edit_angles")
            # edit_angles(charsheet専用)は量子化設定に関わらず常に bf16 fuse 経路。
            self._unload_t2i_group_for_edit_fuse(state.edit_angles_group["loaded"])
            edit_angles_mod.get_edit_angles_pipeline()
        elif mode == "edit_angles_bf16":
            self._unload_other_edit_variants(keep="edit_angles_bf16")
            # fp8-base-adapters変種(charsheet専用)は fuse を行わないが、ロード過程で
            # bf16 transformer(~38GB)を一時的に確保してから fp8 ストレージ化するため、
            # このロード中のピークが T2I/ControlNet(~19GB)+共有(~15.7GB)と共存できない
            # 点は fuse 系と同じ。そのため同じ解放処理を再利用する。
            self._unload_t2i_group_for_edit_fuse(state.edit_angles_bf16_group["loaded"])
            edit_angles_bf16_mod.get_edit_angles_bf16_pipeline()
        elif mode == "edit_angles_bf16group":
            self._unload_other_edit_variants(keep="edit_angles_bf16group")
            # bf16-group変種(charsheet専用・新既定、CLAUDE.md 33番)は fuse も fp8化も
            # 行わず、transformer は CPU へ streaming ロードしてから group offload するため
            # VRAM上の一時ピークは他の fuse 系ほど大きくないが、T2I/ControlNet(~19GB)+
            # 共有(~15.7GB)が常駐したまま transformer を group offload で GPU へ載せ始めると
            # 共存できない点は同じため、同じ解放処理を再利用する(fuse系と扱いを揃える)。
            self._unload_t2i_group_for_edit_fuse(state.edit_angles_bf16group_group["loaded"])
            edit_angles_bf16group_mod.get_edit_angles_bf16group_pipeline()
        elif mode == "controlnet":
            # ControlNet は無印ベース(t2i.py 参照)。切替後に t2i の fuse ロードが
            # 走る場合は Edit系を先に解放する(T2I/I2I と同じ理由)。
            t2i_mod._switch_t2i_model_if_needed("qwen-image")
            self._unload_edit_groups_for_t2i_fuse()
            controlnet_mod.get_controlnet_pipeline()
        elif mode == "inpaint":
            t2i_mod._switch_t2i_model_if_needed("qwen-image")
            self._unload_edit_groups_for_t2i_fuse()
            controlnet_mod.get_controlnet_inpaint_pipeline()
        elif mode == "layered":
            layered_mod.get_layered_pipeline()
        else:
            raise ValueError(f"unknown qwen_image mode: {mode!r}")

    def generate(self, request: dict) -> dict:
        mode = request["mode"]
        if mode == "t2i":
            return generate_mod.run_t2i(request)
        if mode == "i2i":
            return generate_mod.run_i2i(request, request["_image"])
        if mode == "edit":
            return generate_mod.run_edit(request, request["_images"])
        if mode == "edit_angles":
            return generate_mod.run_edit_angles(request, request["_images"])
        if mode == "edit_angles_bf16":
            return generate_mod.run_edit_angles_bf16(request, request["_images"])
        if mode == "edit_angles_bf16group":
            return generate_mod.run_edit_angles_bf16group(request, request["_images"])
        if mode == "controlnet":
            return generate_mod.run_controlnet(request, request["_control_image"])
        if mode == "inpaint":
            return generate_mod.run_inpaint(request, request["_image"], request["_mask_image"])
        if mode == "layered":
            return generate_mod.run_layered(request, request["_image"])
        raise ValueError(f"unknown qwen_image mode: {mode!r}")

    def unload(self) -> dict:
        return lifecycle.unload("all")

    def unload_target(self, target: str) -> dict:
        """/api/unload の target(t2i/edit/controlnet/layered/all)による部分解放
        (旧実装互換。ModelFamily の抽象 unload() は "全解放" 用途なので、部分解放は
        このメソッドを app.py から個別に呼ぶ)。
        """
        return lifecycle.unload(target)

    def status(self) -> dict:
        return lifecycle.get_status()

    def is_loaded(self) -> bool:
        return lifecycle.is_any_loaded()
