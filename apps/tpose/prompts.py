# -*- coding: utf-8 -*-
"""Tポーズ4ビューのプロンプト定義(image-3d / rig-service 向け)。

charsheet(apps/charsheet/prompts.py)との違い(2026-07-26の実機検証で確定):

  - **Multiple-angles LoRA を使わない**。Tポーズ用途では通常 Edit(fp8-lightning)の方が
    明確に優位だった: angles LoRA 経路(edit_angles_bf16group)は背面で頭部が黒髪へ
    変質する同一性ドリフトが出て、かつ4倍遅い(42〜46秒 vs 5〜11秒/枚)。
    そのため本アプリの文言は「LoRAトリガー文の一字一句維持」の制約を受けず、
    Tポーズ保持を明示する自由な文言にしてよい(charsheet の VIEWS_LORA が
    "neutral standing A-pose" と書いているのはそのままでは有害)。
  - **真横(90度)ビューを持たない**。Tポーズの真横投影は「手前の腕がカメラを向いて
    完全に短縮し胴体を隠す」構図で、通常Edit / angles LoRA / LoRAトリガー文 / JoyAI /
    2枚参照 / 幾何を明示した指示の6通りすべてで破綻した(手前腕が胸の上の肉塊になり、
    両脚が1本の柱に融合する)。代わりに 3/4(45度)ビュー2枚を既定に含める。
    Hunyuan3D-2mv(image-3d)のビュータグは front/left/back/right 限定のため、
    45度画像を left/right スロットへ入れてはいけない(image-3d 側では front+back のみを
    マルチビュー入力に使い、45度は参考出力として扱う)。

生成順序: front を必ず最初に生成し、back / 45度2枚は **生成した front 画像を入力**に
連鎖生成する(元画像から直接背面化すると綺麗でも帽子・尻尾の造形が前面と食い違う)。

ビュー別に文言を変えている理由(いずれも実機で出た不具合の修正):
  - 背面ビューで手のひらの文言を正面と同じにすると、**背面画像に肉球が描かれる**
    (手のひらを正面へ向けたTポーズなら背面からは手の甲しか見えないはずで、
    3D再構成用のビューとして矛盾する)。背面は手の甲側の文言を使う。
  - 背面ビューの視点句に "head" を使うと後頭部が黒髪の人型へ変質しやすい
    (下記 _palms_clause 手前のコメント参照)。"fur" 基準の語彙にする。

「脚が伸びる問題」(ユーザー報告 -> 実測で対処、2026-07-26):
  ぬいぐるみ(短い脚・大きい頭)の入力に対し、出力の脚が長くなりハーフパンツが
  長ズボンになる劣化が出た。定量指標として **arm_rel = 肩ライン(最大幅の行)の高さ /
  全高** を使うと切り分けられる(値が小さい = 肩より下が伸びている)。
  同一seed=42・momo.png での実測:
    - 初期プローブ(素のTポーズ指示・肉球色なし・しっぽなし)        arm_rel 0.414 ← 良好
    - 「維持」表現 + 詳細な肉球記述 + しっぽ記述(問題が出た版)      arm_rel 0.366 ← 劣化
    - 上記から しっぽ記述だけ除去                                  arm_rel 0.392
    - 上記から 肉球記述も短縮 + 「Change the pose」表現へ           arm_rel 0.397
    - さらに body="short stubby legs and a large head" を追加       arm_rel 0.450 ← 最良
    - body + しっぽ記述(短い形)の併用(現在の既定構成)            arm_rel 0.428 ← 採用
  効かなかった対処(記録として):
    - "keep exactly the same body proportions, limb lengths, head-to-body ratio ..."
      のような**汎用的な**体型維持文の追加(arm_rel 0.364、ほぼ無効)。
    - 正方形キャンバスを 1024x768 の横長にする(「縦の余白を埋めようとして脚が伸びる」
      という仮説の検証。arm_rel 0.364 で**仮説は否定**された)。
  結論: (1) プロンプトを盛るほど体型が伸びる(しっぽ記述 -0.026、詳細な肉球記述
  -0.017 程度)ので既定の文言は短く保つ。(2) 汎用文では効かず、**体型を具体的に
  言語化する(`body` パラメータ)のが唯一効いた対処**。しっぽと同じ「言語化すれば従う」
  性質で、入力画像から推定できる情報でもモデルは勝手にドリフトするため明示が有効。
"""

NEGATIVE_PROMPT = "低分辨率，低画质，肢体畸形，手指畸形"

# Tポーズ本体の指示。
#
# 正面(1段目)は元画像のポーズを**変更する**ので "Change the pose to a T-pose:" と
# 明示的に命令する。2段目以降(背面・45度)は既にTポーズの画像を参照するので
# 「同じTポーズを維持」と書く。この区別は実測で効く: 正面で「維持」表現を使うと
# 体型が伸びやすかった(下の「脚が伸びる問題」のコメント参照)。
_T_POSE_ARMS = (
    "both arms extended straight out horizontally to the left and right at "
    "shoulder height, elbows straight, legs straight and slightly apart"
)
_T_POSE_CLAUSE_FRONT = _T_POSE_ARMS
_T_POSE_CLAUSE_OTHER = f"keeping exactly the same T-pose with {_T_POSE_ARMS}"

_KEEP_CLAUSE = (
    "full body visible from head to toe, plain white background, "
    "keep the character design, costume and colors exactly the same"
)

# 各エントリ: key, label_ja, label_en, view(視点句), for_3d(2mvのビュースロットに使えるか)
VIEWS = [
    {
        "key": "front",
        "label_ja": "正面",
        "label_en": "Front",
        # 1段目(元画像からポーズを変える)では前に "Change the pose to a T-pose: " が
        # 付く(_view_clause 参照)。実測でこの命令形の方が体型が保たれる。
        "view": (
            "the character stands upright facing directly toward the camera"
        ),
        "for_3d": True,
    },
    {
        "key": "back",
        "label_ja": "背面",
        "label_en": "Back",
        # 背面の視点句は動物型(fur)/人物型(hair)で語彙を切替える(_back_view() 参照)。
        # ここは既定(動物型)の文言。
        "view": (
            "Show the character from behind in a full body back view, rear facing "
            "the camera, showing all back details of the fur, costume and "
            "accessories from behind"
        ),
        "for_3d": True,
    },
    {
        "key": "front_left_45",
        "label_ja": "左前45度",
        "label_en": "Front-Left 45°",
        "view": (
            "Show the character from a 3/4 front-left angle, showing both front and "
            "left side details"
        ),
        "for_3d": False,
    },
    {
        "key": "front_right_45",
        "label_ja": "右前45度",
        "label_en": "Front-Right 45°",
        "view": (
            "Show the character from a 3/4 front-right angle, showing both front and "
            "right side details"
        ),
        "for_3d": False,
    },
]

VIEW_BY_KEY = {v["key"]: v for v in VIEWS}

# 手のひら(palms): リグ用Tポーズの標準は「手のひらを正面へ」。
# 実機検証済み: 肉球の色・形状まで指示どおり描かれる。
PALMS_MODES = ("forward", "natural")

# 被写体タイプ(subject)。**語彙の切替はこのパラメータだけで行う**。
#
# 経緯(ユーザー報告 -> 修正、2026-07-26): 当初は paw_pads を「動物か人間か」の判定に
# 流用しており(paw_pads="none" なら人物、それ以外は動物)、既定 paw_pads="auto" が
# 動物語彙("fur" / "paws" / "paw pads")を無条件に注入していた。その結果
# **リアルな人形の入力を既定のまま流すと、背面が動物化し手に肉球が付く**という
# 報告が出た。被写体タイプを独立パラメータに分離し、既定 "auto" は
# **中立語彙(fur/hair/paw/pad のどれも書かない)**にした。
#   - "auto"  (既定): 中立。参照画像に判断を委ねる
#   - "animal": 毛皮・肉球のある動物/ぬいぐるみ("fur" / "paws" / "paw pads")
#   - "human" : 人物・リアルな人形("hair" / "hands" / "fingers")
SUBJECT_MODES = ("auto", "animal", "human")

_PALMS_FORWARD_ANIMAL = (
    "both front paws open with the palms rotated to face the camera so that the "
    "paw pads are clearly visible"
)
_PALMS_FORWARD_HUMAN = (
    "both hands open with the palms facing the camera, fingers spread"
)
# 中立(subject="auto"): 手が paw なのか hand なのかを書かない。動物語彙を書くと
# リアルな人形に肉球が付き、人間語彙を書くとぬいぐるみの前足が人間の手になるため。
_PALMS_FORWARD_NEUTRAL = (
    "both palms open and rotated to face the camera"
)

# 背面ビュー用: 手のひらは(正面へ向けたまま)カメラの反対を向くので、見えるのは手の甲。
# ここを正面と同じ「肉球が見える」文言にすると、背面画像に肉球が描かれてしまい
# 3D再構成用のビューとして矛盾する(実機で発生・修正済み)。
# 「爪」は書かない: ここに "with their claws" と書いていたために背面で黒い爪が
# 目立つ出力になっていた(ユーザー報告 -> 削除、2026-07-27)。
_PALMS_BACK_ANIMAL = (
    "the palms still face away from the camera so we see the backs of both paws, "
    "the paw pads are not visible from behind"
)
_PALMS_BACK_HUMAN = (
    "the palms still face away from the camera so we see the backs of both hands"
)
_PALMS_BACK_NEUTRAL = (
    "the palms still face away from the camera, so the palms are not visible "
    "from behind"
)

# 背面ビューの同一性ドリフト(後頭部が黒髪の人型へ変質する)について、実機で確定した
# 対処と失敗例(この順で試して最後のものだけが効いた):
#   × "do not turn the head into human hair" のような否定文で抑止する
#     -> **悪化した**。拡散モデルは否定を扱えず、文中の "human hair" が誘引になる。
#   × "the whole head stays covered with the same fur..." の肯定形の補強文を足す
#     -> 効果なし(黒い髪型のまま)。
#   ○ **視点句から "head" という語を外し "fur" にする**
#     ("showing all back details of the fur, costume and accessories from behind")
#     + 補強文を足さない -> 白い毛のまま・耳の黒斑も正しい位置に描かれた。
#   ○ 併せて元画像も2枚目の参照として渡す(apps/tpose/jobs.py の refs 組み立て)。
# 教訓: 後頭部の描写は「head/hair」語彙に敏感で、補強文を足すより語彙を変える方が効く。


def resolve_subject(subject: str, paw_pads: str) -> str:
    """被写体タイプを "animal" / "human" / "neutral" のいずれかへ解決する。

    - subject="animal"/"human" が明示されていればそれに従う。
    - subject="auto"(既定)のときは paw_pads から推測する:
        * 色などの明示指定(例 "pink")-> 肉球を描く意図なので "animal"
        * "none"(肉球に言及しない)   -> "human"(旧仕様の後方互換)
        * "auto"/空                   -> "neutral"(動物/人間どちらの語彙も使わない)
    """
    subject = (subject or "auto").strip().lower()
    if subject in ("animal", "human"):
        return subject
    pads = (paw_pads or "auto").strip().lower()
    if pads == "none":
        return "human"
    if pads not in ("", "auto"):
        return "animal"
    return "neutral"


def _palms_clause(view_key: str, palms: str, paw_pads: str, subject_kind: str) -> str:
    """ビュー別・被写体タイプ別の手のひら指示句を組み立てる。

    paw_pads(subject_kind="animal" のときのみ肉球の記述に使う):
      - ""/"auto" : 参照画像の肉球の色をそのまま踏襲させる(色を指定しない)
      - "none"    : 肉球に言及しない
      - それ以外  : 色などの自由記述(例 "pink", "dark brown")を肉球指示に埋め込む
    """
    if palms != "forward":
        return ""
    pads = (paw_pads or "auto").strip().lower()
    if view_key == "back":
        return {
            "animal": _PALMS_BACK_ANIMAL,
            "human": _PALMS_BACK_HUMAN,
        }.get(subject_kind, _PALMS_BACK_NEUTRAL)
    if subject_kind == "human":
        return _PALMS_FORWARD_HUMAN
    if subject_kind == "neutral":
        return _PALMS_FORWARD_NEUTRAL
    # animal
    if pads in ("", "auto", "none"):
        return _PALMS_FORWARD_ANIMAL
    # 色・質感の自由記述。指球の数(「指球4+中央球1」)まで書くと肉球の描写自体は
    # 更に丁寧になるが、実測ではプロンプトを盛るほど体型が伸びる副作用があったため
    # (下の「脚が伸びる問題」のコメント参照)、色を差し込むだけの短い形にしてある。
    return (
        "both front paws open with the palms rotated to face the camera so that "
        f"the {paw_pads.strip()} paw pads are clearly visible"
    )


# 爪(claws)。subject が animal に解決されるときだけ使う。
#
# 実測(momo.png、2026-07-27):
#   ✕ "the paws have soft rounded tips without claws"(否定形)-> **爪が残る**
#     (前足の縁と足先に黒い爪が描かれたまま。CLAUDE.md 57番の「否定文は逆効果」と同じ)
#   ○ "the paw tips are soft, round and smooth"(肯定形のみ)-> 前足(手)の爪は消える
#     (体型への悪影響もなし: arm_rel 0.401 -> 0.406)
#   ただし上記だけでは **後足(足先)に茶色の爪がわずかに残る**(ユーザー報告。
#   足先の暗色ピクセル実測 1805px)。対処の追試:
#   ✕ "all four paws end in soft, round, smooth fur in the same color as the
#     surrounding fur"(まとめて言う)-> 悪化(足先 3173px、手にも 151px)
#   ○ **後足を名指しして重ねる**(採用): "..., the toes of the hind feet are also
#     soft, round and smooth in the same fur color" -> 足先 764px(-58%)、
#     目視でも茶色の爪が消える。手の 52px は輪郭のアンチエイリアス由来で爪は無い。
#   教訓: 部位を名指ししないとモデルは手だけに適用する(「all four paws」のような
#   総称よりも "hind feet" と具体的に書く方が効く)。
#   **さらに追試(「足の指も消えた」の報告 -> 2026-07-27)**: 上の文だけだと後足の指の
#   分離までのっぺりして「指が無い」見た目になる(指の内部勾配 1151px -> 965px)。
#   「指」と「爪なし」の両立を測る指標を追加した(足領域の 暗色px=爪 と 内部勾配px=指)。
#   実測(momo.png、目標: 爪<1200 かつ 指>1400):
#     ✕ "the hind feet keep their separate rounded toes in the same fur color as the body"
#       -> 指は戻る(2018px)が **爪も戻る**(3681px)
#     ✕ 上に "every toe tip is soft, round and smooth" を足す -> 爪 3700px(戻らない)
#     ✕ "...in cream white fur with no dark tips"(否定形)-> 爪 676px だが指 1024px
#     ○ **色を具体的に明示**(採用): "the hind feet have separate rounded toes and each toe
#       is <毛色> fur right to the tip" -> 爪 889px / 指 1401px。目視でも指が4本に分かれ、
#       黒い爪が消える
#   教訓: 「同じ毛色で」のような相対表現では効かず、**具体的な色名**が必要。色名は
#   入力画像から自動サンプリングする(sample_fur_color()、apps/tpose/generate.py)。
CLAWS_MODES = ("none", "auto")
# 毛色が分からない場合のフォールバック(爪は消えるが後足の指の分離も弱くなる)
_CLAWS_NONE = (
    "the paw tips are soft, round and smooth, the toes of the hind feet are also "
    "soft, round and smooth in the same fur color"
)


def _claws_none_with_color(fur_color: str) -> str:
    """毛色が分かるときの爪抑制文(実測で最良、上のコメント参照)。"""
    return (
        "the paw tips are soft, round and smooth, the hind feet have separate "
        f"rounded toes and each toe is {fur_color.strip()} fur right to the tip"
    )


def _claws_clause(claws: str, subject_kind: str, fur_color: str = "") -> str:
    """爪の指示句。animal 以外(人物・中立)では何も足さない(爪の語自体を出さない)。

    - "none"(既定): 爪を消す(ぬいぐるみでは爪が無い方が自然というユーザー判断)
    - "auto"       : 何も書かない(参照画像に爪があればそのまま出る)
    - それ以外     : 自由記述(例 "short white claws")をそのまま使う
    """
    if subject_kind != "animal":
        return ""
    value = (claws or "none").strip().lower()
    if value == "auto":
        return ""
    if value == "none":
        if fur_color and fur_color.strip():
            return _claws_none_with_color(fur_color)
        return _CLAWS_NONE
    return f"the paws have {claws.strip()}"


def _tail_clause(tail: str) -> str:
    """しっぽの指示句。

    しっぽ形状は正面画像からは推定できないため、未指定(auto)だとビューごとに
    別形状が創作される(実機で 白ポンポン / 黒混じりポンポン / なし / 長い黒尻尾 の
    4通りにばらけることを確認)。指定した場合は全ビューで同一文言を使い、
    正面ビューにも入れる(正面でもしっぽが描かれ、後続ビューの参照画像になるため)。
    """
    tail = (tail or "").strip()
    if not tail or tail.lower() == "auto":
        return ""
    if tail.lower() in ("none", "no", "nothing"):
        return "the character has no tail"
    # 短い形("with ...")にしている理由は下の「脚が伸びる問題」のコメント参照
    # (長い形 "the character has ..., clearly visible" は体型の伸びを助長した)。
    return f"with {tail}"


def _body_clause(body: str) -> str:
    """体型(頭身・脚の長さ)の指示句。

    「脚が伸びる問題」への主要な対処(実測値は下のコメント参照)。しっぽと同じく
    「言語化すればモデルは従う」性質で、`body="short stubby legs and a large head"`
    のように書くと元のぬいぐるみ体型が保たれる。空なら何も足さない。
    """
    body = (body or "").strip()
    if not body or body.lower() in ("auto", "none"):
        return ""
    return f"the character has {body}"


_BACK_VIEW_HUMAN = (
    "Show the character from behind in a full body back view, rear facing the "
    "camera, showing all back details of the hair, costume and accessories from "
    "behind"
)
# 中立(subject="auto"): "fur" も "hair" も書かない。"head" を書くと後頭部が黒髪化する
# 罠(上記コメント)があるため、頭部を指す語自体を避ける。
_BACK_VIEW_NEUTRAL = (
    "Show the character from behind in a full body back view, rear facing the "
    "camera, showing all back details of the costume and accessories from behind"
)


def _view_clause(view_key: str, subject_kind: str, first_stage: bool = False) -> str:
    """視点句。背面のみ被写体タイプで語彙を切替える。

    毛で覆われたキャラで "head"/"hair" 語彙を使うと後頭部が黒髪へ変質しやすい
    (上のコメント参照)ため animal では "fur"、human では "hair"、
    auto(neutral)では**どちらも書かない**("costume and accessories" のみ)。

    first_stage(元画像からポーズを変える段)では "Change the pose to a T-pose: " を
    前置する(「維持」表現だと体型が伸びる、下の実測値参照)。
    """
    if view_key == "back":
        text = {
            "animal": VIEW_BY_KEY["back"]["view"],
            "human": _BACK_VIEW_HUMAN,
        }.get(subject_kind, _BACK_VIEW_NEUTRAL)
    else:
        text = VIEW_BY_KEY[view_key]["view"]
    if first_stage:
        return "Change the pose to a T-pose: " + text[0].lower() + text[1:]
    return text


# 「生成画像の色が参照元より薄い」というユーザー指摘への調査結果(2026-07-27、**未採用**)。
#
# 実測は指摘を裏付けた: 参照元 momo.png(被写体の輝度中央値112 / 白に近い画素1.9% /
# 輪郭5px帯130)に対し、生成は 正面155/2.2%/178、**背面184/10.1%/201** と明るく、
# とくに背面が白飛びする。これが背景除去で淡い部位が消える一因だった。
#
# 対策として「色調保持の一文」を末尾に足す案を試したが **採用しなかった**:
#   - 背面単体では効果があった(白に近い画素 5.8% -> 1.6%、輪郭帯 196 -> 186)。
#   - しかし **爪抑制文が末尾から外れると爪が戻る**(正面の爪 889px -> 3493px)。
#     参照元に言及しない "natural soft studio lighting with gentle shading" でも
#     爪 3509px・明るさも改善せず(中央値 149 -> 162 で悪化)。
#   - つまり**末尾の句が最も強く効く**ため、爪抑制文の後ろには何も足せない。
# 結論: 明るさは**プロンプトではなく切り抜き側で対処する**(`_cutout_rgba()` の
# 白キーイング。背面の取りこぼしは 17,338px -> 774px まで解消済み)。
# なお `extra_prompt` も同じ理由で**爪抑制文より前**に差し込む(下記 build_prompt)。
def build_prompt(view_key: str, palms: str = "forward", paw_pads: str = "auto",
                 tail: str = "", body: str = "", extra: str = "",
                 first_stage: bool = False, subject: str = "auto",
                 claws: str = "none", fur_color: str = "") -> str:
    """1ビュー分のプロンプトを組み立てる。

    句の順序は実測で決めてある(下の「脚が伸びる問題」のコメント参照):
    視点 → Tポーズ → 手のひら → 同一性維持 → 体型 → しっぽ → 追記。
    体型・しっぽは「同一性維持」より後ろに置く(前に置くと本文が長くなるほど
    同一性維持の効きが薄まる傾向があったため)。

    first_stage: 元画像から生成する1段目(通常は front)かどうか。
      - True  : ポーズを**変更する**ので "Change the pose to a T-pose:" 命令形を使い、
                体型(body)の指定もここだけに入れる。
      - False : 既にTポーズの生成画像を参照するので「同じTポーズを維持」と書く。
                **体型句は入れない**: 参照画像が体型そのものを持っているため不要な上、
                ユーザーの体型記述に "head" が含まれると(例 "a large head")背面ビューで
                黒髪化のトリガーになることを実機で確認したため(下記コメント参照)。
    """
    subject_kind = resolve_subject(subject, paw_pads)
    t_pose = _T_POSE_CLAUSE_FRONT if first_stage else _T_POSE_CLAUSE_OTHER
    parts = [_view_clause(view_key, subject_kind, first_stage), t_pose]
    palms_clause = _palms_clause(view_key, palms, paw_pads, subject_kind)
    if palms_clause:
        parts.append(palms_clause)
    parts.append(_KEEP_CLAUSE)
    if first_stage:
        body_clause = _body_clause(body)
        if body_clause:
            parts.append(body_clause)
    tail_clause = _tail_clause(tail)
    if tail_clause:
        parts.append(tail_clause)
    if extra and extra.strip():
        parts.append(extra.strip())
    # 爪抑制文は**必ず末尾**に置く(末尾の句が最も強く効くため。上のコメント参照)
    claws_clause = _claws_clause(claws, subject_kind, fur_color)
    if claws_clause:
        parts.append(claws_clause)
    return ", ".join(parts)


# 体型プリセット(UI用。値はそのまま body パラメータへ渡せる自由記述)。
# 「脚が伸びる問題」への対処が発見しやすいようプリセット化してある(冒頭コメント参照)。
BODY_PRESETS = [
    {"key": "auto", "label_ja": "指定しない", "value": ""},
    {
        "key": "plush",
        "label_ja": "ぬいぐるみ体型(短い脚・大きい頭)",
        "value": "short stubby legs and a large head",
    },
    {
        "key": "chibi",
        "label_ja": "2頭身デフォルメ",
        "value": "a very large head and a small short body, two-heads-tall proportions",
    },
    {
        "key": "human",
        "label_ja": "人体標準プロポーション",
        "value": "normal human body proportions",
    },
]

# 生成後の色調整(recolor、2026-07-27追加)。
#
# 経緯: 「生成画像の色が参照元より薄い」問題は1パス目のプロンプトでは解決できなかった
# (上の「未採用」コメント: 末尾に句を足すと爪抑制が壊れる)。ユーザーが**2パス目の
# Edit で色を任意に変えられる**ことを発見したため、オプションとして取り込んだ。
# 2パス目は独立した Edit 呼び出しなので「末尾が最強」の制約と衝突しない。
#
# 実測(1パス目の正面を入力に、seed 42):
#   - "Make the fur a warmer cream tone with richer shading and slightly deeper
#     contrast" -> 輝度中央値 149 -> **95**(参照元99に接近)、輪郭帯 178 -> 142。
#     ただし**毛色が黄褐色へ寄り、耳の黒斑も変化**し、爪も戻った(889px -> 2448px)。
#   - "Increase the color saturation and contrast a little ..." -> ほぼ変化なし
#     (中央値 148)。
# つまり効果の強さは文言次第で、**強い指示は同一性も動かす**。既定は無効(空文字)にし、
# 文言はユーザーに委ねる(プリセットは用意しない: 望ましい色は用途ごとに違うため)。
_RECOLOR_KEEP = (
    "keep the pose, composition, design and background exactly the same, "
    "plain white background, full body visible from head to toe"
)


def build_edit_prompt(instruction: str, keep_pose: bool = True) -> str:
    """生成済みビューへ追加でかける Edit のプロンプトを組み立てる。

    色調整だけでなく汎用の修正指示に使える(「帽子を外す」「服を赤くする」等)。
    keep_pose=True(既定)なら「ポーズ・構図・デザイン・背景は変えない」を付け、
    Tポーズと白背景・画角を保つ。False なら指示文だけを渡す(構図ごと変えたい場合)。
    """
    instruction = instruction.strip()
    return f"{instruction}, {_RECOLOR_KEEP}" if keep_pose else instruction


def build_recolor_prompt(recolor: str) -> str:
    """生成時 `recolor` パラメータ用(= build_edit_prompt の別名)。"""
    return build_edit_prompt(recolor)


# しっぽプリセット(UI用。値はそのまま tail パラメータへ渡せる自由記述)
TAIL_PRESETS = [
    {"key": "auto", "label_ja": "指定しない(入力画像に任せる)", "value": "auto"},
    {"key": "none", "label_ja": "しっぽなし", "value": "none"},
    {"key": "pompom", "label_ja": "短い丸いポンポン", "value": "a short round pompom tail"},
    {"key": "fluffy_long", "label_ja": "長くふさふさ", "value": "a long fluffy tail hanging down"},
    {
        "key": "fluffy_long_black_tip",
        "label_ja": "長くふさふさ(先端が黒)",
        "value": "a long fluffy tail with a black tip, hanging down",
    },
    {"key": "thin", "label_ja": "細い長いしっぽ", "value": "a thin long tail"},
]
