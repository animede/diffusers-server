// Qwen-Image 評価UI フロントエンド(素のJS、外部ライブラリなし)

const gallery = []; // セッション内蓄積(生成結果のメタデータ+画像URL)

// --- タブ切替 ---
// [data-tab] 限定でバインドすること: サブタブボタン(LTX2等)も .tab-btn クラスを持つため、
// クラスだけで選択すると サブタブクリック時にメインパネルの active まで剥がしてしまう
// (パネル全体が消えるバグの実績あり)。active の付け外しも [data-tab] 同士に限定する。
document.querySelectorAll(".tab-btn[data-tab]").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn[data-tab]").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
  });
});

// --- 汎用ドロップゾーン化(全タブの input[type="file"] に一括適用) ---
// 目的: 個々のフォームを個別実装せず、DOM上の input[type="file"] を一括で
// ドラッグ&ドロップ対応にする。将来フォームが増えても initGenericDropzones() を
// 呼び直すだけで自動的に対応させたい設計のため、フォーム固有のロジックには
// 一切触れない(既存の FormData 構築・submit/change ハンドラは無変更のまま動く)。
//
// 実装方針:
// - 各 input を「ラッパー(.gd-dropzone-wrap)」で包み、input 自体は
//   position:absolute + opacity:0 で透明化するが、サイズはドロップゾーンいっぱいに
//   広げて残す(input 自身が dragover/drop/click を受け取れるため、ブラウザ標準の
//   ファイル選択ダイアログ挙動をそのまま利用できて実装が単純になる)。
// - ドロップされたファイルは DataTransfer 経由で input.files に代入し、
//   "change" イベントを dispatch する。これにより既存の change ハンドラ
//   (controlnet/inpaint/layered のプレビュー処理等)がそのまま動作する。
// - accept 属性に合わないファイルは拒否し、ドロップゾーン内に一時的にエラー表示する。
// - 既にドロップゾーン化されている input(data-gd-skip="1", 例: charsheetタブは
//   専用の #cs-dropzone 実装を持つため対象外)はスキップする。
const GD_DROPZONE_INIT_FLAG = "gdDropzoneInit";

function _gdFormatBytes(bytes) {
  if (bytes == null) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function _gdAcceptMatches(file, acceptAttr) {
  if (!acceptAttr) return true;
  const patterns = acceptAttr.split(",").map((s) => s.trim()).filter(Boolean);
  if (patterns.length === 0) return true;
  return patterns.some((pattern) => {
    if (pattern.startsWith(".")) {
      return file.name.toLowerCase().endsWith(pattern.toLowerCase());
    }
    if (pattern.endsWith("/*")) {
      const prefix = pattern.slice(0, -1); // "image/"
      return file.type.startsWith(prefix);
    }
    return file.type === pattern;
  });
}

function _gdRenderFileList(listEl, files) {
  listEl.innerHTML = "";
  Array.from(files).forEach((file) => {
    const item = document.createElement("div");
    item.className = "gd-file-item";

    if (file.type.startsWith("image/")) {
      const img = document.createElement("img");
      img.className = "gd-file-thumb";
      const reader = new FileReader();
      reader.onload = (e) => {
        img.src = e.target.result;
      };
      reader.readAsDataURL(file);
      item.appendChild(img);
    } else {
      const icon = document.createElement("div");
      icon.className = "gd-file-icon";
      icon.textContent = file.type.startsWith("audio/") ? "♪" : file.type.startsWith("video/") ? "▶" : "📄";
      item.appendChild(icon);
    }

    const name = document.createElement("div");
    name.className = "gd-file-name";
    name.textContent = file.name;
    item.appendChild(name);

    const size = document.createElement("div");
    size.className = "gd-file-size";
    size.textContent = _gdFormatBytes(file.size);
    item.appendChild(size);

    listEl.appendChild(item);
  });
}

function _gdShowError(errorEl, message) {
  if (!message) {
    errorEl.textContent = "";
    errorEl.classList.add("hidden");
    return;
  }
  errorEl.textContent = message;
  errorEl.classList.remove("hidden");
  clearTimeout(errorEl._gdTimer);
  errorEl._gdTimer = setTimeout(() => {
    errorEl.textContent = "";
    errorEl.classList.add("hidden");
  }, 4000);
}

function _gdApplyFilesToInput(input, files, labelEl, listEl, errorEl) {
  // accept 属性でフィルタし、不一致があればエラー表示して拒否する。
  const accepted = [];
  const rejected = [];
  Array.from(files).forEach((f) => {
    if (_gdAcceptMatches(f, input.accept)) accepted.push(f);
    else rejected.push(f);
  });

  if (rejected.length > 0) {
    _gdShowError(errorEl, `対応していないファイル形式です: ${rejected.map((f) => f.name).join(", ")}`);
  }
  if (accepted.length === 0) return;

  const dt = new DataTransfer();
  if (input.multiple) {
    accepted.forEach((f) => dt.items.add(f));
  } else {
    dt.items.add(accepted[0]);
  }
  input.files = dt.files;
  input.dispatchEvent(new Event("change", { bubbles: true }));

  labelEl.classList.add("hidden");
  _gdRenderFileList(listEl, input.files);
}

function _gdSetupOne(input) {
  if (input.dataset[GD_DROPZONE_INIT_FLAG] === "1" || input.dataset.gdSkip === "1") return;
  input.dataset[GD_DROPZONE_INIT_FLAG] = "1";

  const wrap = document.createElement("div");
  wrap.className = "gd-dropzone-wrap";
  wrap.style.position = "relative";

  input.parentNode.insertBefore(wrap, input);

  const zone = document.createElement("div");
  zone.className = "gd-dropzone";

  const label = document.createElement("p");
  label.className = "gd-dropzone-label";
  label.innerHTML = "<strong>ここにファイルをドラッグ&ドロップ</strong><br>またはクリックで選択";

  const fileList = document.createElement("div");
  fileList.className = "gd-file-list";

  const errorEl = document.createElement("p");
  errorEl.className = "gd-dropzone-error hidden";

  zone.appendChild(label);
  zone.appendChild(fileList);
  wrap.appendChild(zone);
  wrap.appendChild(errorEl);
  zone.appendChild(input);

  // 元の input はドロップゾーンいっぱいに重ねて透明化する(クリック・ドラッグ&ドロップの
  // ブラウザ標準挙動をそのまま利用するため input 自体をイベントターゲットにする)。
  input.style.position = "absolute";
  input.style.inset = "0";
  input.style.width = "100%";
  input.style.height = "100%";
  input.style.opacity = "0";
  input.style.cursor = "pointer";
  input.style.margin = "0";
  zone.style.position = "relative";

  function refreshFromInput() {
    if (input.files && input.files.length > 0) {
      label.classList.add("hidden");
      _gdRenderFileList(fileList, input.files);
    } else {
      label.classList.remove("hidden");
      fileList.innerHTML = "";
    }
  }

  input.addEventListener("change", () => {
    // 手動選択(ダイアログ経由)の場合はそのままプレビュー更新のみ行う
    // (change の dispatch はしない = 既存ハンドラの二重発火を避ける)。
    refreshFromInput();
  });

  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("gd-dragover");
  });
  zone.addEventListener("dragleave", (e) => {
    e.preventDefault();
    zone.classList.remove("gd-dragover");
  });
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("gd-dragover");
    const dtFiles = e.dataTransfer && e.dataTransfer.files;
    if (!dtFiles || dtFiles.length === 0) return;
    _gdApplyFilesToInput(input, dtFiles, label, fileList, errorEl);
  });

  refreshFromInput();
}

function initGenericDropzones(root) {
  const scope = root || document;
  scope.querySelectorAll('input[type="file"]').forEach(_gdSetupOne);
}
initGenericDropzones();

// --- ステータスバー ---
async function refreshStatus() {
  const el = document.getElementById("status-text");
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const vram = data.vram
      ? `VRAM: ${data.vram.allocated_gb.toFixed(1)}GB使用中 / ピーク${data.vram.max_allocated_gb.toFixed(1)}GB / 空き${data.vram.free_gb.toFixed(1)}GB / 総量${data.vram.total_gb.toFixed(1)}GB`
      : "VRAM: CUDA無効";
    const t2iModel = data.t2i_model || "qwen-image";
    const t2iIsLoaded = data.t2i_loaded;
    const t2iLoraAvailable = data.t2i_lora_available;
    const t2iQuant = t2iIsLoaded
      ? `(model=${t2iModel}${data.t2i_quant ? `, quant=${data.t2i_quant}` : ""}${t2iLoraAvailable === false ? " LoRA非対応" : ""})`
      : ` (model=${t2iModel} 未ロード)`;
    const editQuant = data.edit_loaded ? `(quant=${data.edit_quant || "bf16/fp8"}${data.edit_lora_available === false ? " LoRA非対応" : ""}${data.edit_lightning_merged ? " Lightning焼込済" : ""})` : "";
    const cnParts = [];
    if (data.controlnet_union_loaded) cnParts.push("Union");
    if (data.controlnet_inpaint_loaded) cnParts.push("Inpaint");
    const controlnetStatus = cnParts.length ? `ControlNet:${cnParts.join("+")}ロード済` : "ControlNet:未ロード";
    const layeredStatus = data.layered_loaded
      ? `Layered:ロード済(quant=${data.layered_quant || "bf16"}${data.layered_lightning_merged ? " Lightning焼込済" : ""})`
      : "Layered:未ロード";
    const flux2 = data.flux2 || {};
    const flux2Status = flux2.loaded
      ? `FLUX.2:ロード済(precision=${flux2.precision || "bnb-4bit"}, offload=${flux2.offload_mode || "?"})`
      : "FLUX.2:未ロード";
    const zimage = data.z_image || {};
    const zimageStatus = zimage.loaded
      ? `Z-Image:ロード済(offload=${zimage.offload_mode || "?"})`
      : "Z-Image:未ロード";
    const ltx2 = data.ltx2 || {};
    const ltx2Status = ltx2.loaded
      ? `LTX-2.3:ロード済(offload=${ltx2.offload_mode || "?"})`
      : "LTX-2.3:未ロード";
    const joyai = data.joyai || {};
    const joyaiStatus = joyai.loaded
      ? `JoyAI:ロード済(te_offload=${joyai.te_offload || "?"})`
      : "JoyAI:未ロード";
    // Mage-Flow は別プロセスのラッパーサービス(/api/mageflow/status がプロキシ)。
    // 未起動時は {available:false} が200で返る(毎回502にしない設計、app.py参照)。
    let mageflowStatus = "Mage-Flow:未起動";
    try {
      const mfRes = await fetch("/api/mageflow/status");
      const mf = await mfRes.json();
      if (mf.available === false) {
        mageflowStatus = "Mage-Flow:未起動";
      } else if (mf.loaded) {
        mageflowStatus = `Mage-Flow:ロード済(${mf.kind}/${mf.variant})${mf.busy ? "[生成中]" : ""}`;
      } else {
        mageflowStatus = "Mage-Flow:起動済・未ロード";
      }
    } catch (e) {
      /* ステータスバー全体を壊さない */
    }
    const loaded = `T2I/I2I:${data.t2i_loaded ? "ロード済" : "未ロード"}${t2iQuant} Edit:${data.edit_loaded ? "ロード済" + editQuant : "未ロード"} ${controlnetStatus} ${layeredStatus} ${flux2Status} ${zimageStatus} ${ltx2Status} ${joyaiStatus} ${mageflowStatus}`;
    const busy = data.gpu_busy ? " [GPU使用中]" : "";
    const offload = data.offload_mode ? `offload=${data.offload_mode}` : "offload=未確定(初回ロード時に決定)";
    const last = data.last_generation && data.last_generation.mode
      ? ` / 直近: ${data.last_generation.mode} ${data.last_generation.elapsed_s.toFixed(1)}s`
      : "";
    el.textContent = `${loaded} / ${offload} / ${vram}${busy}${last}`;
  } catch (e) {
    el.textContent = "ステータス取得に失敗しました";
  }
}
document.getElementById("status-refresh-btn").addEventListener("click", refreshStatus);
refreshStatus();
setInterval(refreshStatus, 8000);

// --- 解放ボタン ---
async function unload(target) {
  try {
    const res = await fetch("/api/unload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target }),
    });
    await res.json();
    refreshStatus();
  } catch (e) {
    alert("解放に失敗しました: " + e);
  }
}
document.getElementById("unload-t2i-btn").addEventListener("click", () => unload("t2i"));
document.getElementById("unload-edit-btn").addEventListener("click", () => unload("edit"));
document.getElementById("unload-controlnet-btn").addEventListener("click", () => unload("controlnet"));
document.getElementById("unload-layered-btn").addEventListener("click", () => unload("layered"));
document.getElementById("unload-flux2-btn").addEventListener("click", () => unload("flux2"));
document.getElementById("unload-zimage-btn").addEventListener("click", () => unload("zimage"));
document.getElementById("unload-ltx2-btn").addEventListener("click", () => unload("ltx2"));
document.getElementById("unload-joyai-btn").addEventListener("click", () => unload("joyai"));
// Mage-Flow は別プロセス(ラッパーサービス)のため、本体の /api/unload ではなく
// 専用の /api/mageflow/unload を叩く(サービス未起動時は502メッセージを表示)。
document.getElementById("unload-mageflow-btn").addEventListener("click", async () => {
  try {
    const res = await fetch("/api/mageflow/unload", { method: "POST" });
    await handleJsonResponse(res);
    refreshStatus();
  } catch (e) {
    alert("Mage-Flowの解放に失敗しました: " + e.message);
  }
});
document.getElementById("unload-all-btn").addEventListener("click", () => unload("all"));

// --- ギャラリー ---
function renderGallery() {
  const grid = document.getElementById("gallery-grid");
  document.getElementById("gallery-count").textContent = `${gallery.length} 件`;
  grid.innerHTML = "";
  // 新しいものを先頭に表示
  for (let i = gallery.length - 1; i >= 0; i--) {
    const item = gallery[i];
    const card = document.createElement("div");
    card.className = "gallery-card";

    const imgWrap = document.createElement("div");
    imgWrap.className = "gallery-card-image-wrap";
    const img = document.createElement("img");
    img.src = item.image_url;
    img.alt = item.mode;
    if (item.mode === "layered" || item.bg_removed) img.classList.add("checkerboard-bg");
    img.addEventListener("click", () => openLightbox(img.src));
    imgWrap.appendChild(img);
    const dlBtn = makeDownloadButton(item.image_url, null, "⬇");
    dlBtn.title = "ダウンロード";
    dlBtn.classList.add("gallery-card-download-btn");
    imgWrap.appendChild(dlBtn);
    // layered の合成/レイヤーは既にRGBA(透過含む)のため対象外にする。
    if (item.mode !== "layered") {
      const rbBtn = makeRemoveBgButton(
        img,
        (newUrl) => {
          item.image_url = newUrl; // gallery配列側も更新し、再描画時に結果を保持する
          item.bg_removed = true; // 再描画時にcheckerboard-bgを再適用するためのフラグ
          dlBtn.dataset.url = newUrl;
        },
        "✂"
      );
      rbBtn.classList.add("gallery-card-removebg-btn");
      // renderGallery() は既存カードも毎回作り直すため、bg_removed 済みのアイテムは
      // ボタンを最初から完了状態で描画する(素の"✂"のままだと、再描画のたびに
      // 既に透過済みの画像へもう一度rembgをかけ直せてしまう)。
      if (item.bg_removed) {
        rbBtn.disabled = true;
        rbBtn.textContent = "済";
        rbBtn.title = "背景削除済み";
      }
      imgWrap.appendChild(rbBtn);
    }
    card.appendChild(imgWrap);

    const body = document.createElement("div");
    body.className = "gallery-card-body";

    const modeTag = document.createElement("span");
    modeTag.className = `gallery-card-mode mode-${item.mode}`;
    modeTag.textContent = item.mode.toUpperCase();
    body.appendChild(modeTag);

    const meta = document.createElement("div");
    meta.className = "gallery-card-meta";
    const rows = [
      ["生成時間", item.elapsed_s != null ? `${item.elapsed_s.toFixed(2)} 秒` : "-"],
      ["ピークVRAM", item.peak_vram_gb != null ? `${item.peak_vram_gb.toFixed(2)} GB` : "-"],
      ["steps", item.steps],
      ["cfg", item.cfg],
    ];
    if (item.shift != null || item.mode === "t2i" || item.mode === "i2i" || item.mode === "edit") {
      rows.push(["shift", item.shift != null ? item.shift : "既定"]);
    }
    rows.push(["seed", item.seed != null ? item.seed : "ランダム"]);
    if (item.mode === "t2i" || item.mode === "i2i" || item.mode === "edit") {
      rows.push(["Lightning", item.lightning ? "ON" : "OFF"]);
    }
    if (item.strength != null) rows.splice(3, 0, ["strength", item.strength]);
    if (item.num_reference_images != null) rows.push(["参照画像枚数", item.num_reference_images]);
    if (item.width) rows.splice(2, 0, ["サイズ", `${item.width}x${item.height}`]);
    if (item.model && item.model !== "qwen-image") rows.push(["モデル", item.model]);
    if (item.quant) rows.push(["量子化", item.quant]);
    if (item.offload_mode) rows.push(["offload", item.offload_mode]);
    if (item.control_type) rows.push(["制御タイプ", item.control_type]);
    if (item.controlnet_conditioning_scale != null) rows.push(["conditioning_scale", item.controlnet_conditioning_scale]);
    if (item.mode === "layered") {
      rows.push(["レイヤー数", item.layers]);
      rows.push(["resolution", item.resolution]);
      rows.push(["Lightning焼込", item.lightning_merged ? "済" : "未(GGUF等)"]);
    }
    rows.forEach(([k, v]) => {
      const row = document.createElement("div");
      row.innerHTML = `<span class="key">${k}</span><span class="val">${v}</span>`;
      meta.appendChild(row);
    });
    body.appendChild(meta);

    if (item.lightning_unavailable_reason) {
      const warn = document.createElement("div");
      warn.className = "gallery-card-warning";
      warn.textContent = `⚠ ${item.lightning_unavailable_reason}`;
      body.appendChild(warn);
    }

    const prompt = document.createElement("div");
    prompt.className = "gallery-card-prompt";
    prompt.textContent = item.prompt;
    body.appendChild(prompt);

    card.appendChild(body);
    grid.appendChild(card);
  }
}

document.getElementById("gallery-clear-btn").addEventListener("click", () => {
  gallery.length = 0;
  renderGallery();
});

// --- ライトボックス ---
const lightbox = document.getElementById("lightbox");
const lightboxImage = document.getElementById("lightbox-image");
function openLightbox(url) {
  lightboxImage.src = url;
  lightbox.classList.remove("hidden");
}
lightbox.addEventListener("click", () => lightbox.classList.add("hidden"));

// --- 共通: 生成物(画像/動画/音声)のダウンロード ---
// url は同一オリジンの /outputs/... を想定(a[download] がそのまま機能する)。
function downloadFile(url, filename) {
  if (!url) return;
  const a = document.createElement("a");
  a.href = url;
  a.download = filename || url.split("/").pop().split("?")[0];
  document.body.appendChild(a);
  a.click();
  a.remove();
}

function makeDownloadButton(url, filename, label = "ダウンロード") {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "tiny-btn secondary download-btn";
  btn.textContent = label;
  // dataset経由でURLを保持する(クロージャではなくbtn自身に持たせることで、
  // 背景削除ボタン(makeRemoveBgButton)が画像差し替え後にこのボタンのDL先も
  // 差し替えられるようにするため)。
  btn.dataset.url = url;
  if (filename) btn.dataset.filename = filename;
  btn.addEventListener("click", (ev) => {
    ev.stopPropagation();
    downloadFile(btn.dataset.url, btn.dataset.filename || null);
  });
  return btn;
}

// --- 共通: 表示中の画像に対してその場で背景削除を実行するボタン(動画タブ以外で使用) ---
// imgEl の現在の src を /api/remove_bg に送り、成功したら imgEl.src を結果画像に差し替える
// (checkerboard-bg 表示に切り替え、完了後は再実行不可にする。透過済み画像に対して
// rembg をもう一度かけると、アルファ抜け部分がRGB変換で黒潰れする等の不具合が
// 起きうるため)。onReplaced(newUrl) は差し替え後にダウンロードボタンのURL更新等を
// 行うためのコールバック。
function makeRemoveBgButton(imgEl, onReplaced, label = "背景削除") {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "tiny-btn secondary removebg-instant-btn";
  btn.textContent = label;
  btn.title = "この画像の背景を削除(透過PNG化)";
  btn.addEventListener("click", async (ev) => {
    ev.stopPropagation();
    if (btn.disabled) return;
    const originalHtml = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner-inline spinner-inline-secondary"></span>`;
    try {
      const srcRes = await fetch(imgEl.src);
      const blob = await srcRes.blob();
      const fd = new FormData();
      fd.append("image", blob, "image.png");
      const res = await fetch("/api/remove_bg", { method: "POST", body: fd });
      const data = await handleJsonResponse(res);
      imgEl.src = data.image_url;
      imgEl.classList.add("checkerboard-bg");
      btn.textContent = "済";
      btn.title = "背景削除済み";
      if (onReplaced) onReplaced(data.image_url);
    } catch (e) {
      btn.disabled = false;
      btn.innerHTML = originalHtml;
      alert("背景削除に失敗しました: " + e.message);
    }
  });
  return btn;
}

// --- 生成進捗バー(/api/progress を500ms間隔でポーリングして表示) ---
// 対象タブの result-panel の直前に進捗バーを動的挿入する(showLatestResult() は
// result-panel の innerHTML を丸ごと差し替えるため、その外側に置くことで両者は独立に
// 動作する)。1回の生成につき1回 startProgressPolling(panelId) を呼び、レスポンス到着
// (成功/エラー問わず) の finally で stopProgressPolling(panelId) を呼ぶ運用にする。
const _progressPollers = {}; // panelId -> intervalId

function _ensureProgressBar(panelId) {
  const panel = document.getElementById(panelId);
  if (!panel) return null;
  let bar = document.getElementById(`${panelId}-progress`);
  if (!bar) {
    bar = document.createElement("div");
    bar.id = `${panelId}-progress`;
    bar.className = "progress-bar-wrap hidden";
    bar.innerHTML = `
      <div class="progress-bar-label">
        <span class="progress-bar-phase">-</span>
        <span class="progress-bar-count"></span>
      </div>
      <div class="progress-bar-track">
        <div class="progress-bar-fill"></div>
      </div>
      <div class="progress-bar-sub"></div>
    `;
    panel.parentNode.insertBefore(bar, panel);
  }
  return bar;
}

const _PHASE_LABEL = {
  loading: "モデルロード中…",
  generating: "生成中",
  decoding: "画像デコード中…",
};

function _renderProgress(panelId, data) {
  const bar = _ensureProgressBar(panelId);
  if (!bar) return;
  const track = bar.querySelector(".progress-bar-track");
  const fill = bar.querySelector(".progress-bar-fill");
  const phaseEl = bar.querySelector(".progress-bar-phase");
  const countEl = bar.querySelector(".progress-bar-count");
  const subEl = bar.querySelector(".progress-bar-sub");

  if (!data || !data.active) {
    bar.classList.add("hidden");
    return;
  }
  bar.classList.remove("hidden");

  const phase = data.phase || "generating";
  phaseEl.textContent = _PHASE_LABEL[phase] || phase;

  const determinate = phase === "generating" && data.total_steps > 0;
  if (determinate) {
    track.classList.remove("indeterminate");
    const pct = Math.max(0, Math.min(100, (data.step / data.total_steps) * 100));
    fill.style.width = `${pct}%`;
    countEl.textContent = `${data.step} / ${data.total_steps}`;
  } else {
    track.classList.add("indeterminate");
    fill.style.width = "";
    countEl.textContent = "";
  }

  const extra = data.extra || null;
  if (extra && extra.direction_total) {
    subEl.textContent = `方向 ${extra.direction_index}/${extra.direction_total}${extra.direction_label ? `(${extra.direction_label})` : ""}`;
  } else if (data.elapsed_s != null) {
    subEl.textContent = `経過 ${data.elapsed_s.toFixed(1)} 秒`;
  } else {
    subEl.textContent = "";
  }
}

async function _pollProgressOnce(panelId) {
  try {
    const res = await fetch("/api/progress");
    if (!res.ok) return;
    const data = await res.json();
    _renderProgress(panelId, data);
  } catch (e) {
    /* ポーリング失敗は無視(次回リトライ) */
  }
}

function startProgressPolling(panelId) {
  stopProgressPolling(panelId);
  _ensureProgressBar(panelId);
  _pollProgressOnce(panelId);
  _progressPollers[panelId] = setInterval(() => _pollProgressOnce(panelId), 500);
}

function stopProgressPolling(panelId) {
  if (_progressPollers[panelId]) {
    clearInterval(_progressPollers[panelId]);
    delete _progressPollers[panelId];
  }
  const bar = document.getElementById(`${panelId}-progress`);
  if (bar) bar.classList.add("hidden");
}

// --- 各タブ右カラムの「直近の結果」パネル表示 ---
function showLatestResult(panelId, data) {
  const panel = document.getElementById(panelId);
  if (!panel) return;
  panel.innerHTML = "";

  const img = document.createElement("img");
  img.className = "result-panel-image";
  img.src = data.image_url;
  img.alt = data.mode || "生成結果";
  if (data.mode === "layered" || data.mode === "remove_bg") img.classList.add("checkerboard-bg");
  img.addEventListener("click", () => openLightbox(img.src));
  panel.appendChild(img);

  const downloadRow = document.createElement("div");
  downloadRow.className = "result-download-row";
  const dlBtn = makeDownloadButton(data.image_url, null, "画像をダウンロード");
  downloadRow.appendChild(dlBtn);
  // remove_bg タブ自身の結果は既に背景削除済みなのでボタンを出さない(動画タブは
  // showLtx2Result 側が別実装のためそもそも対象外)。
  if (data.mode !== "remove_bg") {
    downloadRow.appendChild(
      makeRemoveBgButton(img, (newUrl) => {
        dlBtn.dataset.url = newUrl;
      })
    );
  }
  panel.appendChild(downloadRow);

  const meta = document.createElement("div");
  meta.className = "result-panel-meta";
  const rows = [["生成時間", data.elapsed_s != null ? `${data.elapsed_s.toFixed(2)} 秒` : "-"]];
  // remove_bg 等のGPU不使用ユーティリティは peak_vram_gb/steps/cfg/seed を返さないため、
  // 存在するフィールドのみ行として追加する(空欄「-」の羅列を避ける)。
  if (data.peak_vram_gb != null) rows.push(["ピークVRAM", `${data.peak_vram_gb.toFixed(2)} GB`]);
  if (data.steps != null) rows.push(["steps", data.steps]);
  if (data.cfg != null) rows.push(["cfg", data.cfg]);
  if (data.seed !== undefined) rows.push(["seed", data.seed != null ? data.seed : "ランダム"]);
  if (data.width) rows.push(["サイズ", `${data.width}x${data.height}`]);
  rows.forEach(([k, v]) => {
    const row = document.createElement("div");
    row.innerHTML = `<span class="key">${k}</span><span class="val">${v}</span>`;
    meta.appendChild(row);
  });
  panel.appendChild(meta);
}

// --- 共通: フォーム送信ヘルパ ---
function setFormBusy(form, busy) {
  const btn = form.querySelector("button[type=submit]");
  btn.disabled = busy;
  btn.dataset.originalText = btn.dataset.originalText || btn.textContent;
  btn.innerHTML = busy ? `<span class="spinner-inline"></span>生成中...` : btn.dataset.originalText;
}

function showError(id, message) {
  const el = document.getElementById(id);
  if (!message) {
    el.classList.add("hidden");
    el.textContent = "";
    return;
  }
  el.textContent = message;
  el.classList.remove("hidden");
}

async function handleJsonResponse(res) {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const data = await res.json();
      detail = data.detail || detail;
    } catch (e) {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

// --- LLMプロンプト支援(LLM強化 / 英訳 / 元に戻す) ---
// プロンプトtextareaを持つ各フォーム(T2I/I2I/Edit/ControlNet/Inpaint/Layered)に、
// 近くの .llm-btn-row 内の3ボタンを配線する。押すと textarea の内容を対象APIに送り、
// 結果で置き換える。直前の内容は1世代だけ保持し「元に戻す」で戻せる。
// 生成(GPU)機能とは完全に独立したAPI(/api/prompt/*)を叩くだけなので、
// LLMサーバが未接続でも画像生成そのものには一切影響しない。
function setupLlmButtons(formId, errorId) {
  const form = document.getElementById(formId);
  if (!form) return;
  const row = form.querySelector(".llm-btn-row");
  const textarea = form.querySelector('textarea[name="prompt"]');
  if (!row || !textarea) return;

  const enhanceBtn = row.querySelector(".llm-enhance-btn");
  const translateBtn = row.querySelector(".llm-translate-btn");
  const undoBtn = row.querySelector(".llm-undo-btn");
  let previousValue = null;

  function setBusy(busy) {
    [enhanceBtn, translateBtn, undoBtn].forEach((b) => {
      if (b) b.disabled = busy;
    });
  }

  function setBtnLoading(btn, loading, label) {
    if (!btn) return;
    btn.dataset.originalText = btn.dataset.originalText || btn.textContent;
    btn.innerHTML = loading
      ? `<span class="spinner-inline spinner-inline-secondary"></span>${label}`
      : btn.dataset.originalText;
  }

  async function callLlmApi(endpoint, btn, loadingLabel) {
    const text = textarea.value;
    if (!text || !text.trim()) return;
    showError(errorId, null);
    previousValue = textarea.value;
    setBusy(true);
    setBtnLoading(btn, true, loadingLabel);
    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await handleJsonResponse(res);
      textarea.value = data.result;
      undoBtn.classList.remove("hidden");
    } catch (e) {
      showError(errorId, e.message);
      previousValue = null;
    } finally {
      setBusy(false);
      setBtnLoading(btn, false, loadingLabel);
    }
  }

  enhanceBtn.addEventListener("click", () => callLlmApi("/api/prompt/enhance", enhanceBtn, "強化中..."));
  translateBtn.addEventListener("click", () => callLlmApi("/api/prompt/translate", translateBtn, "翻訳中..."));
  undoBtn.addEventListener("click", () => {
    if (previousValue === null) return;
    textarea.value = previousValue;
    previousValue = null;
    undoBtn.classList.add("hidden");
  });
}
setupLlmButtons("t2i-form", "t2i-error");
setupLlmButtons("i2i-form", "i2i-error");
setupLlmButtons("edit-form", "edit-error");
setupLlmButtons("controlnet-form", "controlnet-error");
setupLlmButtons("inpaint-form", "inpaint-error");
setupLlmButtons("layered-form", "layered-error");

// FLUX.2: モデル選択に応じて steps/guidance_scale の既定値を切り替える(既定 dev のみ、
// 2026-07-19 ecocoro廃止)。
const FLUX2_MODEL_DEFAULTS = {
  dev: { steps: 28, guidance_scale: 4.0 },
};

// --- T2I / I2I: モデル選択に FLUX.2(dev)・Z-Image-Turbo を統合 ---
// select の value は "2512" | "qwen-image" | "flux2-dev" | "zimage-turbo"。
// flux2-* / zimage-* 選択時はそれぞれ /api/flux2/*, /api/zimage/* に振り分け、
// Qwen固有パラメータ欄(negative_prompt/shift/Lightning/strength)を隠し、
// 該当ファミリー固有のパラメータ欄を表示する。
function modelFamilyOf(modelValue) {
  if (modelValue === "flux2-dev") return "flux2";
  if (modelValue === "zimage-turbo") return "zimage";
  return "qwen";
}
function isFlux2Model(modelValue) {
  return modelFamilyOf(modelValue) === "flux2";
}
function isZimageModel(modelValue) {
  return modelFamilyOf(modelValue) === "zimage";
}
function flux2ModelName(_modelValue) {
  return "dev";
}

function setupModelFieldToggle(formId, modelSelectId) {
  const form = document.getElementById(formId);
  const modelSelect = document.getElementById(modelSelectId);
  if (!form || !modelSelect) return;
  const qwenFields = form.querySelectorAll(".qwen-only-fields");
  const flux2Fields = form.querySelectorAll(".flux2-only-fields");
  const zimageFields = form.querySelectorAll(".zimage-only-fields");
  const flux2ImageHint = document.getElementById("i2i-flux2-image-hint");
  const imageInput = document.getElementById("i2i-image-input");

  function applyFlux2FieldDefaults() {
    const defaults = FLUX2_MODEL_DEFAULTS[flux2ModelName(modelSelect.value)] || FLUX2_MODEL_DEFAULTS.dev;
    const stepsInput = form.querySelector('[name="flux2_steps"]');
    const guidanceInput = form.querySelector('[name="guidance_scale"]');
    if (stepsInput) stepsInput.value = defaults.steps;
    if (guidanceInput) guidanceInput.value = defaults.guidance_scale;
  }

  function update() {
    const family = modelFamilyOf(modelSelect.value);
    const isFlux2 = family === "flux2";
    const isZimage = family === "zimage";
    qwenFields.forEach((el) => el.classList.toggle("hidden", isFlux2 || isZimage));
    flux2Fields.forEach((el) => el.classList.toggle("hidden", !isFlux2));
    zimageFields.forEach((el) => el.classList.toggle("hidden", !isZimage));
    if (isFlux2) applyFlux2FieldDefaults();
    // I2I: FLUX.2選択時は参照画像の複数アップロードに切り替える(strengthではなく参照条件付け)。
    // Z-Image選択時は単一画像アップロード(Qwenと同じ)のまま。
    if (imageInput) {
      if (isFlux2) {
        imageInput.setAttribute("multiple", "multiple");
        imageInput.name = "images";
      } else {
        imageInput.removeAttribute("multiple");
        imageInput.name = "image";
      }
    }
    if (flux2ImageHint) flux2ImageHint.classList.toggle("hidden", !isFlux2);
  }

  modelSelect.addEventListener("change", update);
  update();
}
setupModelFieldToggle("t2i-form", "t2i-model-select");
setupModelFieldToggle("i2i-form", "i2i-model-select");

// --- T2I フォーム ---
document.getElementById("t2i-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  showError("t2i-error", null);
  const fd = new FormData(form);
  const modelValue = fd.get("model") || "";
  setFormBusy(form, true);
  startProgressPolling("t2i-result-panel");
  try {
    let res;
    if (isFlux2Model(modelValue)) {
      const body = {
        prompt: fd.get("prompt"),
        width: parseInt(fd.get("flux2_width"), 10),
        height: parseInt(fd.get("flux2_height"), 10),
        steps: parseInt(fd.get("flux2_steps"), 10),
        guidance_scale: parseFloat(fd.get("guidance_scale")),
        seed: parseInt(fd.get("flux2_seed"), 10),
        model: flux2ModelName(modelValue),
      };
      res = await fetch("/api/flux2/t2i", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } else if (isZimageModel(modelValue)) {
      const body = {
        prompt: fd.get("prompt"),
        negative_prompt: fd.get("zimage_negative_prompt") || "",
        width: parseInt(fd.get("zimage_width"), 10),
        height: parseInt(fd.get("zimage_height"), 10),
        steps: parseInt(fd.get("zimage_steps"), 10),
        guidance_scale: parseFloat(fd.get("zimage_guidance_scale")),
        seed: parseInt(fd.get("zimage_seed"), 10),
      };
      res = await fetch("/api/zimage/t2i", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } else {
      const body = {
        prompt: fd.get("prompt"),
        negative_prompt: fd.get("negative_prompt") || "",
        width: parseInt(fd.get("width"), 10),
        height: parseInt(fd.get("height"), 10),
        steps: parseInt(fd.get("steps"), 10),
        cfg: parseFloat(fd.get("cfg")),
        seed: parseInt(fd.get("seed"), 10),
        lightning: fd.get("lightning") === "on",
        shift: fd.get("shift") ? parseFloat(fd.get("shift")) : null,
        model: modelValue || null,
      };
      res = await fetch("/api/t2i", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    }
    const data = await handleJsonResponse(res);
    gallery.push(data);
    renderGallery();
    showLatestResult("t2i-result-panel", data);
  } catch (e) {
    showError("t2i-error", e.message);
  } finally {
    stopProgressPolling("t2i-result-panel");
    setFormBusy(form, false);
    refreshStatus();
  }
});

// --- I2I フォーム ---
document.getElementById("i2i-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  showError("i2i-error", null);
  const modelValue = form.querySelector('[name="model"]').value || "";
  setFormBusy(form, true);
  startProgressPolling("i2i-result-panel");
  try {
    let res;
    if (isFlux2Model(modelValue)) {
      const imageInput = document.getElementById("i2i-image-input");
      const fd = new FormData();
      for (const file of imageInput.files) fd.append("images", file);
      fd.set("prompt", form.querySelector('[name="prompt"]').value);
      fd.set("width", form.querySelector('[name="flux2_width"]').value);
      fd.set("height", form.querySelector('[name="flux2_height"]').value);
      fd.set("steps", form.querySelector('[name="flux2_steps"]').value);
      fd.set("guidance_scale", form.querySelector('[name="guidance_scale"]').value);
      fd.set("seed", form.querySelector('[name="flux2_seed"]').value);
      fd.set("model", flux2ModelName(modelValue));
      res = await fetch("/api/flux2/i2i", { method: "POST", body: fd });
    } else if (isZimageModel(modelValue)) {
      const imageInput = document.getElementById("i2i-image-input");
      const fd = new FormData();
      if (imageInput.files[0]) fd.append("image", imageInput.files[0]);
      fd.set("prompt", form.querySelector('[name="prompt"]').value);
      fd.set("negative_prompt", form.querySelector('[name="zimage_negative_prompt"]').value || "");
      fd.set("strength", form.querySelector('[name="zimage_strength"]').value);
      fd.set("width", form.querySelector('[name="zimage_width"]').value);
      fd.set("height", form.querySelector('[name="zimage_height"]').value);
      fd.set("steps", form.querySelector('[name="zimage_steps"]').value);
      fd.set("guidance_scale", form.querySelector('[name="zimage_guidance_scale"]').value);
      fd.set("seed", form.querySelector('[name="zimage_seed"]').value);
      res = await fetch("/api/zimage/i2i", { method: "POST", body: fd });
    } else {
      const fd = new FormData(form);
      fd.set("lightning", form.querySelector('[name="lightning"]').checked ? "true" : "false");
      if (!fd.get("shift")) fd.delete("shift");
      res = await fetch("/api/i2i", { method: "POST", body: fd });
    }
    const data = await handleJsonResponse(res);
    gallery.push(data);
    renderGallery();
    showLatestResult("i2i-result-panel", data);
  } catch (e) {
    showError("i2i-error", e.message);
  } finally {
    stopProgressPolling("i2i-result-panel");
    setFormBusy(form, false);
    refreshStatus();
  }
});

// --- Edit フォーム: モデル選択("qwen-edit-2511"(既定) | "joyai-edit-plus")に応じて
// 表示フィールドを切り替え、送信先APIを振り分ける。
(() => {
  const form = document.getElementById("edit-form");
  const modelSelect = document.getElementById("edit-model-select");
  if (!form || !modelSelect) return;
  const qwenFields = form.querySelectorAll(".qwen-edit-only-fields");
  const joyaiFields = form.querySelectorAll(".joyai-only-fields");
  function update() {
    const isJoyai = modelSelect.value === "joyai-edit-plus";
    qwenFields.forEach((el) => el.classList.toggle("hidden", isJoyai));
    joyaiFields.forEach((el) => el.classList.toggle("hidden", !isJoyai));
  }
  modelSelect.addEventListener("change", update);
  update();
})();

document.getElementById("edit-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  showError("edit-error", null);
  const modelValue = form.querySelector('[name="model"]').value;
  setFormBusy(form, true);
  startProgressPolling("edit-result-panel");
  try {
    let res;
    if (modelValue === "joyai-edit-plus") {
      const fd = new FormData();
      const imagesInput = form.querySelector('[name="images"]');
      for (const file of imagesInput.files) fd.append("images", file);
      fd.set("prompt", form.querySelector('[name="prompt"]').value);
      fd.set("negative_prompt", form.querySelector('[name="negative_prompt"]').value || "");
      fd.set("steps", form.querySelector('[name="joyai_steps"]').value);
      fd.set("guidance_scale", form.querySelector('[name="joyai_guidance_scale"]').value);
      fd.set("seed", form.querySelector('[name="joyai_seed"]').value);
      const w = form.querySelector('[name="joyai_width"]').value;
      const h = form.querySelector('[name="joyai_height"]').value;
      if (w) fd.set("width", w);
      if (h) fd.set("height", h);
      res = await fetch("/api/joyai/edit", { method: "POST", body: fd });
    } else {
      const fd = new FormData(form);
      fd.delete("joyai_steps");
      fd.delete("joyai_guidance_scale");
      fd.delete("joyai_seed");
      fd.delete("joyai_width");
      fd.delete("joyai_height");
      fd.delete("model");
      fd.set("lightning", form.querySelector('[name="lightning"]').checked ? "true" : "false");
      if (!fd.get("shift")) fd.delete("shift");
      res = await fetch("/api/edit", { method: "POST", body: fd });
    }
    const data = await handleJsonResponse(res);
    gallery.push(data);
    renderGallery();
    showLatestResult("edit-result-panel", data);
  } catch (e) {
    showError("edit-error", e.message);
  } finally {
    stopProgressPolling("edit-result-panel");
    setFormBusy(form, false);
    refreshStatus();
  }
});

// --- ControlNet フォーム ---
const controlnetImageInput = document.getElementById("controlnet-image-input");
const controlnetPreview = document.getElementById("controlnet-preview");
const controlnetCannyBtn = document.getElementById("controlnet-canny-btn");
const controlnetCannyStatus = document.getElementById("controlnet-canny-status");
let controlnetOverrideBlob = null; // Canny生成結果を使う場合、元ファイルの代わりにこれを送信する

function showControlnetPreview(url) {
  controlnetPreview.src = url;
  controlnetPreview.classList.remove("hidden");
}

controlnetImageInput.addEventListener("change", () => {
  controlnetOverrideBlob = null; // 新しいファイルを選び直したらCanny上書きをリセット
  controlnetCannyStatus.textContent = "";
  const file = controlnetImageInput.files[0];
  if (file) showControlnetPreview(URL.createObjectURL(file));
});

controlnetCannyBtn.addEventListener("click", async () => {
  const file = controlnetImageInput.files[0];
  if (!file) {
    alert("先に制御画像(元にする写真)を選択してください。");
    return;
  }
  controlnetCannyStatus.textContent = "Canny生成中...";
  try {
    const fd = new FormData();
    fd.set("image", file);
    const res = await fetch("/api/canny", { method: "POST", body: fd });
    const data = await handleJsonResponse(res);
    const imgRes = await fetch(data.image_url);
    controlnetOverrideBlob = await imgRes.blob();
    showControlnetPreview(data.image_url);
    controlnetCannyStatus.textContent = "Canny生成完了(この画像を制御画像として使用します)";
    const typeSelect = document.querySelector('#controlnet-form [name="control_type"]');
    if (typeSelect) typeSelect.value = "canny";
  } catch (e) {
    controlnetCannyStatus.textContent = "";
    alert("Canny生成に失敗しました: " + e.message);
  }
});

document.getElementById("controlnet-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  showError("controlnet-error", null);
  const fd = new FormData(form);
  if (controlnetOverrideBlob) {
    fd.set("control_image", controlnetOverrideBlob, "canny.png");
  }
  if (!fd.get("width")) fd.delete("width");
  if (!fd.get("height")) fd.delete("height");
  setFormBusy(form, true);
  startProgressPolling("controlnet-result-panel");
  try {
    const res = await fetch("/api/controlnet", { method: "POST", body: fd });
    const data = await handleJsonResponse(res);
    gallery.push(data);
    renderGallery();
    showLatestResult("controlnet-result-panel", data);
  } catch (e) {
    showError("controlnet-error", e.message);
  } finally {
    stopProgressPolling("controlnet-result-panel");
    setFormBusy(form, false);
    refreshStatus();
  }
});

// --- Inpaint フォーム(マスクをキャンバスにブラシ描画する簡易エディタ) ---
const inpaintImageInput = document.getElementById("inpaint-image-input");
const inpaintMaskEditor = document.getElementById("inpaint-mask-editor");
const inpaintBaseCanvas = document.getElementById("inpaint-base-canvas");
const inpaintMaskCanvas = document.getElementById("inpaint-mask-canvas");
const inpaintMaskUpload = document.getElementById("inpaint-mask-upload");
const maskBrushSizeInput = document.getElementById("mask-brush-size");
const maskModeBrushBtn = document.getElementById("mask-mode-brush");
const maskModeEraserBtn = document.getElementById("mask-mode-eraser");
const maskClearBtn = document.getElementById("mask-clear-btn");

let maskEraserMode = false;
let maskDrawing = false;
let inpaintNaturalWidth = 0;
let inpaintNaturalHeight = 0;
const MASK_CANVAS_MAX_DISPLAY = 800; // 表示・描画用キャンバスの最大幅(送信時は元画像解像度に拡大)

inpaintImageInput.addEventListener("change", () => {
  const file = inpaintImageInput.files[0];
  if (!file) return;
  const img = new Image();
  img.onload = () => {
    inpaintNaturalWidth = img.naturalWidth;
    inpaintNaturalHeight = img.naturalHeight;
    const scale = Math.min(1, MASK_CANVAS_MAX_DISPLAY / img.naturalWidth);
    const dispW = Math.round(img.naturalWidth * scale);
    const dispH = Math.round(img.naturalHeight * scale);
    [inpaintBaseCanvas, inpaintMaskCanvas].forEach((c) => {
      c.width = dispW;
      c.height = dispH;
    });
    const baseCtx = inpaintBaseCanvas.getContext("2d");
    baseCtx.clearRect(0, 0, dispW, dispH);
    baseCtx.drawImage(img, 0, 0, dispW, dispH);
    const maskCtx = inpaintMaskCanvas.getContext("2d");
    maskCtx.clearRect(0, 0, dispW, dispH);
    inpaintMaskEditor.classList.remove("hidden");
  };
  img.src = URL.createObjectURL(file);
});

function maskPointerPos(ev) {
  const rect = inpaintMaskCanvas.getBoundingClientRect();
  const scaleX = inpaintMaskCanvas.width / rect.width;
  const scaleY = inpaintMaskCanvas.height / rect.height;
  return {
    x: (ev.clientX - rect.left) * scaleX,
    y: (ev.clientY - rect.top) * scaleY,
  };
}

function maskStroke(x, y) {
  const ctx = inpaintMaskCanvas.getContext("2d");
  const size = parseInt(maskBrushSizeInput.value, 10);
  ctx.globalCompositeOperation = maskEraserMode ? "destination-out" : "source-over";
  ctx.fillStyle = "rgba(255,255,255,1)";
  ctx.beginPath();
  ctx.arc(x, y, size / 2, 0, Math.PI * 2);
  ctx.fill();
}

inpaintMaskCanvas.addEventListener("pointerdown", (ev) => {
  maskDrawing = true;
  const p = maskPointerPos(ev);
  maskStroke(p.x, p.y);
});
inpaintMaskCanvas.addEventListener("pointermove", (ev) => {
  if (!maskDrawing) return;
  const p = maskPointerPos(ev);
  maskStroke(p.x, p.y);
});
window.addEventListener("pointerup", () => {
  maskDrawing = false;
});

maskModeBrushBtn.addEventListener("click", () => {
  maskEraserMode = false;
  maskModeBrushBtn.classList.add("active-tool");
  maskModeEraserBtn.classList.remove("active-tool", "secondary");
  maskModeBrushBtn.classList.remove("secondary");
  maskModeEraserBtn.classList.add("secondary");
});
maskModeEraserBtn.addEventListener("click", () => {
  maskEraserMode = true;
  maskModeEraserBtn.classList.add("active-tool");
  maskModeBrushBtn.classList.remove("active-tool", "secondary");
  maskModeEraserBtn.classList.remove("secondary");
  maskModeBrushBtn.classList.add("secondary");
});
maskClearBtn.addEventListener("click", () => {
  const ctx = inpaintMaskCanvas.getContext("2d");
  ctx.globalCompositeOperation = "source-over";
  ctx.clearRect(0, 0, inpaintMaskCanvas.width, inpaintMaskCanvas.height);
});

function exportMaskBlob() {
  // マスクキャンバス(白=編集領域、透明=背景)を元画像の解像度に拡大し、
  // 黒背景の上に合成することでB/Wのマスク画像を作る。
  return new Promise((resolve) => {
    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = inpaintNaturalWidth || inpaintMaskCanvas.width;
    exportCanvas.height = inpaintNaturalHeight || inpaintMaskCanvas.height;
    const ctx = exportCanvas.getContext("2d");
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
    ctx.drawImage(inpaintMaskCanvas, 0, 0, exportCanvas.width, exportCanvas.height);
    exportCanvas.toBlob(resolve, "image/png");
  });
}

// モデル選択("qwen-controlnet"(既定) | "zimage-turbo")に応じて表示フィールドを切り替える。
(() => {
  const form = document.getElementById("inpaint-form");
  const modelSelect = document.getElementById("inpaint-model-select");
  if (!form || !modelSelect) return;
  const qwenFields = form.querySelectorAll(".qwen-inpaint-only-fields");
  const zimageFields = form.querySelectorAll(".zimage-inpaint-only-fields");
  function update() {
    const isZimage = modelSelect.value === "zimage-turbo";
    qwenFields.forEach((el) => el.classList.toggle("hidden", isZimage));
    zimageFields.forEach((el) => el.classList.toggle("hidden", !isZimage));
  }
  modelSelect.addEventListener("change", update);
  update();
})();

document.getElementById("inpaint-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  showError("inpaint-error", null);

  const uploadedMask = inpaintMaskUpload.files[0];
  if (!uploadedMask && inpaintNaturalWidth === 0) {
    showError("inpaint-error", "元画像を選択してマスクを描くか、マスク画像をアップロードしてください。");
    return;
  }

  const modelValue = form.querySelector('[name="model"]').value || "qwen-controlnet";
  const isZimage = modelValue === "zimage-turbo";

  let maskFile;
  if (uploadedMask) {
    maskFile = uploadedMask;
  } else {
    const maskBlob = await exportMaskBlob();
    if (!maskBlob) {
      showError("inpaint-error", "マスク画像の生成に失敗しました。");
      return;
    }
    maskFile = maskBlob;
  }

  setFormBusy(form, true);
  startProgressPolling("inpaint-result-panel");
  try {
    let res;
    if (isZimage) {
      const fd = new FormData();
      fd.set("image", form.querySelector('[name="image"]').files[0]);
      fd.set("mask_image", maskFile, maskFile.name || "mask.png");
      fd.set("prompt", form.querySelector('[name="prompt"]').value);
      fd.set("negative_prompt", form.querySelector('[name="zimage_negative_prompt"]').value || "");
      fd.set("strength", form.querySelector('[name="zimage_strength"]').value);
      fd.set("width", form.querySelector('[name="zimage_width"]').value);
      fd.set("height", form.querySelector('[name="zimage_height"]').value);
      fd.set("steps", form.querySelector('[name="zimage_steps"]').value);
      fd.set("guidance_scale", form.querySelector('[name="zimage_guidance_scale"]').value);
      fd.set("seed", form.querySelector('[name="zimage_seed"]').value);
      res = await fetch("/api/zimage/inpaint", { method: "POST", body: fd });
    } else {
      const fd = new FormData(form);
      fd.delete("mask_image_upload");
      fd.delete("model");
      fd.set("mask_image", maskFile, maskFile.name || "mask.png");
      res = await fetch("/api/inpaint", { method: "POST", body: fd });
    }
    const data = await handleJsonResponse(res);
    gallery.push(data);
    renderGallery();
    showLatestResult("inpaint-result-panel", data);
  } catch (e) {
    showError("inpaint-error", e.message);
  } finally {
    stopProgressPolling("inpaint-result-panel");
    setFormBusy(form, false);
    refreshStatus();
  }
});

// --- Layered フォーム(画像 -> 複数RGBAレイヤー分解) ---
const layeredImageInput = document.getElementById("layered-image-input");
const layeredInputPreview = document.getElementById("layered-input-preview");
const layeredResult = document.getElementById("layered-result");
const layeredCompositeImg = document.getElementById("layered-composite-img");
const layeredLayersRow = document.getElementById("layered-layers-row");

layeredImageInput.addEventListener("change", () => {
  const file = layeredImageInput.files[0];
  if (!file) return;
  layeredInputPreview.src = URL.createObjectURL(file);
  layeredInputPreview.classList.remove("hidden");
});

function renderLayeredResult(data) {
  layeredCompositeImg.src = data.composite_image_url;
  const compositeDownloadRow = document.getElementById("layered-composite-download-row");
  if (compositeDownloadRow) {
    compositeDownloadRow.innerHTML = "";
    const compositeDlBtn = makeDownloadButton(data.composite_image_url, null, "合成画像をダウンロード");
    compositeDownloadRow.appendChild(compositeDlBtn);
    compositeDownloadRow.appendChild(
      makeRemoveBgButton(layeredCompositeImg, (newUrl) => {
        compositeDlBtn.dataset.url = newUrl;
      })
    );
  }
  layeredLayersRow.innerHTML = "";
  data.layer_image_urls.forEach((url, idx) => {
    const thumb = document.createElement("div");
    thumb.className = "layered-layer-thumb checkerboard-bg";
    const img = document.createElement("img");
    img.src = url;
    img.alt = `layer ${idx}`;
    img.addEventListener("click", () => openLightbox(url));
    thumb.appendChild(img);
    const label = document.createElement("span");
    label.className = "layered-layer-label";
    label.textContent = `layer ${idx}`;
    thumb.appendChild(label);
    const dlBtn = makeDownloadButton(url, null, "DL");
    dlBtn.classList.add("layered-layer-download-btn");
    thumb.appendChild(dlBtn);
    layeredLayersRow.appendChild(thumb);
  });
  layeredResult.classList.remove("hidden");
  const placeholder = document.getElementById("layered-result-placeholder");
  if (placeholder) placeholder.classList.add("hidden");
}

document.getElementById("layered-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  showError("layered-error", null);
  const fd = new FormData(form);
  fd.set("cfg_normalize", form.querySelector('[name="cfg_normalize"]').checked ? "true" : "false");
  fd.set("use_en_prompt", form.querySelector('[name="use_en_prompt"]').checked ? "true" : "false");
  if (!fd.get("steps")) fd.delete("steps");
  if (!fd.get("cfg")) fd.delete("cfg");
  if (!fd.get("shift")) fd.delete("shift");
  setFormBusy(form, true);
  startProgressPolling("layered-result");
  try {
    const res = await fetch("/api/layered", { method: "POST", body: fd });
    const data = await handleJsonResponse(res);
    gallery.push(data);
    renderGallery();
    renderLayeredResult(data);
  } catch (e) {
    showError("layered-error", e.message);
  } finally {
    stopProgressPolling("layered-result");
    setFormBusy(form, false);
    refreshStatus();
  }
});

// --- 背景削除(GPU不使用ユーティリティ。generation_lock/進捗バーとは無関係) ---
const removebgImageInput = document.getElementById("removebg-image-input");
const removebgInputPreview = document.getElementById("removebg-input-preview");

removebgImageInput.addEventListener("change", () => {
  const file = removebgImageInput.files[0];
  if (!file) return;
  removebgInputPreview.src = URL.createObjectURL(file);
  removebgInputPreview.classList.remove("hidden");
});

document.getElementById("removebg-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  showError("removebg-error", null);
  const fd = new FormData(form);
  setFormBusy(form, true);
  try {
    const res = await fetch("/api/remove_bg", { method: "POST", body: fd });
    const data = await handleJsonResponse(res);
    showLatestResult("removebg-result-panel", data);
  } catch (e) {
    showError("removebg-error", e.message);
  } finally {
    setFormBusy(form, false);
  }
});


// ============================================================================
// キャラシート(charsheet)タブ(旧 charsheet/static/app.js 移植。
// API は /api/charsheet/ 接頭辞、DOM ID は cs- 接頭辞に変更)
// ============================================================================
(() => {
  "use strict";

  const CS_VIEW_ORDER = [
    "front", "back", "left", "right",
    "front_left_45", "front_right_45", "back_left_45", "back_right_45",
  ];
  const CS_VIEW_LABELS = {
    front: { ja: "前", en: "Front" },
    back: { ja: "後ろ", en: "Back" },
    left: { ja: "左", en: "Left" },
    right: { ja: "右", en: "Right" },
    front_left_45: { ja: "左前45度", en: "Front-Left 45°" },
    front_right_45: { ja: "右前45度", en: "Front-Right 45°" },
    back_left_45: { ja: "左後ろ45度", en: "Back-Left 45°" },
    back_right_45: { ja: "右後ろ45度", en: "Back-Right 45°" },
  };

  const dropzone = document.getElementById("cs-dropzone");
  const dropzoneText = document.getElementById("cs-dropzone-text");
  const fileInput = document.getElementById("cs-file-input");
  const previewImage = document.getElementById("cs-preview-image");
  const seedInput = document.getElementById("cs-seed-input");
  const generateBtn = document.getElementById("cs-generate-btn");
  const errorMessage = document.getElementById("cs-error-message");
  const statusMessage = document.getElementById("cs-status-message");
  const viewsGrid = document.getElementById("cs-views-grid");
  const sheetPanel = document.getElementById("cs-sheet-panel");
  const sheetImage = document.getElementById("cs-sheet-image");
  const downloadZipBtn = document.getElementById("cs-download-zip-btn");
  const downloadSheetBtn = document.getElementById("cs-download-sheet-btn");
  const removeBgAllBtn = document.getElementById("cs-remove-bg-all-btn");

  // --- 進捗バー(全体 n/8 + 現在方向の step。/api/charsheet/jobs/{id} の
  //     progress/total と、/api/progress の step/total_steps を組み合わせて表示) ---
  const overallProgressWrap = document.getElementById("cs-overall-progress");
  const overallProgressFill = document.getElementById("cs-overall-progress-fill");
  const overallProgressCount = document.getElementById("cs-overall-progress-count");
  const stepProgressWrap = document.getElementById("cs-step-progress");
  const stepProgressPhase = document.getElementById("cs-step-progress-phase");
  const stepProgressCount = document.getElementById("cs-step-progress-count");
  const stepProgressTrack = document.getElementById("cs-step-progress-track");
  const stepProgressFill = document.getElementById("cs-step-progress-fill");
  let csProgressTimer = null;

  function updateOverallProgress(progress, total) {
    if (!total) {
      overallProgressWrap.classList.add("hidden");
      return;
    }
    overallProgressWrap.classList.remove("hidden");
    const pct = Math.max(0, Math.min(100, (progress / total) * 100));
    overallProgressFill.style.width = `${pct}%`;
    overallProgressCount.textContent = `${progress} / ${total} 方向完了`;
  }

  async function pollStepProgress() {
    try {
      const res = await fetch("/api/progress");
      if (!res.ok) return;
      const data = await res.json();
      if (!data.active || data.mode !== "charsheet") {
        stepProgressWrap.classList.add("hidden");
        return;
      }
      stepProgressWrap.classList.remove("hidden");
      const phaseLabel = { loading: "モデルロード中…", generating: "生成中", decoding: "画像デコード中…" }[data.phase] || data.phase;
      stepProgressPhase.textContent = phaseLabel;
      const determinate = data.phase === "generating" && data.total_steps > 0;
      if (determinate) {
        stepProgressTrack.classList.remove("indeterminate");
        const pct = Math.max(0, Math.min(100, (data.step / data.total_steps) * 100));
        stepProgressFill.style.width = `${pct}%`;
        stepProgressCount.textContent = `${data.step} / ${data.total_steps} steps`;
      } else {
        stepProgressTrack.classList.add("indeterminate");
        stepProgressFill.style.width = "";
        stepProgressCount.textContent = "";
      }
    } catch (e) {
      /* ポーリング失敗は無視 */
    }
  }

  function startStepProgressPolling() {
    stopStepProgressPolling();
    pollStepProgress();
    csProgressTimer = setInterval(pollStepProgress, 500);
  }

  function stopStepProgressPolling() {
    if (csProgressTimer) {
      clearInterval(csProgressTimer);
      csProgressTimer = null;
    }
    stepProgressWrap.classList.add("hidden");
  }

  const splitSection = document.getElementById("cs-split-section");
  const splitMessage = document.getElementById("cs-split-message");
  const splitGrid = document.getElementById("cs-split-grid");
  const useWholeLink = document.getElementById("cs-use-whole-link");

  if (!dropzone) return; // タブ未描画時のガード

  let selectedFile = null;
  let pollTimer = null;
  let currentJobId = null;
  let busy = false; // 生成 or refine 実行中

  // シート分解(split)状態
  let inputMode = "file"; // "file" = 画像全体 / "figure" = 分割 crop を入力に使う
  let currentSplitId = null;
  let selectedFigureIndex = null;

  function csShowError(msg) {
    errorMessage.textContent = msg;
    errorMessage.classList.remove("hidden");
  }

  function csClearError() {
    errorMessage.textContent = "";
    errorMessage.classList.add("hidden");
  }

  function csShowStatus(msg) {
    statusMessage.textContent = msg;
    statusMessage.classList.remove("hidden");
  }

  function csClearStatus() {
    statusMessage.textContent = "";
    statusMessage.classList.add("hidden");
  }

  function csOpenLightbox(src) {
    // 既存のグローバル lightbox(T2I等と共通)を再利用する
    lightboxImage.src = src;
    lightbox.classList.remove("hidden");
  }

  function buildViewTiles() {
    viewsGrid.innerHTML = "";
    for (const key of CS_VIEW_ORDER) {
      const label = CS_VIEW_LABELS[key];
      const tile = document.createElement("div");
      tile.className = "view-tile";
      tile.id = `cs-tile-${key}`;

      const imgWrap = document.createElement("div");
      imgWrap.className = "view-tile-image-wrap";

      const spinner = document.createElement("div");
      spinner.className = "spinner";
      spinner.id = `cs-spinner-${key}`;

      const img = document.createElement("img");
      img.id = `cs-img-${key}`;
      img.alt = label.ja;
      img.addEventListener("click", () => {
        if (img.src && img.style.display !== "none") {
          csOpenLightbox(img.src);
        }
      });

      const statusBadge = document.createElement("div");
      statusBadge.className = "view-tile-status";
      statusBadge.id = `cs-status-${key}`;
      statusBadge.textContent = "待機中";

      imgWrap.appendChild(spinner);
      imgWrap.appendChild(img);
      imgWrap.appendChild(statusBadge);

      const labelDiv = document.createElement("div");
      labelDiv.className = "view-tile-label";
      labelDiv.innerHTML = `<div class="ja">${label.ja}</div><div class="en">${label.en}</div>`;

      // --- 修正(refine)UI ---
      const actions = document.createElement("div");
      actions.className = "view-tile-actions";
      actions.id = `cs-actions-${key}`;

      const refineToggleBtn = document.createElement("button");
      refineToggleBtn.className = "tile-btn";
      refineToggleBtn.id = `cs-refine-toggle-${key}`;
      refineToggleBtn.textContent = "修正";
      refineToggleBtn.disabled = true;
      refineToggleBtn.addEventListener("click", () => toggleRefineForm(key));

      const removeBgBtn = document.createElement("button");
      removeBgBtn.className = "tile-btn secondary";
      removeBgBtn.id = `cs-remove-bg-${key}`;
      removeBgBtn.textContent = "背景削除";
      removeBgBtn.disabled = true;
      removeBgBtn.addEventListener("click", () => doRemoveBg(key));

      const undoBtn = document.createElement("button");
      undoBtn.className = "tile-btn secondary hidden";
      undoBtn.id = `cs-undo-${key}`;
      undoBtn.textContent = "元に戻す";
      undoBtn.addEventListener("click", () => doUndo(key));

      const tileDownloadBtn = document.createElement("button");
      tileDownloadBtn.className = "tile-btn secondary";
      tileDownloadBtn.id = `cs-download-${key}`;
      tileDownloadBtn.textContent = "DL";
      tileDownloadBtn.disabled = true;
      tileDownloadBtn.addEventListener("click", () => {
        const viewImg = document.getElementById(`cs-img-${key}`);
        if (viewImg && viewImg.src) downloadFile(viewImg.src, `${key}.png`);
      });

      actions.appendChild(refineToggleBtn);
      actions.appendChild(removeBgBtn);
      actions.appendChild(undoBtn);
      actions.appendChild(tileDownloadBtn);

      const refineForm = document.createElement("div");
      refineForm.className = "refine-form hidden";
      refineForm.id = `cs-refine-form-${key}`;

      const refineInput = document.createElement("input");
      refineInput.type = "text";
      refineInput.id = `cs-refine-input-${key}`;
      refineInput.placeholder = "例: 頭の黒い髪の毛を削除して";
      refineInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") doRefine(key);
      });

      const refineApplyBtn = document.createElement("button");
      refineApplyBtn.className = "tile-btn";
      refineApplyBtn.id = `cs-refine-apply-${key}`;
      refineApplyBtn.textContent = "適用";
      refineApplyBtn.addEventListener("click", () => doRefine(key));

      refineForm.appendChild(refineInput);
      refineForm.appendChild(refineApplyBtn);

      tile.appendChild(imgWrap);
      tile.appendChild(labelDiv);
      tile.appendChild(actions);
      tile.appendChild(refineForm);
      viewsGrid.appendChild(tile);
    }
  }

  function toggleRefineForm(key) {
    const form = document.getElementById(`cs-refine-form-${key}`);
    if (!form) return;
    const willShow = form.classList.contains("hidden");
    for (const k of CS_VIEW_ORDER) {
      const f = document.getElementById(`cs-refine-form-${k}`);
      if (f) f.classList.add("hidden");
    }
    if (willShow) {
      form.classList.remove("hidden");
      const input = document.getElementById(`cs-refine-input-${key}`);
      if (input) input.focus();
    }
  }

  function setRefineButtonsEnabled(enabled) {
    for (const key of CS_VIEW_ORDER) {
      const toggleBtn = document.getElementById(`cs-refine-toggle-${key}`);
      const applyBtn = document.getElementById(`cs-refine-apply-${key}`);
      const undoBtn = document.getElementById(`cs-undo-${key}`);
      const bgBtn = document.getElementById(`cs-remove-bg-${key}`);
      const downloadBtn = document.getElementById(`cs-download-${key}`);
      const img = document.getElementById(`cs-img-${key}`);
      const hasImage = img && img.style.display !== "none" && img.src;
      if (toggleBtn) toggleBtn.disabled = !enabled || !hasImage;
      if (applyBtn) applyBtn.disabled = !enabled;
      if (undoBtn) undoBtn.disabled = !enabled;
      if (bgBtn) bgBtn.disabled = !enabled || !hasImage;
      if (downloadBtn) downloadBtn.disabled = !hasImage;
    }
    if (removeBgAllBtn) removeBgAllBtn.disabled = !enabled;
  }

  function updateViewTile(view) {
    const key = view.key;
    const spinner = document.getElementById(`cs-spinner-${key}`);
    const img = document.getElementById(`cs-img-${key}`);
    const statusBadge = document.getElementById(`cs-status-${key}`);
    const undoBtn = document.getElementById(`cs-undo-${key}`);
    if (!spinner || !img || !statusBadge) return;

    const statusText = {
      queued: "待機中",
      running: "生成中",
      refining: "修正中",
      removing_bg: "背景削除中",
      done: "完了",
      error: "エラー",
    };
    statusBadge.textContent = statusText[view.status] || view.status;

    if (view.status === "done" && view.url) {
      spinner.style.display = "none";
      img.style.display = "block";
      img.style.opacity = "1";
      if (!img.src || img.dataset.reload === "1") {
        img.src = view.url + "?t=" + Date.now();
        img.dataset.reload = "0";
      }
    } else if (view.status === "refining" || view.status === "removing_bg") {
      spinner.style.display = "block";
      img.style.opacity = "0.35";
      img.dataset.reload = "1";
    } else if (view.status === "running") {
      spinner.style.display = "block";
      img.style.display = "none";
      img.dataset.reload = "1";
    } else if (view.status === "error") {
      spinner.style.display = "none";
      img.style.display = "none";
    } else {
      spinner.style.display = "none";
      img.style.display = "none";
    }

    if (undoBtn) {
      if (view.has_prev) {
        undoBtn.classList.remove("hidden");
      } else {
        undoBtn.classList.add("hidden");
      }
    }
  }

  // --- ファイル選択 / D&D ---
  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleFileSelect(fileInput.files[0]);
    }
  });

  function handleFileSelect(file) {
    if (!file.type.startsWith("image/")) {
      csShowError("画像ファイルを選択してください");
      return;
    }
    csClearError();
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      previewImage.classList.remove("hidden");
      dropzoneText.classList.add("hidden");
    };
    reader.readAsDataURL(file);

    resetSplitState();
    runSplit(file);
  }

  // --- シート分解(split) ---
  function resetSplitState() {
    inputMode = "file";
    currentSplitId = null;
    selectedFigureIndex = null;
    splitSection.classList.add("hidden");
    splitGrid.innerHTML = "";
    splitMessage.textContent = "";
  }

  async function runSplit(file) {
    generateBtn.disabled = true;
    csShowStatus("キャラクターを検出中...");
    try {
      const formData = new FormData();
      formData.append("image", file);
      const resp = await fetch("/api/charsheet/split", { method: "POST", body: formData });
      if (!resp.ok) {
        csClearStatus();
        generateBtn.disabled = false;
        return;
      }
      const data = await resp.json();
      csClearStatus();
      if (data.count >= 2) {
        currentSplitId = data.split_id;
        showSplitSelection(data);
        generateBtn.disabled = true;
      } else {
        generateBtn.disabled = false;
      }
    } catch (err) {
      console.error(err);
      csClearStatus();
      generateBtn.disabled = false;
    }
  }

  function showSplitSelection(data) {
    splitMessage.textContent =
      `${data.count}体のキャラクターを検出しました。8方向生成の元にする1体を選んでください(通常は正面向き)。`;
    splitGrid.innerHTML = "";
    for (const fig of data.figures) {
      const cell = document.createElement("div");
      cell.className = "split-cell";
      cell.id = `cs-split-cell-${fig.index}`;

      const img = document.createElement("img");
      img.src = fig.url;
      img.alt = `キャラクター ${fig.index + 1}`;

      const num = document.createElement("div");
      num.className = "split-cell-num";
      num.textContent = String(fig.index + 1);

      cell.appendChild(img);
      cell.appendChild(num);
      cell.addEventListener("click", () => selectFigure(fig.index));
      splitGrid.appendChild(cell);
    }
    splitSection.classList.remove("hidden");
  }

  function selectFigure(index) {
    inputMode = "figure";
    selectedFigureIndex = index;
    for (const cell of splitGrid.children) {
      cell.classList.toggle("selected", cell.id === `cs-split-cell-${index}`);
    }
    generateBtn.disabled = false;
    csClearError();
    csShowStatus(`キャラクター ${index + 1} を選択しました。`);
  }

  useWholeLink.addEventListener("click", (e) => {
    e.preventDefault();
    inputMode = "file";
    selectedFigureIndex = null;
    for (const cell of splitGrid.children) {
      cell.classList.remove("selected");
    }
    generateBtn.disabled = !selectedFile;
    csShowStatus("シート全体をそのまま入力に使います。");
  });

  // --- 生成開始 ---
  generateBtn.addEventListener("click", async () => {
    if (!selectedFile) {
      csShowError("画像を選択してください");
      return;
    }
    csClearError();
    csClearStatus();
    generateBtn.disabled = true;
    busy = true;
    sheetPanel.classList.add("hidden");
    buildViewTiles();

    const formData = new FormData();
    if (inputMode === "figure" && currentSplitId !== null && selectedFigureIndex !== null) {
      formData.append("split_id", currentSplitId);
      formData.append("figure_index", selectedFigureIndex);
    } else {
      formData.append("image", selectedFile);
    }
    const seed = parseInt(seedInput.value, 10) || 0;
    formData.append("seed", seed);

    try {
      const resp = await fetch("/api/charsheet/generate", {
        method: "POST",
        body: formData,
      });
      if (resp.status === 409) {
        const data = await resp.json().catch(() => ({}));
        csShowError(data.detail || "別のジョブが実行中です。しばらく待ってから再試行してください。");
        generateBtn.disabled = false;
        busy = false;
        return;
      }
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        csShowError(data.detail || `生成開始に失敗しました (status ${resp.status})`);
        generateBtn.disabled = false;
        busy = false;
        return;
      }
      const data = await resp.json();
      currentJobId = data.job_id;
      csShowStatus("生成を開始しました。モデルの初回ロードには数分かかる場合があります。");
      startPolling(currentJobId);
    } catch (err) {
      csShowError("通信エラー: " + err.message);
      generateBtn.disabled = false;
      busy = false;
    }
  });

  // --- 修正(refine) ---
  async function doRefine(key) {
    if (!currentJobId || busy) return;
    const input = document.getElementById(`cs-refine-input-${key}`);
    const instruction = (input ? input.value : "").trim();
    if (!instruction) {
      csShowError("修正指示を入力してください");
      return;
    }
    csClearError();
    busy = true;
    setRefineButtonsEnabled(false);
    generateBtn.disabled = true;
    const form = document.getElementById(`cs-refine-form-${key}`);
    if (form) form.classList.add("hidden");

    const seed = parseInt(seedInput.value, 10) || 0;

    try {
      const resp = await fetch(`/api/charsheet/jobs/${currentJobId}/refine`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key, instruction, seed }),
      });
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        csShowError(data.detail || `修正の開始に失敗しました (status ${resp.status})`);
        busy = false;
        setRefineButtonsEnabled(true);
        generateBtn.disabled = !selectedFile;
        return;
      }
      csShowStatus(`「${CS_VIEW_LABELS[key].ja}」を修正中...`);
      startPolling(currentJobId);
    } catch (err) {
      csShowError("通信エラー: " + err.message);
      busy = false;
      setRefineButtonsEnabled(true);
      generateBtn.disabled = !selectedFile;
    }
  }

  // --- 背景削除(remove_bg) ---
  async function doRemoveBg(key) {
    if (!currentJobId || busy) return;
    csClearError();
    busy = true;
    setRefineButtonsEnabled(false);
    generateBtn.disabled = true;

    try {
      const resp = await fetch(`/api/charsheet/jobs/${currentJobId}/remove_bg`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      });
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        csShowError(data.detail || `背景削除の開始に失敗しました (status ${resp.status})`);
        busy = false;
        setRefineButtonsEnabled(true);
        generateBtn.disabled = !selectedFile;
        return;
      }
      csShowStatus(key === "all" ? "全方向の背景を削除中..." : `「${CS_VIEW_LABELS[key].ja}」の背景を削除中...`);
      startPolling(currentJobId);
    } catch (err) {
      csShowError("通信エラー: " + err.message);
      busy = false;
      setRefineButtonsEnabled(true);
      generateBtn.disabled = !selectedFile;
    }
  }

  // --- 元に戻す(undo) ---
  async function doUndo(key) {
    if (!currentJobId || busy) return;
    csClearError();
    busy = true;
    setRefineButtonsEnabled(false);

    try {
      const resp = await fetch(`/api/charsheet/jobs/${currentJobId}/undo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key }),
      });
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        csShowError(data.detail || `復元に失敗しました (status ${resp.status})`);
        return;
      }
      const job = await resp.json();
      const img = document.getElementById(`cs-img-${key}`);
      if (img) img.dataset.reload = "1";
      for (const view of job.views) {
        updateViewTile(view);
      }
      refreshSheet(job);
      csShowStatus(`「${CS_VIEW_LABELS[key].ja}」を入れ替えました(もう一度押すと戻ります)。`);
    } catch (err) {
      csShowError("通信エラー: " + err.message);
    } finally {
      busy = false;
      setRefineButtonsEnabled(true);
    }
  }

  function refreshSheet(job) {
    if (job.sheet_url) {
      sheetImage.src = job.sheet_url + "?t=" + Date.now();
      sheetPanel.classList.remove("hidden");
      downloadZipBtn.dataset.url = job.zip_url;
      downloadSheetBtn.dataset.url = job.sheet_url;
    }
  }

  function startPolling(jobId) {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(() => pollJob(jobId), 1500);
    startStepProgressPolling();
    pollJob(jobId);
  }

  async function pollJob(jobId) {
    try {
      const resp = await fetch(`/api/charsheet/jobs/${jobId}`);
      if (!resp.ok) {
        return;
      }
      const job = await resp.json();

      for (const view of job.views) {
        updateViewTile(view);
      }

      if (job.status === "running" || job.status === "queued") {
        csShowStatus(`生成中... (${job.progress} / ${job.total} 方向完了)`);
        updateOverallProgress(job.progress, job.total);
      } else if (job.status === "refining") {
        csShowStatus("修正を適用中...");
        overallProgressWrap.classList.add("hidden");
      } else if (job.status === "removing_bg") {
        csShowStatus("背景を削除中...");
        overallProgressWrap.classList.add("hidden");
      }

      if (job.status === "done") {
        clearInterval(pollTimer);
        pollTimer = null;
        stopStepProgressPolling();
        overallProgressWrap.classList.add("hidden");
        busy = false;
        generateBtn.disabled = !selectedFile;
        csClearStatus();
        if (job.refine_error) {
          csShowError("修正エラー: " + job.refine_error);
        } else {
          csShowStatus("完了しました。各タイルの「修正」ボタンで個別に修正できます。");
        }
        refreshSheet(job);
        setRefineButtonsEnabled(true);
      } else if (job.status === "error") {
        clearInterval(pollTimer);
        pollTimer = null;
        stopStepProgressPolling();
        overallProgressWrap.classList.add("hidden");
        busy = false;
        generateBtn.disabled = !selectedFile;
        csClearStatus();
        csShowError("生成エラー: " + (job.error || "不明なエラー"));
        setRefineButtonsEnabled(true);
      }
    } catch (err) {
      console.error(err);
    }
  }

  downloadZipBtn.addEventListener("click", () => {
    const url = downloadZipBtn.dataset.url;
    if (url) window.location.href = url;
  });

  removeBgAllBtn.addEventListener("click", () => doRemoveBg("all"));

  downloadSheetBtn.addEventListener("click", () => {
    const url = downloadSheetBtn.dataset.url;
    if (!url) return;
    const a = document.createElement("a");
    a.href = url;
    a.download = "character_sheet.png";
    document.body.appendChild(a);
    a.click();
    a.remove();
  });

  buildViewTiles();
})();

// ============================================================================
// LTX-2.3(動画+音声生成)タブ
// ============================================================================
setupLlmButtons("ltx2-t2v-form", "ltx2-t2v-error");

// サブタブ切替(T2V/I2V等、data-subtab属性で独立管理)。
// 注意: サブタブボタンも .tab-btn クラスを持つため、メインタブ切替ハンドラは
// [data-tab] 属性セレクタで限定してある(冒頭の「タブ切替」コメント参照。クラスだけで
// バインドするとサブタブクリックでメインパネルが消えるバグになる、2026-07-20修正済み)。
// サブタブパネルのクラスは ".ltx2-subtab, .mageflow-subtab"(タブごとに独立)。
// closest(".gen-col-left") でスコープするため、LTX2タブとMage-Flowタブが互いの
// パネルを操作することはない(Mage-Flowタブ追加時に共通セレクタ化、2026-07-22)。
const SUBTAB_PANEL_SELECTOR = ".ltx2-subtab, .mageflow-subtab";
document.querySelectorAll('nav.tabs button[data-subtab]').forEach((btn) => {
  btn.addEventListener("click", () => {
    const parent = btn.closest(".gen-col-left");
    parent.querySelectorAll('nav.tabs button[data-subtab]').forEach((b) => b.classList.remove("active"));
    parent.querySelectorAll(SUBTAB_PANEL_SELECTOR).forEach((p) => p.classList.add("hidden"));
    parent.querySelectorAll(SUBTAB_PANEL_SELECTOR).forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    const target = parent.querySelector(`[data-subtab="${btn.dataset.subtab}"]:not(button)`);
    if (target) {
      target.classList.remove("hidden");
      target.classList.add("active");
    }
  });
});

// --- LTX-2.3 結果表示(video要素、音声込みmp4を再生) ---
function showLtx2Result(panelId, data) {
  const panel = document.getElementById(panelId);
  if (!panel) return;
  panel.innerHTML = "";

  const video = document.createElement("video");
  video.className = "result-panel-image";
  video.src = data.video_url;
  video.controls = true;
  video.autoplay = false;
  video.loop = true;
  panel.appendChild(video);

  const downloadRow = document.createElement("div");
  downloadRow.className = "result-download-row";
  downloadRow.appendChild(makeDownloadButton(data.video_url, null, "動画をダウンロード"));
  // audio_url は動画にmux済みの音声とは別の単体トラック(wav)。has_audioの時のみ存在。
  if (data.has_audio && data.audio_url) {
    downloadRow.appendChild(makeDownloadButton(data.audio_url, null, "音声(wav)をダウンロード"));
  }
  panel.appendChild(downloadRow);

  const meta = document.createElement("div");
  meta.className = "result-panel-meta";
  // 動画の実再生時間(フレーム数÷fps)。「生成時間」(=サーバの処理時間)との混同防止のため
  // 明示的に別行で表示する
  const durationS = data.num_frames && data.fps ? (data.num_frames / data.fps).toFixed(2) : null;
  const rows = [
    ["生成時間(処理)", data.elapsed_s != null ? `${data.elapsed_s.toFixed(2)} 秒` : "-"],
    ["動画長", durationS != null ? `${durationS} 秒` : "-"],
    ["ピークVRAM", data.peak_vram_gb != null ? `${data.peak_vram_gb.toFixed(2)} GB` : "-"],
    ["steps", data.steps],
    ["guidance_scale", data.guidance_scale],
    ["seed", data.seed != null ? data.seed : "ランダム"],
    ["サイズ", data.width ? `${data.width}x${data.height}` : "-"],
    ["フレーム数/fps", `${data.num_frames} / ${data.fps}`],
    ["音声", data.has_audio ? "あり" : "なし"],
    ["offload", data.offload_mode || "-"],
  ];
  if (data.strength != null) {
    rows.splice(4, 0, ["strength", data.strength]);
  }
  if (data.upscale) {
    rows.push(["アップスケール", "2x(latent upsampler)"]);
  }
  rows.forEach(([k, v]) => {
    const row = document.createElement("div");
    row.innerHTML = `<span class="key">${k}</span><span class="val">${v}</span>`;
    meta.appendChild(row);
  });
  panel.appendChild(meta);
}

document.getElementById("ltx2-t2v-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  showError("ltx2-t2v-error", null);
  const fd = new FormData(form);
  const body = {
    prompt: fd.get("prompt"),
    negative_prompt: fd.get("negative_prompt") || "",
    width: parseInt(fd.get("width"), 10),
    height: parseInt(fd.get("height"), 10),
    num_frames: parseInt(fd.get("num_frames"), 10),
    fps: parseInt(fd.get("fps"), 10),
    steps: parseInt(fd.get("steps"), 10),
    guidance_scale: parseFloat(fd.get("guidance_scale")),
    seed: parseInt(fd.get("seed"), 10),
    audio_out: fd.get("audio_out") || "on",
    upscale: fd.get("upscale") ? 1 : 0,
  };
  setFormBusy(form, true);
  startProgressPolling("ltx2-result-panel");
  try {
    const res = await fetch("/api/ltx2/t2v", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await handleJsonResponse(res);
    showLtx2Result("ltx2-result-panel", data);
  } catch (e) {
    showError("ltx2-t2v-error", e.message);
  } finally {
    stopProgressPolling("ltx2-result-panel");
    setFormBusy(form, false);
    refreshStatus();
  }
});

document.getElementById("ltx2-i2v-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  showError("ltx2-i2v-error", null);
  const fd = new FormData(form);
  setFormBusy(form, true);
  startProgressPolling("ltx2-result-panel");
  try {
    const res = await fetch("/api/ltx2/i2v", { method: "POST", body: fd });
    const data = await handleJsonResponse(res);
    showLtx2Result("ltx2-result-panel", data);
  } catch (e) {
    showError("ltx2-i2v-error", e.message);
  } finally {
    stopProgressPolling("ltx2-result-panel");
    setFormBusy(form, false);
    refreshStatus();
  }
});

document.getElementById("ltx2-flf-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  showError("ltx2-flf-error", null);
  const fd = new FormData(form);
  setFormBusy(form, true);
  startProgressPolling("ltx2-result-panel");
  try {
    const res = await fetch("/api/ltx2/flf", { method: "POST", body: fd });
    const data = await handleJsonResponse(res);
    showLtx2Result("ltx2-result-panel", data);
  } catch (e) {
    showError("ltx2-flf-error", e.message);
  } finally {
    stopProgressPolling("ltx2-result-panel");
    setFormBusy(form, false);
    refreshStatus();
  }
});

document.getElementById("ltx2-keyframes-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  showError("ltx2-keyframes-error", null);
  const fd = new FormData(form);
  setFormBusy(form, true);
  startProgressPolling("ltx2-result-panel");
  try {
    const res = await fetch("/api/ltx2/keyframes", { method: "POST", body: fd });
    const data = await handleJsonResponse(res);
    showLtx2Result("ltx2-result-panel", data);
  } catch (e) {
    showError("ltx2-keyframes-error", e.message);
  } finally {
    stopProgressPolling("ltx2-result-panel");
    setFormBusy(form, false);
    refreshStatus();
  }
});

document.getElementById("ltx2-v2a-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  showError("ltx2-v2a-error", null);
  const fd = new FormData(form);
  setFormBusy(form, true);
  startProgressPolling("ltx2-result-panel");
  try {
    const res = await fetch("/api/ltx2/v2a", { method: "POST", body: fd });
    const data = await handleJsonResponse(res);
    showLtx2Result("ltx2-result-panel", data);
  } catch (e) {
    showError("ltx2-v2a-error", e.message);
  } finally {
    stopProgressPolling("ltx2-result-panel");
    setFormBusy(form, false);
    refreshStatus();
  }
});

document.getElementById("ltx2-iclora-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  showError("ltx2-iclora-error", null);
  const fd = new FormData(form);
  setFormBusy(form, true);
  startProgressPolling("ltx2-result-panel");
  try {
    const res = await fetch("/api/ltx2/iclora", { method: "POST", body: fd });
    const data = await handleJsonResponse(res);
    showLtx2Result("ltx2-result-panel", data);
  } catch (e) {
    showError("ltx2-iclora-error", e.message);
  } finally {
    stopProgressPolling("ltx2-result-panel");
    setFormBusy(form, false);
    refreshStatus();
  }
});

// ============================================================================
// Mage-Flow タブ(別プロセスのラッパーサービスへのプロキシ /api/mageflow/*)
// 2026-07-22追加。フォーム値の組み立ては他タブと同じ流儀(T2I=JSON、Edit=multipart)。
// exclusive チェックボックス(既定ON)は「他の生成と同時実行しない」= 本体の
// generation_lock を取ってから転送する(取得失敗は409。詳細は app.py のコメント参照)。
// ============================================================================
setupLlmButtons("mageflow-t2i-form", "mageflow-t2i-error");
setupLlmButtons("mageflow-edit-form", "mageflow-edit-error");

document.getElementById("mageflow-t2i-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  showError("mageflow-t2i-error", null);
  const fd = new FormData(form);
  const body = {
    prompt: fd.get("prompt"),
    negative_prompt: fd.get("negative_prompt") || null,
    width: parseInt(fd.get("width"), 10),
    height: parseInt(fd.get("height"), 10),
    steps: fd.get("steps") ? parseInt(fd.get("steps"), 10) : null,
    cfg: fd.get("cfg") ? parseFloat(fd.get("cfg")) : null,
    seed: parseInt(fd.get("seed"), 10),
    model: fd.get("model") || "rl",
    exclusive: fd.get("exclusive") === "on",
  };
  setFormBusy(form, true);
  startProgressPolling("mageflow-result-panel");
  try {
    const res = await fetch("/api/mageflow/t2i", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await handleJsonResponse(res);
    gallery.push(data);
    renderGallery();
    showLatestResult("mageflow-result-panel", data);
  } catch (e) {
    showError("mageflow-t2i-error", e.message);
  } finally {
    stopProgressPolling("mageflow-result-panel");
    setFormBusy(form, false);
    refreshStatus();
  }
});

document.getElementById("mageflow-edit-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const form = ev.target;
  showError("mageflow-edit-error", null);
  const imageInput = document.getElementById("mageflow-edit-image-input");
  if (!imageInput.files || imageInput.files.length === 0) {
    showError("mageflow-edit-error", "参照画像を1枚以上選択してください");
    return;
  }
  if (imageInput.files.length > 3) {
    showError("mageflow-edit-error", "参照画像は最大3枚までです");
    return;
  }
  const fd = new FormData();
  for (const file of imageInput.files) fd.append("image", file);
  fd.set("prompt", form.querySelector('[name="prompt"]').value);
  fd.set("negative_prompt", form.querySelector('[name="negative_prompt"]').value || "");
  fd.set("model", form.querySelector('[name="model"]').value || "rl");
  fd.set("max_size", form.querySelector('[name="max_size"]').value || "1024");
  fd.set("seed", form.querySelector('[name="seed"]').value || "-1");
  // 空欄のオプション値は送らない(ラッパー側でバリアント既定値/自動解像度になる)
  const stepsVal = form.querySelector('[name="steps"]').value;
  if (stepsVal) fd.set("steps", stepsVal);
  const cfgVal = form.querySelector('[name="cfg"]').value;
  if (cfgVal) fd.set("cfg", cfgVal);
  const widthVal = form.querySelector('[name="width"]').value;
  if (widthVal) fd.set("width", widthVal);
  const heightVal = form.querySelector('[name="height"]').value;
  if (heightVal) fd.set("height", heightVal);
  fd.set("exclusive", form.querySelector('[name="exclusive"]').checked ? "true" : "false");
  setFormBusy(form, true);
  startProgressPolling("mageflow-result-panel");
  try {
    const res = await fetch("/api/mageflow/edit", { method: "POST", body: fd });
    const data = await handleJsonResponse(res);
    gallery.push(data);
    renderGallery();
    showLatestResult("mageflow-result-panel", data);
  } catch (e) {
    showError("mageflow-edit-error", e.message);
  } finally {
    stopProgressPolling("mageflow-result-panel");
    setFormBusy(form, false);
    refreshStatus();
  }
});

// ============================================================================
// シーンアングル(scene_angles)タブ(2026-07-24追加)。
// 1枚のシーン画像 → カメラ指示プロンプト8種で8アングル生成(ジョブ式)。
// charsheetタブの簡易版(split/refine/undo/シート合成なし)。
// ============================================================================
(() => {
  const form = document.getElementById("sa-form");
  if (!form) return;

  const fileInput = document.getElementById("sa-file-input");
  const seedInput = document.getElementById("sa-seed-input");
  const generateBtn = document.getElementById("sa-generate-btn");
  const errorEl = document.getElementById("sa-error-message");
  const statusEl = document.getElementById("sa-status-message");
  const progressWrap = document.getElementById("sa-progress");
  const progressFill = document.getElementById("sa-progress-fill");
  const progressCount = document.getElementById("sa-progress-count");
  const viewsGrid = document.getElementById("sa-views-grid");
  const checkboxesEl = document.getElementById("sa-angle-checkboxes");

  let pollTimer = null;
  let busy = false;

  const showError = (msg) => {
    errorEl.textContent = msg;
    errorEl.classList.toggle("hidden", !msg);
  };
  const showStatus = (msg) => {
    statusEl.textContent = msg;
    statusEl.classList.toggle("hidden", !msg);
  };

  // アングル一覧をサーバから取得してチェックボックスを構築(既定: 全チェック)
  fetch("/api/scene_angles/angles")
    .then((r) => r.json())
    .then((data) => {
      checkboxesEl.innerHTML = "";
      for (const a of data.angles) {
        const label = document.createElement("label");
        label.style.fontWeight = "normal";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = true;
        cb.value = a.key;
        cb.className = "sa-angle-cb";
        label.appendChild(cb);
        label.appendChild(document.createTextNode(` ${a.label_ja}`));
        checkboxesEl.appendChild(label);
      }
    })
    .catch(() => {
      checkboxesEl.innerHTML = "<p class='form-error'>アングル一覧の取得に失敗しました</p>";
    });

  function renderViews(views) {
    viewsGrid.innerHTML = "";
    const statusText = {
      queued: "待機中", running: "生成中", done: "完了", error: "エラー",
    };
    for (const v of views) {
      const tile = document.createElement("div");
      tile.className = "view-tile";
      const wrap = document.createElement("div");
      wrap.className = "view-tile-image-wrap";
      if (v.status === "done" && v.url) {
        const img = document.createElement("img");
        img.src = v.url + "?t=" + Date.now();
        img.alt = v.label_ja;
        wrap.appendChild(img);
      } else if (v.status === "running") {
        const spinner = document.createElement("div");
        spinner.className = "spinner";
        wrap.appendChild(spinner);
      }
      tile.appendChild(wrap);
      const label = document.createElement("div");
      label.className = "view-tile-label";
      label.innerHTML = `<span class="ja">${v.label_ja}</span> <span class="en">${v.label_en}</span>`;
      tile.appendChild(label);
      const st = document.createElement("div");
      st.className = "view-tile-status";
      st.textContent = statusText[v.status] || v.status;
      tile.appendChild(st);
      viewsGrid.appendChild(tile);
    }
  }

  async function pollJob(jobId) {
    try {
      const resp = await fetch(`/api/scene_angles/jobs/${jobId}`);
      if (!resp.ok) return;
      const job = await resp.json();
      renderViews(job.views);
      if (job.status === "running" || job.status === "queued") {
        showStatus(`生成中... (${job.progress} / ${job.total} アングル完了)`);
        progressWrap.classList.remove("hidden");
        progressCount.textContent = `${job.progress} / ${job.total}`;
        progressFill.style.width = `${(job.progress / Math.max(1, job.total)) * 100}%`;
        return;
      }
      // done / error で終了
      clearInterval(pollTimer);
      pollTimer = null;
      busy = false;
      generateBtn.disabled = false;
      progressWrap.classList.add("hidden");
      if (job.status === "done") {
        showStatus("完了しました。");
      } else {
        showStatus("");
        showError("生成エラー: " + (job.error || "不明なエラー"));
      }
    } catch (_) {
      /* 一時的な失敗は次のポーリングに任せる */
    }
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (busy) return;
    showError("");
    if (!fileInput.files || fileInput.files.length === 0) {
      showError("入力画像を選択してください");
      return;
    }
    const selected = Array.from(document.querySelectorAll(".sa-angle-cb:checked")).map((c) => c.value);
    if (selected.length === 0) {
      showError("アングルを1つ以上選択してください");
      return;
    }
    const fd = new FormData();
    fd.append("image", fileInput.files[0]);
    fd.append("seed", seedInput.value || "0");
    // 8種全選択時は angles を省略(サーバ既定=全部)
    if (selected.length < 8) fd.append("angles", selected.join(","));

    busy = true;
    generateBtn.disabled = true;
    showStatus("ジョブを投入中...");
    try {
      const resp = await fetch("/api/scene_angles/generate", { method: "POST", body: fd });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }
      const data = await resp.json();
      pollTimer = setInterval(() => pollJob(data.job_id), 1500);
      pollJob(data.job_id);
    } catch (err) {
      busy = false;
      generateBtn.disabled = false;
      showStatus("");
      showError(err.message);
    }
  });
})();
