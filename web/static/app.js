/* 材料曲线工作台 — 提取前端（上传/批量/结果/内嵌校正入口） */
"use strict";

const $ = (id) => document.getElementById(id);
const dz = $("dropzone");
const fileInput = $("file-input");

let fileList = [];          // 待提取文件 [{file, name, size, status}]
let busy = false;           // 提取任务进行中

/* ================= 导航（提取 / 合成） ================= */
(function initNav() {
  const nav = $("main-nav");
  if (!nav) return;
  const extract = $("view-extract");
  const synth = $("view-synth");
  nav.addEventListener("click", (e) => {
    const btn = e.target.closest(".nav-item");
    if (!btn) return;
    const view = btn.dataset.view;
    nav.querySelectorAll(".nav-item").forEach((t) => t.classList.toggle("active", t === btn));
    if (extract) extract.classList.toggle("hidden", view !== "extract");
    if (synth) synth.classList.toggle("hidden", view !== "synth");
    if (view === "synth") location.hash = "synth";
    else history.replaceState(null, "", location.pathname);
  });
  if (location.hash === "#synth") {
    const btn = nav.querySelector('.nav-item[data-view="synth"]');
    if (btn) btn.click();
  }
})();

/* ================= 上传（多文件） ================= */
function addFiles(files) {
  let added = 0;
  for (const f of files) {
    if (!/\.(png|jpe?g)$/i.test(f.name)) {
      flash(`跳过非图片文件：${f.name}（仅支持 PNG/JPG）`, true);
      continue;
    }
    if (fileList.some((it) => it.name === f.name)) {
      flash(`跳过重复文件：${f.name}`, true);
      continue;
    }
    fileList.push({ file: f, name: f.name, size: f.size, status: "pending" });
    added++;
  }
  if (added) renderFileList();
}

function removeFile(idx) {
  fileList.splice(idx, 1);
  renderFileList();
}

function renderFileList() {
  const ul = $("file-list");
  ul.hidden = fileList.length === 0;
  ul.innerHTML = fileList.map((it, i) => `
    <li class="file-item" data-idx="${i}">
      <img class="file-thumb" src="${URL.createObjectURL(it.file)}" alt="">
      <span class="file-name" title="${it.name}">${it.name}</span>
      <span class="file-size">${(it.size / 1024).toFixed(0)} KB</span>
      <span class="badge b-${it.status}">${badgeText(it.status)}</span>
      <button class="file-remove" title="移除">✕</button>
    </li>`).join("");
  ul.querySelectorAll(".file-remove").forEach((btn) =>
    btn.addEventListener("click", () => removeFile(+btn.closest("li").dataset.idx)));
  const n = fileList.length;
  $("btn-extract").disabled = n === 0 || busy;
  $("btn-clear-files").disabled = n === 0;
  $("btn-label").textContent = `批量提取（${n} 张）`;
}

function badgeText(s) {
  return { pending: "待提取", processing: "提取中…", done: "完成", fail: "失败" }[s] || s;
}

function setFileStatus(name, status) {
  const it = fileList.find((x) => x.name === name);
  if (it) {
    it.status = status;
    renderFileList();
  }
}

dz.addEventListener("click", () => fileInput.click());
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
dz.addEventListener("drop", (e) => {
  e.preventDefault();
  dz.classList.remove("dragover");
  if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
});
fileInput.addEventListener("change", () => { addFiles(fileInput.files); fileInput.value = ""; });
$("btn-clear-files").addEventListener("click", () => { fileList = []; renderFileList(); });

function flash(msg, isErr) {
  const el = $("status");
  el.textContent = msg;
  el.className = "status " + (isErr ? "err" : "ok");
}

/* ================= 提取（单图请求，批量串行） ================= */
function errorMessage(detail, status) {
  if (detail && typeof detail === "object") {
    let msg = detail.message || `请求失败（HTTP ${status}）`;
    if (detail.reject_code) msg += `［${detail.reject_code}］`;
    if (detail.reject_detail) msg += ` — ${detail.reject_detail}`;
    return msg;
  }
  return typeof detail === "string" ? detail : `请求失败（HTTP ${status}）`;
}

async function postExtract(fd) {
  const resp = await fetch("/api/extract", { method: "POST", body: fd });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(errorMessage(data.detail, resp.status));
  return data;
}

function currentParams() {
  return {
    ocr: $("ocr-select").value,
    segmenter: $("segmenter-select").value,
    points: parseInt($("points-select").value, 10) || 0,
  };
}

async function extractAll() {
  if (busy || fileList.length === 0) return;
  busy = true;
  const btn = $("btn-extract");
  btn.classList.add("loading");
  $("btn-clear-files").disabled = true;
  const params = currentParams();

  for (let i = 0; i < fileList.length; i++) {
    const it = fileList[i];
    if (it.status === "done") continue;   // 重试时跳过已完成
    setFileStatus(it.name, "processing");
    $("btn-label").textContent = `提取中 ${i + 1}/${fileList.length}：${it.name}`;
    flash(`正在提取 ${i + 1}/${fileList.length}：${it.name}（首次加载模型约需数秒）…`);

    const fd = new FormData();
    fd.append("file", it.file);
    fd.append("ocr", params.ocr);
    fd.append("segmenter", params.segmenter);
    fd.append("points", params.points);
    try {
      const data = await postExtract(fd);
      renderResultCard(data);
      setFileStatus(it.name, "done");
    } catch (err) {
      renderResultError(it.name, err.message);
      setFileStatus(it.name, "fail");
    }
  }

  busy = false;
  btn.classList.remove("loading");
  $("btn-label").textContent = `批量提取（${fileList.length} 张）`;
  $("btn-clear-files").disabled = fileList.length === 0;
  flash("批量提取完成", false);
  $("result-card").scrollIntoView({ behavior: "smooth", block: "start" });
}

$("btn-extract").addEventListener("click", extractAll);

/* ================= 结果渲染（卡片列表） ================= */
function segmenterLabel(data) {
  const seg = data.segmenter;
  const auto = data.auto_segmenter;
  if (seg === "auto") {
    const name = { multi: "多曲线 U-Net", single: "单曲线 U-Net", cv: "经典 CV" }[auto] || auto || "—";
    return `自动判定 → ${name}`;
  }
  return { unet: "单曲线 U-Net", multi_unet: "多曲线 U-Net", cv: "经典 CV" }[seg] || seg;
}

function pointsLabel(data) {
  if (!data.points_param || data.points_param <= 0) return "原生密度";
  const total = (data.curves || []).reduce((s, c) => s + (c.n_points_exported || 0), 0);
  return `${data.points_param} 点/条（已重采样，共 ${total} 点）`;
}

function renderResultCard(data) {
  $("result-card").hidden = false;
  $("result-empty").classList.add("hidden");
  const card = document.createElement("div");
  card.className = "result-card";
  card.dataset.tid = data.task_id;
  const curves = data.curves.map((c) => `曲线${c.index + 1}：${c.n_points} 点`).join(" · ");
  const qualityBadge = data.quality === "B"
    ? `<span class="badge b-warn" title="数据已产出但需人工确认">需确认</span>`
    : data.status === "corrected"
      ? `<span class="badge b-done">已校正</span>`
      : `<span class="badge b-done">完成</span>`;
  const autoBadge = data.segmenter === "auto"
    ? `<span class="badge b-auto">自动识别 ${data.n_curves} 条曲线</span>` : "";
  card.innerHTML = `
    <div class="rc-left">
      <div class="rc-image-row">
        <div class="rc-image-col">
          <div class="rc-img-label">原图</div>
          <img class="rc-overlay" src="${data.downloads.image}?t=${Date.now()}" alt="原图" style="cursor:pointer" onclick="openLightbox('${data.downloads.image}?t=${Date.now()}')">
        </div>
        <div class="rc-image-col">
          <div class="rc-img-label">重绘图（提取点 → 再绘曲线）</div>
          <img class="rc-overlay" src="${data.downloads.redraw}?t=${Date.now()}" alt="重绘" onerror="this.closest('.rc-image-col').style.display='none'" style="cursor:pointer" onclick="openLightbox('${data.downloads.redraw}?t=${Date.now()}')">
        </div>
        <div class="rc-image-col">
          <div class="rc-img-label">叠加图（原图 + 提取曲线）</div>
          <img class="rc-overlay" src="${data.downloads.overlay}?t=${Date.now()}" alt="叠加" style="cursor:pointer" onclick="openLightbox('${data.downloads.overlay}?t=${Date.now()}')">
        </div>
      </div>
      <div class="rc-name" title="${data.filename}">${data.filename}</div>
      <!-- 校正编辑器挂载点 -->
      <div class="corr-wrap" data-corr></div>
    </div>
    <div class="rc-right">
      <table class="summary">
        <tr><td>状态</td><td>${qualityBadge} ${autoBadge}</td></tr>
        <tr><td>图像尺寸</td><td>${data.image_size[0]} × ${data.image_size[1]} px</td></tr>
        <tr><td>曲线</td><td>${data.n_curves} 条（${curves}）</td></tr>
        <tr><td>坐标类型</td><td>x: ${data.x_axis} · y: ${data.y_axis}</td></tr>
        <tr><td>导出点数</td><td>${pointsLabel(data)}</td></tr>
        <tr><td>提取耗时</td><td>${data.elapsed_s}s</td></tr>
        <tr><td>OCR 后端</td><td>${data.ocr === "paddle" ? "真实识别（PaddleOCR v5）" : "标准答案（stub）"}</td></tr>
        <tr><td>分割模型</td><td>${segmenterLabel(data)}</td></tr>
      </table>
      <div class="downloads">
        <a class="btn small" href="${data.downloads.csv}" download="${data.filename.replace(/\.[^.]+$/, "")}_curves.csv">⬇ CSV</a>
        <a class="btn small" href="${data.downloads.json}" download="${data.filename.replace(/\.[^.]+$/, "")}_result.json">⬇ JSON</a>
        <a class="btn small" href="${data.downloads.image}" download="${data.filename.replace(/\.[^.]+$/, "")}_original.png">⬇ 原图</a>
        <a class="btn small" href="${data.downloads.overlay}" download="${data.filename.replace(/\.[^.]+$/, "")}_overlay.png">⬇ 叠加图</a>
        <a class="btn small" href="${data.downloads.redraw}" download="${data.filename.replace(/\.[^.]+$/, "")}_redraw.png">⬇ 重绘图</a>
      </div>
      <button class="btn primary" data-corr-btn title="在原图上查看并人工校正提取点（拖拽/加点/删点）">✏️ 人工校正</button>
      ${data.reject_detail ? `<div class="rc-warn">⚠ ${data.reject_detail.replace(/</g, "&lt;")}</div>` : ""}
      ${(data.warnings || []).length ? `<div class="rc-warn">⚠ ${data.warnings.join("；")}</div>` : ""}
    </div>`;
  $("results").prepend(card);
  // 绑定校正按钮
  const corrBtn = card.querySelector("[data-corr-btn]");
  corrBtn.addEventListener("click", () => {
    if (typeof window.Corrector === "undefined") {
      flash("校正编辑器未加载（缺少 Konva）", true);
      return;
    }
    window.Corrector.toggle(card, data);
  });
}

function renderResultError(name, msg) {
  $("result-card").hidden = false;
  $("result-empty").classList.add("hidden");
  const card = document.createElement("div");
  card.className = "result-card rc-error";
  card.innerHTML = `
    <div class="rc-left"><div class="rc-name">${name}</div></div>
    <div class="rc-right"><span class="badge b-fail">失败</span>
      <div class="rc-warn">${msg.replace(/</g, "&lt;")}</div></div>`;
  $("results").prepend(card);
}

$("btn-clear-results").addEventListener("click", () => {
  $("results").innerHTML = "";
  $("result-card").hidden = true;
  $("result-empty").classList.remove("hidden");
});

/* ================= 示例图（可刷新） ================= */
async function loadExamples(refresh) {
  try {
    const resp = await fetch("/api/examples" + (refresh ? "?refresh=1" : ""));
    const data = await resp.json();
    const box = $("examples");
    box.innerHTML = "";
    for (const it of data.items) {
      const div = document.createElement("div");
      div.className = "example-item";
      div.title = it.name;
      div.innerHTML = `<img src="${it.url}" alt="${it.name}" loading="lazy">
        <div class="ex-name">${it.name}</div><div class="ex-group">${it.group_label || it.group}</div>`;
      div.addEventListener("click", () => extractExample(it.group, it.name));
      box.appendChild(div);
    }
  } catch (e) {
    $("examples").innerHTML = `<p class="hint">示例图加载失败：${e.message}</p>`;
  }
}
$("btn-refresh-examples").addEventListener("click", () => loadExamples(true));
loadExamples(false);

async function extractExample(group, name) {
  if (busy) {
    flash("有提取任务进行中，请稍候…", true);
    return;
  }
  busy = true;
  const btn = $("btn-extract");
  btn.classList.add("loading");
  $("btn-label").textContent = `提取中：${name}`;
  flash(`正在提取示例图 ${name}…`);
  const params = currentParams();

  const fd = new FormData();
  fd.append("group", group);
  fd.append("name", name);
  fd.append("ocr", params.ocr);
  fd.append("segmenter", params.segmenter);
  fd.append("points", params.points);
  try {
    const resp = await fetch("/api/extract_example", { method: "POST", body: fd });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(errorMessage(data.detail, resp.status));
    renderResultCard(data);
    flash(`提取完成：${data.filename}，${data.elapsed_s}s`, false);
  } catch (err) {
    renderResultError(name, err.message);
    flash(`提取失败：${err.message}`, true);
  } finally {
    busy = false;
    btn.classList.remove("loading");
    $("btn-label").textContent = `批量提取（${fileList.length} 张）`;
    $("result-card").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

/* ================= Lightbox 放大查看 ================= */
window.openLightbox = function (url) {
  var lb = document.getElementById("lightbox");
  var img = document.getElementById("lightbox-img");
  if (lb && img) { img.src = url; lb.classList.remove("hidden"); }
};
window.closeLightbox = function () {
  var lb = document.getElementById("lightbox");
  if (lb) lb.classList.add("hidden");
};
