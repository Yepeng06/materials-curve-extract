/* 材料曲线智能提取 — 前端逻辑（单图 + 批量） */
"use strict";

const $ = (id) => document.getElementById(id);
const dz = $("dropzone");
const fileInput = $("file-input");

let fileList = [];          // 待提取文件 [{file, name, size, status}]
let busy = false;           // 提取任务进行中

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
  $("btn-label").textContent = `③ 批量提取（${n} 张）`;
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

async function extractAll() {
  if (busy || fileList.length === 0) return;
  busy = true;
  const btn = $("btn-extract");
  btn.classList.add("loading");
  $("btn-clear-files").disabled = true;

  for (let i = 0; i < fileList.length; i++) {
    const it = fileList[i];
    if (it.status === "done") continue;   // 重试时跳过已完成
    setFileStatus(it.name, "processing");
    $("btn-label").textContent = `提取中 ${i + 1}/${fileList.length}：${it.name}`;
    flash(`正在提取 ${i + 1}/${fileList.length}：${it.name}（首次加载模型约需数秒）…`);

    const fd = new FormData();
    fd.append("file", it.file);
    fd.append("ocr", $("ocr-select").value);
    fd.append("segmenter", $("segmenter-select").value);
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
  $("btn-label").textContent = `③ 批量提取（${fileList.length} 张）`;
  $("btn-clear-files").disabled = fileList.length === 0;
  $("result-card").hidden = false;
  flash("批量提取完成", false);
  $("result-card").scrollIntoView({ behavior: "smooth", block: "start" });
}

$("btn-extract").addEventListener("click", extractAll);

/* ================= 结果渲染（卡片列表） ================= */
function renderResultCard(data) {
  $("result-card").hidden = false;
  const card = document.createElement("div");
  card.className = "result-card";
  const curves = data.curves.map((c) => `曲线${c.index + 1}：${c.n_points} 点`).join(" · ");
  const qualityBadge = data.quality === "B"
    ? `<span class="badge b-warn" title="数据已产出但需人工确认">需确认</span>`
    : `<span class="badge b-done">完成</span>`;
  card.innerHTML = `
    <div class="rc-left">
      <img class="rc-overlay" src="${data.downloads.overlay}?t=${Date.now()}" alt="overlay">
      <div class="rc-name" title="${data.filename}">${data.filename}</div>
    </div>
    <div class="rc-right">
      <table class="summary">
        <tr><td>状态</td><td>${qualityBadge}</td></tr>
        <tr><td>图像尺寸</td><td>${data.image_size[0]} × ${data.image_size[1]} px</td></tr>
        <tr><td>曲线</td><td>${data.n_curves} 条（${curves}）</td></tr>
        <tr><td>坐标类型</td><td>x: ${data.x_axis} · y: ${data.y_axis}</td></tr>
        ${data.titles && (data.titles.title || data.titles.x_label || data.titles.y_label) ? `
        <tr><td>标题/轴标题</td><td>
          ${data.titles.title ? `标题: ${data.titles.title.text} ` : ''}
          ${data.titles.x_label ? `X: ${data.titles.x_label.text} ` : ''}
          ${data.titles.y_label ? `Y: ${data.titles.y_label.text}` : ''}
        </td></tr>` : ''}
        <tr><td>提取耗时</td><td>${data.elapsed_s}s</td></tr>
        <tr><td>OCR 后端</td><td>${data.ocr === "paddle" ? "真实识别（PaddleOCR）" : "标准答案（stub）"}</td></tr>
        <tr><td>分割模型</td><td>${data.segmenter === "unet" ? "深度学习 U-Net" : data.segmenter === "multi_unet" ? "多曲线 U-Net" : "经典 CV"}</td></tr>
      </table>
      <div class="downloads">
        <a class="btn small" href="${data.downloads.csv}" download="${data.filename.replace(/\.[^.]+$/, "")}_curves.csv">⬇ CSV</a>
        <a class="btn small" href="${data.downloads.json}" download="${data.filename.replace(/\.[^.]+$/, "")}_result.json">⬇ JSON</a>
        <a class="btn small" href="${data.downloads.overlay}" download="${data.filename.replace(/\.[^.]+$/, "")}_overlay.png">⬇ 叠加图</a>
      </div>
      ${data.reject_detail ? `<div class="rc-warn">⚠ ${data.reject_detail.replace(/</g, "&lt;")}</div>` : ""}
      ${data.warnings.length ? `<div class="rc-warn">⚠ ${data.warnings.join("；")}</div>` : ""}
    </div>`;
  $("results").prepend(card);
}

function renderResultError(name, msg) {
  $("result-card").hidden = false;
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
});

/* ================= 示例图 ================= */
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

  const fd = new FormData();
  fd.append("group", group);
  fd.append("name", name);
  fd.append("ocr", $("ocr-select").value);
  fd.append("segmenter", $("segmenter-select").value);
  try {
    const resp = await fetch("/api/extract_example", { method: "POST", body: fd });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.detail || `请求失败（HTTP ${resp.status}）`);
    renderResultCard(data);
    flash(`提取完成：${data.filename}，${data.elapsed_s}s`, false);
  } catch (err) {
    renderResultError(name, err.message);
    flash(`提取失败：${err.message}`, true);
  } finally {
    busy = false;
    btn.classList.remove("loading");
    $("btn-label").textContent = `③ 批量提取（${fileList.length} 张）`;
    $("result-card").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

async function loadExamples() {
  try {
    const resp = await fetch("/api/examples");
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
loadExamples();
