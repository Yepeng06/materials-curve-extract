/* 材料曲线智能提取 — 前端逻辑 */
"use strict";

const $ = (id) => document.getElementById(id);
const dz = $("dropzone");
const fileInput = $("file-input");
let currentFile = null;
let busy = false;

/* ---------- 上传交互 ---------- */
function setFile(file) {
  if (!file) return;
  if (!/\.(png|jpe?g)$/i.test(file.name)) {
    flash(`仅支持 PNG / JPG 图片（${file.name}）`, true);
    return;
  }
  currentFile = file;
  $("preview-img").src = URL.createObjectURL(file);
  $("preview-name").textContent = `${file.name}（${(file.size / 1024).toFixed(0)} KB）`;
  $("preview-row").hidden = false;
  $("btn-extract").disabled = false;
  $("status").textContent = "";
}

dz.addEventListener("click", () => fileInput.click());
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
dz.addEventListener("drop", (e) => {
  e.preventDefault();
  dz.classList.remove("dragover");
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
$("btn-change").addEventListener("click", () => { currentFile = null; $("preview-row").hidden = true; fileInput.value = ""; $("btn-extract").disabled = true; });

function flash(msg, isErr) {
  const el = $("status");
  el.textContent = msg;
  el.className = "status " + (isErr ? "err" : "ok");
}

/* ---------- 提取 ---------- */
$("btn-extract").addEventListener("click", async () => {
  if (!currentFile || busy) return;
  busy = true;
  const btn = $("btn-extract");
  btn.classList.add("loading");
  $("btn-label").textContent = "提取中…";
  flash("正在提取（首次加载模型约需数秒，请稍候）…");

  const fd = new FormData();
  fd.append("file", currentFile);
  fd.append("ocr", $("ocr-select").value);
  fd.append("segmenter", $("segmenter-select").value);

  try {
    const resp = await fetch("/api/extract", { method: "POST", body: fd });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.detail || `请求失败（HTTP ${resp.status}）`);
    showResult(data);
    flash(`提取完成：${data.n_curves} 条曲线，${data.elapsed_s}s`, false);
  } catch (err) {
    flash(`提取失败：${err.message}`, true);
  } finally {
    busy = false;
    btn.classList.remove("loading");
    $("btn-label").textContent = "③ 点击提取";
  }
});

/* ---------- 结果展示 ---------- */
function showResult(data) {
  $("result-card").hidden = false;
  $("overlay-img").src = data.downloads.overlay + "?t=" + Date.now();

  const rows = [
    ["文件名", data.filename],
    ["图像尺寸", `${data.image_size[0]} × ${data.image_size[1]} px`],
    ["曲线数量", data.n_curves],
    ["坐标类型", `x: ${data.x_axis} · y: ${data.y_axis}`],
    ["提取耗时", `${data.elapsed_s}s`],
    ["OCR 后端", data.ocr === "paddle" ? "真实识别（PaddleOCR）" : "标准答案（stub）"],
    ["分割模型", data.segmenter === "unet" ? "深度学习 U-Net" : "经典 CV"],
  ];
  data.curves.forEach((c, i) => {
    rows.push([`曲线 ${i + 1} 点数`, c.n_points]);
  });
  const tb = $("summary-table");
  tb.innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");

  $("warnings").textContent = data.warnings.length
    ? "⚠ 警告：" + data.warnings.join("；") : "";

  $("dl-csv").href = data.downloads.csv;
  $("dl-json").href = data.downloads.json;
  $("dl-overlay").href = data.downloads.overlay;
  $("dl-overlay").setAttribute("download", data.filename.replace(/\.[^.]+$/, "") + "_overlay.png");
  $("dl-csv").setAttribute("download", data.filename.replace(/\.[^.]+$/, "") + "_curves.csv");
  $("dl-json").setAttribute("download", data.filename.replace(/\.[^.]+$/, "") + "_result.json");

  $("result-card").scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ---------- 示例图 ---------- */
async function extractExample(group, name) {
  if (busy) return;
  busy = true;
  const btn = $("btn-extract");
  btn.classList.add("loading");
  $("btn-label").textContent = "提取中…";
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
    showResult(data);
    flash(`提取完成：${data.n_curves} 条曲线，${data.elapsed_s}s`, false);
  } catch (err) {
    flash(`提取失败：${err.message}`, true);
  } finally {
    busy = false;
    btn.classList.remove("loading");
    $("btn-label").textContent = "③ 点击提取";
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
        <div class="ex-name">${it.name}</div><div class="ex-group">${it.group}</div>`;
      div.addEventListener("click", () => extractExample(it.group, it.name));
      box.appendChild(div);
    }
  } catch (e) {
    $("examples").innerHTML = `<p class="hint">示例图加载失败：${e.message}</p>`;
  }
}
loadExamples();
