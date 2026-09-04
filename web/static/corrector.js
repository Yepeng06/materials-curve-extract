/* 人工校正编辑器 — 内嵌在提取结果卡片中（Konva）。
 *
 * 交互（对齐 goal.md 任务 4.4 最小闭环 + 大数据中心 Konva 范式）：
 *   · 提取点以可拖拽锚点叠加在原图上；
 *   · 拖拽锚点微调；Shift+点击在选中曲线上加点（按 x 插入）；
 *   · 点击锚点选中，Delete / 删点按钮移除；
 *   · 保存 → POST /api/runs/{tid}/correct → 后端像素→数据反算并重导出。
 * 全部本地资源（Konva 本地化），离线演示可用。
 */
(function () {
  "use strict";

  let stage = null, layer = null, imgNode = null;
  let activeCard = null, activeData = null;
  let activeCurve = 0;
  let curveGroups = [];      // [{line, anchors:[Konva.Circle], visible}]
  let scale = 1;

  const ESC = (s) => String(s == null ? "" : s).replace(/</g, "&lt;");

  function hexColor(c) {
    return Array.isArray(c) && c.length === 3 ? "#" + c.map(v => (+v).toString(16).padStart(2, "0")).join("") : "#e11d48";
  }

  function toggle(card, data) {
    if (activeCard === card) { close(); return; }
    close();
    activeCard = card;
    activeData = data;
    const wrap = card.querySelector("[data-corr]");
    wrap.classList.add("open");
    wrap.innerHTML = `
      <div class="corr-head">
        <span class="title">✏️ 人工校正</span>
        <span class="ops">拖拽锚点微调 · Shift+点击加点 · 点击选中后删除</span>
        <span class="spacer"></span>
        <button class="btn small primary" data-save>保存并重导出</button>
        <button class="btn small" data-close>收起</button>
      </div>
      <div class="corr-body">
        <div class="corr-canvas-box" data-stage></div>
        <div class="corr-side" data-side></div>
      </div>
      <div class="corr-trace" data-trace hidden></div>`;
    wrap.querySelector("[data-close]").addEventListener("click", close);
    wrap.querySelector("[data-save]").addEventListener("click", save);
    buildStage(wrap);
  }

  function buildStage(wrap) {
    const box = wrap.querySelector("[data-stage]");
    const side = wrap.querySelector("[data-side]");
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const maxW = Math.max(360, box.clientWidth - 2);
      const maxH = 520;
      scale = Math.min(maxW / img.width, maxH / img.height, 1.6);
      const W = Math.round(img.width * scale), H = Math.round(img.height * scale);
      box.style.height = H + "px";
      stage = new Konva.Stage({ container: box, width: W, height: H });
      layer = new Konva.Layer();
      stage.add(layer);

      imgNode = new Konva.Image({ image: img, width: W, height: H, listening: false });
      layer.add(imgNode);

      // build curve groups
      curveGroups = [];
      fetch("/api/runs/" + activeData.task_id + "/json")
        .then(r => r.json())
        .then(result => {
          activeResult = result;
          result.curves.forEach((c, i) => {
            const pts = (c.pixel_points && c.pixel_points.length === c.points.length)
              ? c.pixel_points
              : c.points.map(p => [xValToPx(result.x_axis, p[0]), yValToPx(result.y_axis, p[1])]);
            const color = hexColor(c.color);
            const group = { color, visible: true };
            group.line = new Konva.Line({
              points: pts.flatMap(p => [p[0] * scale, p[1] * scale]),
              stroke: color, strokeWidth: 1.6, opacity: 0.85, listening: false,
            });
            layer.add(group.line);
            group.anchors = pts.map(p => makeAnchor(p[0], p[1], color, i));
            curveGroups.push(group);
          });
          renderSide(side);
          layer.batchDraw();
        })
        .catch(() => { side.innerHTML = `<p class="hint">result.json 加载失败</p>`; });

      // Shift+click adds a point to the active curve (inserted by x order)
      stage.on("click tap", (e) => {
        if (!e.evt.shiftKey || activeResult == null) return;
        const pos = stage.getPointerPosition();
        if (!pos || !curveGroups[activeCurve]) return;
        const px = pos.x / scale, py = pos.y / scale;
        const g = curveGroups[activeCurve];
        const anchor = makeAnchor(px, py, g.color, activeCurve);
        // insert into polyline by pixel x order
        const xs = g.anchors.map(a => a.px);
        let at = xs.findIndex(v => v > px);
        if (at < 0) at = g.anchors.length;
        g.anchors.splice(at, 0, anchor);
        redrawLine(g);
        ops.add++;
        traceMsg(`曲线${activeCurve + 1} 添加点 (${px.toFixed(0)}, ${py.toFixed(0)})`);
        layer.batchDraw();
      });
    };
    img.onerror = () => { box.innerHTML = `<p class="hint">原图加载失败</p>`; };
    img.src = activeData.downloads.image + "?t=" + Date.now();
  }

  let activeResult = null;
  const ops = { moved: 0, add: 0, del: 0 };

  function xValToPx(ax, v) {
    const s = ax.sign != null ? ax.sign : 1;
    const t = ax.kind === "log" ? Math.log10(Math.max(v, 1e-300)) : v;
    return (t - ax.intercept) / ax.slope / s;
  }
  function yValToPx(ax, v) {
    const s = ax.sign != null ? ax.sign : -1;
    const t = ax.kind === "log" ? Math.log10(Math.max(v, 1e-300)) : v;
    return (t - ax.intercept) / ax.slope / s;
  }

  function makeAnchor(px, py, color, curveIdx) {
    const node = new Konva.Circle({
      x: px * scale, y: py * scale, radius: 5,
      fill: color, stroke: "#fff", strokeWidth: 1.4,
      draggable: true,
    });
    node.px = px; node.py = py; node.curveIdx = curveIdx;
    node.on("dragmove", () => {
      node.px = node.x() / scale; node.py = node.y() / scale;
      const g = curveGroups[curveIdx];
      if (g) redrawLine(g);
    });
    node.on("dragend", () => { if (node._originSet !== true) { node._originSet = true; } ops.moved++; traceMsg(`曲线${curveIdx + 1} 移动点`); });
    node.on("click", () => selectAnchor(node));
    layer.add(node);
    return node;
  }

  let selected = null;
  function selectAnchor(node) {
    if (selected) selected.strokeWidth(1.4);
    selected = node;
    node.strokeWidth(3);
    layer.batchDraw();
  }

  document.addEventListener("keydown", (e) => {
    if ((e.key === "Delete" || e.key === "Backspace") && selected && activeCard) {
      e.preventDefault();
      deleteSelected();
    }
  });

  function deleteSelected() {
    if (!selected) return;
    const g = curveGroups[selected.curveIdx];
    const i = g.anchors.indexOf(selected);
    if (i >= 0) g.anchors.splice(i, 1);
    selected.destroy();
    ops.del++;
    traceMsg(`曲线${selected.curveIdx + 1} 删除点`);
    redrawLine(g);
    selected = null;
    layer.batchDraw();
  }

  function redrawLine(g) {
    g.line.points(g.anchors.flatMap(a => [a.x(), a.y()]));
  }

  function renderSide(side) {
    side.innerHTML = "";
    curveGroups.forEach((g, i) => {
      const item = document.createElement("div");
      item.className = "curve-item" + (i === activeCurve ? " active" : "");
      item.innerHTML = `
        <input type="checkbox" checked data-vis>
        <span class="dot" style="background:${g.color}"></span>
        <span class="cname">曲线${i + 1}</span>
        <span class="cpt">${g.anchors.length} 点</span>`;
      item.addEventListener("click", (e) => {
        if (e.target.tagName === "INPUT") return;
        activeCurve = i;
        renderSide(side);
      });
      item.querySelector("[data-vis]").addEventListener("change", (e) => {
        g.visible = e.target.checked;
        g.line.visible(g.visible);
        g.anchors.forEach(a => a.visible(g.visible));
        layer.batchDraw();
      });
      side.appendChild(item);
    });
    const hint = document.createElement("p");
    hint.className = "hint";
    hint.innerHTML = "Shift+点击画布：在选中曲线加点<br>点击锚点选中，Delete 删除<br>拖拽锚点：微调位置<br>保存后自动重算数据坐标并重导出";
    side.appendChild(hint);
    const delBtn = document.createElement("button");
    delBtn.className = "btn small";
    delBtn.textContent = "删除选中点";
    delBtn.addEventListener("click", deleteSelected);
    side.appendChild(delBtn);
  }

  function traceMsg(msg) {
    const t = activeCard.querySelector("[data-trace]");
    if (!t) return;
    t.hidden = false;
    const head = `操作统计 — 移动 ${ops.moved} · 加点 ${ops.add} · 删点 ${ops.del}`;
    t.textContent = head + "\n" + (t.dataset.log ? t.dataset.log + "\n" : "") + msg;
    t.dataset.log = (t.dataset.log ? t.dataset.log + "\n" : "") + msg;
  }

  async function save() {
    if (!activeResult) return;
    const payload = {
      curves: curveGroups.map((g, i) => ({
        index: i,
        pixel_points: g.anchors
          .map(a => [Math.round(a.px), Math.round(a.py)])
          .sort((p, q) => p[0] - q[0]),
      })),
      note: "web editor",
    };
    const btn = activeCard.querySelector("[data-save]");
    btn.disabled = true; btn.textContent = "保存中…";
    try {
      const resp = await fetch("/api/runs/" + activeData.task_id + "/correct", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) throw new Error(errorMessage(data.detail, resp.status));
      traceMsg(`已保存：${data.operator_ops} 次操作，CSV/JSON/重绘图已重新生成`);
      // 刷新卡片图片与下载链接（叠加图/重绘图已更新）
      const card = activeCard;
      const imgs = card.querySelectorAll("img.rc-overlay");
      imgs.forEach(im => { const u = new URL(im.src); u.searchParams.set("t", Date.now()); im.src = u.toString(); });
    } catch (err) {
      traceMsg("保存失败：" + err.message);
    } finally {
      btn.disabled = false; btn.textContent = "保存并重导出";
    }
  }

  function close() {
    if (activeCard) {
      const wrap = activeCard.querySelector("[data-corr]");
      if (wrap) { wrap.classList.remove("open"); wrap.innerHTML = ""; }
    }
    if (stage) { stage.destroy(); stage = null; }
    activeCard = null; activeData = null; activeResult = null;
    curveGroups = []; selected = null; activeCurve = 0;
    ops.moved = ops.add = ops.del = 0;
  }

  window.Corrector = { toggle, close };
})();
