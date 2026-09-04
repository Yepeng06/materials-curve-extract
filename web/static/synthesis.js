// 材料曲线数据合成 Web 端 v2 — 统一生成器 + 点击放大预览
(function () {
  "use strict";

  let currentTid = null;
  let pollTimer = null;

  // ---- 收集参数 ----
  function collectParams() {
    const cts = Array.from(document.querySelectorAll(".ct:checked")).map((c) => c.value);
    return {
      params: {
        count: parseInt(document.getElementById("count").value, 10) || 20,
        seed: parseInt(document.getElementById("seed").value, 10) || 20260828,
        curve_types: cts,
        x_axis: document.getElementById("xaxis").value,
        y_axis: document.getElementById("yaxis").value,
        dpi: document.getElementById("dpi").value,
        line_style: document.getElementById("ls").value,
        degrade: document.getElementById("degrade").checked,
        num_curves: parseInt(document.getElementById("num-curves").value, 10) || 1,
      },
    };
  }

  // ---- 提交生成 ----
  document.getElementById("btn-generate").addEventListener("click", () => {
    const payload = collectParams();
    document.getElementById("gen-error").textContent = "";
    fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then((r) => (r.ok ? r.json() : r.json().then((e) => Promise.reject(e))))
      .then((d) => {
        currentTid = d.task_id;
        resetResult();
        document.getElementById("progress-wrap").classList.remove("hidden");
        startPoll();
      })
      .catch((e) => {
        document.getElementById("gen-error").textContent =
          "提交失败：" + (e.detail || e.message || JSON.stringify(e));
      });
  });

  function resetResult() {
    document.getElementById("gallery").innerHTML = "";
    document.getElementById("result-actions").classList.add("hidden");
    document.getElementById("gen-log").textContent = "";
  }

  // ---- 轮询状态 ----
  function poll() {
    if (!currentTid) return;
    fetch("/api/tasks/" + currentTid)
      .then((r) => r.json())
      .then((t) => {
        const pct = t.target_count ? Math.round((t.progress / t.target_count) * 100) : 0;
        document.getElementById("progress-bar").style.width = pct + "%";
        document.getElementById("progress-text").textContent =
          `${t.progress}/${t.target_count} 张 · ${t.elapsed_s}s · ${t.message}`;
        document.getElementById("gen-log").textContent = t.log || "";

        const st = document.getElementById("task-status");
        st.className = "status " + t.status;
        st.textContent =
          t.status === "running" ? "⏳ 生成中…" :
          t.status === "done" ? "✅ 生成完成" :
          t.status === "failed" ? "❌ 失败" : "空闲";

        if (t.status === "done") {
          document.getElementById("result-actions").classList.remove("hidden");
          document.getElementById("btn-download").href = `/api/tasks/${currentTid}/download`;
          loadSamples();
          clearInterval(pollTimer);
          pollTimer = null;
        } else if (t.status === "failed") {
          clearInterval(pollTimer);
          pollTimer = null;
        }
      })
      .catch(() => {});
  }

  function loadSamples() {
    fetch("/api/tasks/" + currentTid + "/samples?limit=12")
      .then((r) => r.json())
      .then((d) => {
        const g = document.getElementById("gallery");
        g.innerHTML = "";
        d.items.forEach((it) => {
          const img = document.createElement("img");
          img.src = it.url;
          img.alt = it.name;
          img.title = "点击放大查看: " + it.name;
          img.style.cursor = "pointer";
          img.addEventListener("click", function () {
            openLightbox(it.url);
          });
          g.appendChild(img);
        });
        if (d.total > 12) {
          const more = document.createElement("div");
          more.className = "more";
          more.textContent = `… 还有 ${d.total - 12} 张（下载 ZIP 查看全部）`;
          g.appendChild(more);
        }
      })
      .catch(() => {});
  }

  // ---- Lightbox 放大查看 ----
  window.openLightbox = function (url) {
    const lb = document.getElementById("lightbox");
    const img = document.getElementById("lightbox-img");
    img.src = url;
    lb.classList.remove("hidden");
  };

  window.closeLightbox = function () {
    document.getElementById("lightbox").classList.add("hidden");
  };

  // ---- 启动轮询 ----
  function startPoll() {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(poll, 1200);
  }
})();