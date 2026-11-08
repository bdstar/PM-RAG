// Shared front-end helpers for the PM-RAG dashboard.
async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  const ct = res.headers.get("content-type") || "";
  return ct.includes("application/json") ? res.json() : res.text();
}

function showAlert(msg, type = "info") {
  const area = document.getElementById("alert-area");
  if (!area) return;
  const div = document.createElement("div");
  div.className = `alert alert-${type} alert-dismissible fade show`;
  div.innerHTML = `${msg}<button type="button" class="btn-close" data-bs-dismiss="alert"></button>`;
  area.appendChild(div);
  if (type === "success" || type === "info") setTimeout(() => div.remove(), 6000);
}

function fmt(n, digits = 2) {
  if (n === null || n === undefined) return "—";
  if (typeof n === "number") return Number.isInteger(n) ? n.toLocaleString() : n.toFixed(digits);
  return n;
}
function pct(n) { return n === null || n === undefined ? "—" : (n * 100).toFixed(1) + "%"; }

// Poll a background job until it finishes; onProgress(job) called on each tick.
async function pollJob(jobId, onProgress) {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const job = await api(`/api/jobs/${jobId}`);
        if (onProgress) onProgress(job);
        if (job.status === "completed") return resolve(job);
        if (job.status === "failed") return reject(new Error(job.error || "Job failed"));
        setTimeout(tick, 1200);
      } catch (e) { reject(e); }
    };
    tick();
  });
}

async function updateBackendBadge() {
  const badge = document.getElementById("backend-badge");
  if (!badge) return;
  try {
    const s = await api("/api/system/status");
    badge.className = "badge bg-success";
    badge.textContent = `DB:${s.database.backend} · Vec:${s.vector_store.backend} · LLM:${s.models.llm_backend}`;
  } catch (e) {
    badge.className = "badge bg-danger";
    badge.textContent = "offline";
  }
}
document.addEventListener("DOMContentLoaded", updateBackendBadge);

// Minimal Chart.js helper
function barChart(ctx, labels, datasets, options = {}) {
  return new Chart(ctx, { type: "bar", data: { labels, datasets }, options: {
    responsive: true, maintainAspectRatio: false, ...options } });
}
function lineChart(ctx, labels, datasets, options = {}) {
  return new Chart(ctx, { type: "line", data: { labels, datasets }, options: {
    responsive: true, maintainAspectRatio: false, ...options } });
}
const PALETTE = ["#1b4965","#5fa8d3","#62b6cb","#cae9ff","#f4a261","#e76f51","#2a9d8f","#8d99ae"];
