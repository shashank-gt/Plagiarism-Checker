/**
 * app.js — Plag Check Frontend Logic
 */

// ── State ─────────────────────────────────────────────────────
let selectedFiles = [];

// ── DOM Refs ──────────────────────────────────────────────────
const dropZone    = document.getElementById("dropZone");
const fileInput   = document.getElementById("fileInput");
const browseBtn   = document.getElementById("browseBtn");
const analyzeBtn  = document.getElementById("analyzeBtn");
const fileChips   = document.getElementById("fileChips");
const loaderWrap  = document.getElementById("loaderWrap");
const resultsSection = document.getElementById("resultsSection");
const errorBox    = document.getElementById("errorBox");
const statsRow    = document.getElementById("statsRow");
const resultsList = document.getElementById("resultsList");
const previewsGrid= document.getElementById("previewsGrid");
const warningsBox = document.getElementById("warningsBox");
const loaderTitle = document.getElementById("loaderTitle");
const step1 = document.getElementById("step1");
const step2 = document.getElementById("step2");
const step3 = document.getElementById("step3");

// ── File Handling ─────────────────────────────────────────────

browseBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  fileInput.click();
});

dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  addFiles(Array.from(fileInput.files));
  fileInput.value = "";
});

const loadSamplesBtn = document.getElementById("loadSamplesBtn");
if (loadSamplesBtn) {
  loadSamplesBtn.addEventListener("click", async (e) => {
    e.stopPropagation();
    const sampleFiles = [
      "samples/xample1.pdf",
      "samples/xample2.pdf"
    ];
    
    selectedFiles = [];
    renderChips();
    
    try {
      showLoader();
      loaderTitle.textContent = "Loading samples...";
      const fetchedFiles = [];
      
      for (const url of sampleFiles) {
        const resp = await fetch("/" + url);
        if (!resp.ok) throw new Error("Failed to load " + url);
        const blob = await resp.blob();
        const filename = url.split("/").pop();
        const file = new File([blob], filename, { type: "application/vnd.openxmlformats-officedocument.wordprocessingml.document" });
        // Setting a fake property to pass 'isAllowed' check which uses file.name
        fetchedFiles.push(file);
      }
      
      hideLoader();
      addFiles(fetchedFiles);
      runAnalysis(); // Auto-run analysis when samples are loaded
    } catch (error) {
      hideLoader();
      showError("Failed to load sample files. Make sure they exist on the server.");
    }
  });
}

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const files = Array.from(e.dataTransfer.files).filter(isAllowed);
  addFiles(files);
});

function isAllowed(f) {
  return /\.(pdf|docx|doc)$/i.test(f.name);
}

function addFiles(files) {
  files.forEach(f => {
    if (isAllowed(f) && !selectedFiles.find(x => x.name === f.name && x.size === f.size)) {
      selectedFiles.push(f);
    }
  });
  renderChips();
  updateAnalyzeButton();
}

function removeFile(index) {
  selectedFiles.splice(index, 1);
  renderChips();
  updateAnalyzeButton();
}

function renderChips() {
  fileChips.innerHTML = "";
  selectedFiles.forEach((f, i) => {
    const ext  = f.name.split(".").pop().toUpperCase();
    const icon = ext === "PDF" ? "📄" : "📝";
    const size = (f.size / 1024).toFixed(0) + " KB";

    const chip = document.createElement("div");
    chip.className = "chip";
    chip.innerHTML = `
      <span class="chip-icon">${icon}</span>
      <span class="chip-name" title="${f.name}">${f.name}</span>
      <span style="color:var(--muted);font-size:0.72rem;margin-left:2px">${size}</span>
      <button class="chip-remove" data-index="${i}" title="Remove">✕</button>
    `;
    fileChips.appendChild(chip);
  });

  // Bind remove buttons
  fileChips.querySelectorAll(".chip-remove").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      removeFile(parseInt(btn.dataset.index));
    });
  });
}

function updateAnalyzeButton() {
  analyzeBtn.disabled = selectedFiles.length < 2;
  if (selectedFiles.length >= 2) {
    analyzeBtn.textContent = `Analyze ${selectedFiles.length} Documents`;
    analyzeBtn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
      </svg>
      Analyze ${selectedFiles.length} Documents
    `;
  } else {
    analyzeBtn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
      </svg>
      Analyze Documents
    `;
  }
}

// ── Analyze ───────────────────────────────────────────────────

analyzeBtn.addEventListener("click", runAnalysis);

async function runAnalysis() {
  if (selectedFiles.length < 2) return;

  // Reset UI
  hideError();
  hideResults();
  showLoader();

  // Disable button during analysis
  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = `<div style="width:18px;height:18px;border:3px solid rgba(255,255,255,0.3);border-top-color:#fff;border-radius:50%;animation:spin 0.8s linear infinite"></div> Analyzing…`;

  // Build form data
  const formData = new FormData();
  selectedFiles.forEach(f => formData.append("files[]", f));

  // Animate steps
  animateSteps();

  try {
    const res = await fetch("/analyze", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();

    hideLoader();

    if (!res.ok) {
      showError(data.error || "Analysis failed. Please try again.");
    } else {
      renderResults(data);
    }
  } catch (err) {
    hideLoader();
    showError("Could not reach the server. Make sure the app is running on localhost:5000.");
  } finally {
    // Re-enable button
    analyzeBtn.disabled = false;
    updateAnalyzeButton();
  }
}

// ── Step Animation ────────────────────────────────────────────

function animateSteps() {
  const steps = [step1, step2, step3];
  const messages = ["Extracting text…", "Computing embeddings…", "Scoring similarity…"];

  steps.forEach(s => s.classList.remove("active"));
  steps[0].classList.add("active");
  loaderTitle.textContent = messages[0];

  let idx = 0;
  const interval = setInterval(() => {
    idx++;
    if (idx >= steps.length) { clearInterval(interval); return; }
    steps.forEach(s => s.classList.remove("active"));
    steps[idx].classList.add("active");
    loaderTitle.textContent = messages[idx];
  }, 4000);
}

// ── Render Results ────────────────────────────────────────────

function renderResults(data) {
  // Warnings
  if (data.warnings && data.warnings.length > 0) {
    warningsBox.innerHTML = "⚠️ " + data.warnings.join("<br/>⚠️ ");
    warningsBox.style.display = "block";
  } else {
    warningsBox.style.display = "none";
  }

  // Stats
  const highRisk = data.results.filter(r => r.similarity >= 70).length;
  const maxSim   = data.results[0]?.similarity ?? 0;
  const avgSim   = data.results.length
    ? (data.results.reduce((s,r) => s + r.similarity, 0) / data.results.length).toFixed(1)
    : 0;

  statsRow.innerHTML = `
    ${statCard(data.documents.length, "Documents")}
    ${statCard(data.total_pairs, "Pairs Compared")}
    ${statCard(maxSim + "%", "Highest Similarity", colorClass(maxSim))}
    ${statCard(highRisk, "High-Risk Pairs", highRisk > 0 ? "var(--red)" : "var(--green)")}
  `;

  // Results rows
  resultsList.innerHTML = "";
  data.results.forEach((r, i) => {
    const cls = colorClass(r.similarity);
    const rank = i < 3
      ? `<div class="rank-badge rank-${i+1}">${["🥇","🥈","🥉"][i]}</div>`
      : `<div class="rank-badge rank-n">#${i+1}</div>`;

    const row = document.createElement("div");
    row.className = "result-row";
    row.style.animationDelay = `${i * 0.05}s`;
    row.innerHTML = `
      ${rank}
      <div class="result-files">
        <div class="result-pair" title="${r.file1} vs ${r.file2}">
          📄 ${r.file1} <span style="color:var(--muted)">vs</span> 📄 ${r.file2}
        </div>
        <div class="result-breakdown">
          TF-IDF: ${r.tfidf_pct}% &nbsp;|&nbsp; Semantic: ${r.semantic_pct}% &nbsp;|&nbsp; N-gram: ${r.jaccard_pct}%
        </div>
      </div>
      <div class="result-score">
        <span class="score-pct ${cls}">${r.similarity}%</span>
        <div class="progress-bar">
          <div class="progress-fill ${cls}" style="width:${r.similarity}%"></div>
        </div>
      </div>
    `;
    resultsList.appendChild(row);
  });

  // Previews
  previewsGrid.innerHTML = "";
  data.documents.forEach((doc, i) => {
    const methodTag = `<span class="method-tag ${doc.method}">${doc.method.toUpperCase()}</span>`;
    const preview = document.createElement("div");
    preview.className = "preview-item";
    preview.innerHTML = `
      <div class="preview-header" onclick="togglePreview(this)">
        <div class="preview-name">
          📄 ${doc.filename} ${methodTag}
        </div>
        <div style="display:flex;align-items:center;gap:16px">
          <span class="preview-meta">${doc.pages} page(s) · ${doc.char_count.toLocaleString()} chars</span>
          <svg class="preview-toggle" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="6 9 12 15 18 9"/>
          </svg>
        </div>
      </div>
      <pre class="preview-body">${escapeHtml(doc.preview)}</pre>
    `;
    previewsGrid.appendChild(preview);
  });

  showResults();
}

function statCard(value, label, color = "var(--accent2)") {
  return `
    <div class="stat-card" style="animation-delay:${Math.random()*0.3}s">
      <div class="stat-value" style="color:${color}">${value}</div>
      <div class="stat-label">${label}</div>
    </div>
  `;
}

function colorClass(pct) {
  if (pct >= 70) return "high";
  if (pct >= 40) return "medium";
  return "low";
}

function togglePreview(header) {
  const body   = header.nextElementSibling;
  const toggle = header.querySelector(".preview-toggle");
  const isOpen = body.classList.contains("open");
  body.classList.toggle("open", !isOpen);
  toggle.classList.toggle("open", !isOpen);
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// ── UI Helpers ────────────────────────────────────────────────

function showLoader()   { loaderWrap.style.display = "flex"; }
function hideLoader()   { loaderWrap.style.display = "none"; }
function showResults()  { resultsSection.style.display = "block"; }
function hideResults()  { resultsSection.style.display = "none"; }
function showError(msg) { errorBox.innerHTML = "❌ " + msg; errorBox.style.display = "block"; }
function hideError()    { errorBox.style.display = "none"; }
