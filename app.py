"""
app.py
------
Flask REST API for the plagiarism detection system.

Endpoints:
  POST /analyze   — Upload multiple files, get similarity results
  GET  /health    — Health check
"""

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from text_extractor import extract_text
from plagiarism_engine import compute_similarities, build_results

# ─── App Setup ───────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
MAX_FILES = 20
MAX_FILE_SIZE_MB = 50


# ─── Static Frontend ─────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ─── Health Check ────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok", "message": "Plagiarism Detection API running."})


# ─── Main Analysis Endpoint ──────────────────────────────────────────────────

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accepts multipart/form-data with field 'files[]'.
    Returns JSON with extraction results and pairwise similarity rankings.
    """
    uploaded_files = request.files.getlist("files[]")

    # ── Validation ────────────────────────────────────────────────────────────
    if not uploaded_files or len(uploaded_files) < 2:
        return jsonify({"error": "Please upload at least 2 files to compare."}), 400

    if len(uploaded_files) > MAX_FILES:
        return jsonify({"error": f"Maximum {MAX_FILES} files allowed per analysis."}), 400

    for f in uploaded_files:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({
                "error": f"Unsupported file type '{ext}'. Allowed: PDF, DOCX."
            }), 400

    # ── Extract Text ──────────────────────────────────────────────────────────
    extractions = []
    warnings = []

    for file in uploaded_files:
        file_bytes = file.read()
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            return jsonify({"error": f"File '{file.filename}' exceeds {MAX_FILE_SIZE_MB}MB limit."}), 400

        print(f"[INFO] Extracting: {file.filename} ({size_mb:.1f} MB)")
        result = extract_text(file_bytes, file.filename)
        extractions.append(result)

        if not result["text"].strip():
            warnings.append(f"No text could be extracted from '{file.filename}'.")

    # Filter out files with no text
    valid = [e for e in extractions if e["text"].strip()]
    if len(valid) < 2:
        return jsonify({
            "error": "Could not extract enough text from the uploaded files to compare.",
            "warnings": warnings
        }), 422

    # ── Compute Similarities ──────────────────────────────────────────────────
    filenames = [e["filename"] for e in valid]
    texts     = [e["text"]     for e in valid]

    try:
        sim_matrices = compute_similarities(texts)
    except Exception as e:
        return jsonify({"error": f"Similarity computation failed: {str(e)}"}), 500

    results = build_results(filenames, sim_matrices)

    # ── Build Response ────────────────────────────────────────────────────────
    documents = [
        {
            "filename": e["filename"],
            "method":   e["method"],
            "pages":    e["pages"],
            "preview":  e["text"][:800] + ("..." if len(e["text"]) > 800 else ""),
            "char_count": len(e["text"]),
        }
        for e in extractions
    ]

    return jsonify({
        "documents": documents,
        "results":   results,
        "warnings":  warnings,
        "total_pairs": len(results),
    })


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*60)
    print("  Plagiarism Detection System")
    print("  http://localhost:5000")
    print("="*60)
    app.run(host="0.0.0.0", port=5000, debug=False)
