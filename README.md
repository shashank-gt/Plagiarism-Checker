# Plagiarism Detection System

This project detects similarity between student assignment PDFs. It extracts text from PDF files (supports normal text PDFs and scanned PDFs via OCR), computes TF-IDF vectors, and uses cosine similarity to report pairwise similarity percentages. The program produces a CSV report listing suspicious pairs and a short summary of highly similar submissions.

# Project Files

**AssignmentChecker.py** — Main Python script that:

Extracts text from PDFs (PyPDF2 for text PDFs; pdf2image + pytesseract for scanned PDFs),

Preprocesses images for OCR (simple denoising + thresholding),

Builds TF-IDF vectors and computes cosine similarity,

Writes plagiarism_results.csv and opens it automatically.

**submissions/** — Folder where you place all student PDF files to be scanned. (No separate dataset required — the folder itself is the input.)

# Features

**PDF text extraction:** Reads selectable text from digital PDFs with PyPDF2.

**OCR support (optional):** Uses pdf2image + pytesseract with basic preprocessing to handle scanned/image PDFs.

**TF-IDF + Cosine Similarity:** Converts documents to TF-IDF vectors and computes pairwise cosine similarity for robust text overlap detection.

**Batch processing:** Handles many PDFs in one run (e.g., 70+ files).

**CSV report:** Generates plagiarism_results.csv sorted by descending similarity, with a summary section for pairs above a configurable threshold.

**Auto-open:** Attempts to open the CSV automatically on completion (Windows os.startfile).

**Simple thresholds:** Configurable high-similarity threshold (default 80%) for quick triage.

# Requirements

Python 3.8+

System tools (for OCR, optional):

**1. Tesseract OCR** (if OCR is required) — install and add to PATH (e.g., C:\Program Files\Tesseract-OCR\)

**22. Poppler** (for pdf2image) — extract and add C:\poppler\Library\bin to PATH

**3. Python packages:**

pip install PyPDF2 pandas scikit-learn tqdm pdf2image pillow pytesseract opencv-python-headless


# Installation & Usage

1. Clone / copy project folder and open it in VS Code.

2. Create submissions/ folder inside project and add all student PDF files:

project_root/
├─AssignmentChecker.py
└─ submissions/
   ├─ student1.pdf
   ├─ student2.pdf
   └─ ...


3. Install Python dependencies

pip install PyPDF2 pandas scikit-learn tqdm pdf2image pillow pytesseract opencv-python-headless

4. Install system tools
Install Tesseract and add its folder to PATH (e.g., C:\Program Files\Tesseract-OCR\).
Install Poppler and add C:\poppler\Library\bin to PATH.
Restart your terminal after editing PATH.

5. Run the script

python AssignmentChecker.py

6. Inspect output

plagiarism_results.csv columns: File 1, File 2, Similarity (%).

Summary (appended to CSV and printed to console) lists pairs above the high-similarity threshold.
