# Plagiarism Checker

![Status](https://img.shields.io/badge/Status-Deployment%20Ready-success)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Backend-Flask-black)

This project is a user-friendly plagiarism detection web app built for comparing multiple documents at once.
It reads text from digital PDFs and Word files, and it can also extract content from scanned or image-based PDFs using OCR.

## Features
* **Multiple document types:** Supports `.pdf`, `.docx`, and scanned PDF uploads.
* **Robust similarity checks:** Uses a mix of TF-IDF, n-gram analysis, and lightweight semantic comparison.
* **OCR support:** Scans images and handwritten text with Tesseract for accurate extraction.
* **Clean results view:** Returns document previews, page counts, and ranked similarity scores.

---

## How to Run Locally

### Step 1: Install Python Dependencies
From the project root directory, install the required packages:
```bash
pip install -r requirements.txt
```

### Step 2: Install System Dependencies for OCR
If you want scanned or image-based PDFs to work correctly, install the required system tools.

* **Windows:**
  * Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
  * Install [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases) and add the `bin` directory to your PATH.

* **Linux (Ubuntu/Debian):**
  ```bash
  sudo apt-get update
  sudo apt-get install tesseract-ocr poppler-utils
  ```

### Step 3: Start the App
Launch the application:
```bash
python app.py
```
Then open `http://localhost:5000` in your browser.

---

## 🛠️ Built With

* **Backend:** Python, Flask
* **Text analysis:** Scikit-learn
* **Document extraction:** PyPDF2, python-docx
* **OCR / image processing:** Tesseract, OpenCV, pdf2image
* **Frontend:** HTML, CSS, JavaScript
