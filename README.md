# Plagiarism Checker

![Status](https://img.shields.io/badge/Status-Deployment%20Ready-success)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Backend-Flask-black)

A fast, lightweight web application that detects copied or paraphrased content across multiple documents. It works seamlessly with PDFs, Word Documents, and even scanned images.

## Features
* **Multi-Format Support:** Read text from `.pdf`, `.docx`, and scanned image PDFs.
* **Smart Detection:** Catches exact matches, paraphrased text (meaning match), and structural similarities.
* **Built-in OCR:** Uses Tesseract to read text from handwriting and scanned documents.
* **Live Demo:** Try out the included sample documents instantly.

---

## 🚀 How to Run Locally

### Step 1: Install Dependencies
Make sure you have Python installed. Then, open your terminal in the project folder and run:
```bash
pip install -r requirements.txt
```

### Step 2: Install System Tools (For Scanned PDFs)
To process images and scanned PDFs, you need to install Tesseract OCR and Poppler:
* **Windows:** 
  * Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (it should be in `C:\Program Files\Tesseract-OCR\tesseract.exe`).
  * Install [Poppler](https://github.com/oschwartz10612/poppler-windows/releases) and add its `bin` folder to your system PATH.
* **Linux (Ubuntu/Debian):**
  ```bash
  sudo apt-get update
  sudo apt-get install tesseract-ocr poppler-utils
  ```

### Step 3: Start the Application
Run the Flask server:
```bash
python app.py
```
Open `http://localhost:5000` in your web browser.

---

## 🌐 How to Deploy (Render / Docker)

This application is completely production-ready and dockerized.

1. Create a new **Web Service** on Render or your favorite hosting platform.
2. Connect this GitHub repository.
3. Choose **Docker** as the runtime environment.
4. The platform will automatically install all necessary system packages (Poppler, Tesseract) and start the app. No extra configuration needed!

---

## 🛠️ Built With

* **Backend:** Python, Flask
* **AI & NLP:** Scikit-learn (TF-IDF), Sentence-Transformers
* **OCR & Vision:** OpenCV, Tesseract, pdf2image
* **Frontend:** Vanilla HTML, CSS, JavaScript