# Plagiarism Detection

![Status](https://img.shields.io/badge/Status-Deployment%20Ready-success)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![Flask](https://img.shields.io/badge/Backend-Flask-black)

A simple web app to check copied content in documents.
It detects both exact copying and rewritten (paraphrased) text.

**Supports:**
PDF, Word (.docx), and scanned/handwritten files (OCR)

---

## How it works

* Upload documents
* Extracts text (direct or using OCR for scanned files)
* Compares content using:

  * word match
  * meaning match
  * pattern match
* Shows similarity score with High / Medium / Low level

---

## Tech

* **Backend:** Python, Flask
* **AI/ML:** Scikit-learn, Sentence Transformers
* **OCR:** OpenCV, Tesseract
* **Frontend:** HTML, CSS, JavaScript

---

## Deployment on Render
This app is configured for native deployment on Render via Docker.
1. Connect your repository to Render.
2. Render will automatically detect the `render.yaml` or `Dockerfile`.
3. It installs necessary system packages (Tesseract, OpenCV gl dependencies) seamlessly.

---

## Run Locally

```bash
git clone https://github.com/shashank-gt/Plagiarism-Checker.git
cd Plagiarism-Checker
pip install -r requirements.txt
python app.py
```

Open: [http://localhost:10000](http://localhost:10000)

---

## Structure

```
app.py
plagiarism_engine.py
text_extractor.py
frontend/
Dockerfile
render.yaml
Procfile
```

---

## Testing

```bash
python generate_samples.py
```

---

## Features

* Detects copied + paraphrased text
* Works on scanned files
* Clear similarity score
* Simple UI

---

## Author
Shashank H K