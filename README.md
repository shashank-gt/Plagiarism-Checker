# Plag Detect: AI-Powered Multi-Modal Plagiarism Engine

![Plag Detect Demo](https://img.shields.io/badge/Status-Deployment%20Ready-success)
![Python 3.13](https://img.shields.io/badge/Python-3.13-blue)
![Flask](https://img.shields.io/badge/Backend-Flask-black)

**Plag Detect** is a full-stack, AI-powered web application designed to detect plagiarism across multiple document formats. Unlike basic keyword-matching tools, Plag Detect utilizes an ensemble of lexical, semantic, and structural NLP logic to accurately identify direct copy-pasting *and* paraphrased content. It supports digital PDFs, Word Documents (`.docx`), and significantly, **scanned or handwritten documents via intelligent OCR fallback**.

---

## 🏗 Architecture & Workflow

Plag Detect is built around a robust, multi-phase analysis pipeline.

### 1. The Multi-Modal Ingestion Pipeline
When documents are uploaded, the intelligent extraction layer determines the best text-recovery method:
*   **Digital PDFs:** Parsed at high speed using `PyPDF2`.
*   **Word Documents:** Parsed natively using `python-docx`.
*   **Scanned/Handwritten PDFs:** If digital parsing yields fewer than 80 characters, the engine automatically falls back to an intensive OCR pipeline. It uses `pdf2image` to rasterize the document, applies OpenCV image preprocessing (grayscaling, adaptive thresholding, deskewing, and morphological closing), and extracts text via Tesseract OCR (`pytesseract`).

### 2. The Ensemble Similarity Engine
Instead of relying on a single scoring mechanism, Plag Detect scores pairwise document similarities via three independent lenses:
*   **Lexical Analysis (TF-IDF & Bigrams):** Catches exact phrasing and literal copy-pasting. Uses sublinear term-frequency scaling to prevent common words from dominating the score.
*   **Semantic Analysis (Sentence-Transformers MiniLM):** Projects the documents into a 384-dimensional dense vector space. By measuring the Cosine Similarity of these embeddings, the engine detects *paraphrasing* and conceptual overlap where different words are used to express the same original meaning.
*   **Structural Analysis (Character N-Grams):** Calculates the Jaccard index on character 2-grams to catch obfuscation, minor typos (or OCR artifacts), and structural copying.

### 3. The Front-End Output
Calculated scores are proportionally weighted (40% TF-IDF, 45% Semantic, 15% N-Gram). The frontend dynamically updates using a modern Dark Glassmorphism UI, ranking documents asynchronously, marking severity (🔴 High, 🟡 Medium, 🟢 Low), and rendering expandable text previews tagging the origin of the text extraction (`[DIGITAL]`, `[DOCX]`, `[OCR]`).

---

## 🛠 Technologies Used

### Backend & Machine Learning
*   **Python 3.13** (Core Logic)
*   **Flask & Werkzeug** (RESTful API & Server)
*   **scikit-learn** (TF-IDF Vectorization & Matrix Math)
*   **sentence-transformers / HuggingFace** (`all-MiniLM-L6-v2` semantic model)
*   **NumPy** (Dense matrix clipping and ensemble processing)

### Computer Vision & OCR
*   **OpenCV (`cv2`)** (Image Denoising, Deskewing, Thresholding)
*   **Tesseract OCR / `pytesseract`** (Optical Character Recognition)
*   **Pillow (PIL) & `pdf2image`** (PDF rasterization and image matrices)
*   **PyPDF2 & `python-docx`** (Native document parsing)

### Frontend Engine
*   **Vanilla HTML5 & JavaScript** (Asynchronous Fetch API & DOM Manipulation)
*   **CSS3** (Fluid layout, CSS Variables, Glassmorphism backdrop-filters, custom keyframe micro-animations)

---

## 🚀 Setup & Installation

### Prerequisites
1.  **Python 3.10+** (3.13 recommended)
2.  **Tesseract OCR** (Must be installed on your system)
    *   *Windows:* Install Tesseract to `C:\Program Files\Tesseract-OCR\tesseract.exe`.
    *   *Linux/Mac:* Adjust path in `text_extractor.py` or install globally via package manager (`sudo apt install tesseract-ocr`).
3.  **Poppler** (Required for `pdf2image` to convert PDFs to images). Ensure Poppler's `bin` directory is in your system PATH.

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/shashank-gt/Plagiarism-Checker.git
    cd Plagiarism-Checker
    ```

2.  **Install Python Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Server**
    ```bash
    python app.py
    ```

4.  **Access the Application**
    Open your browser and navigate to: `http://localhost:5000`

---

## 📂 Project Structure

```
Plagiarism Detection/
├── app.py                   # Flask server, Routing, API Controller
├── plagiarism_engine.py     # NLP logic (TF-IDF, Neural Embeddings, Jaccard)
├── text_extractor.py        # PDF extraction & OpenCV/Tesseract processing
├── generate_samples.py      # Local script to create 4 test .docx files
├── requirements.txt         # Explicit Python dependencies
└── frontend/
    ├── index.html           # Main web interface
    ├── style.css            # Custom CSS & Glassmorphic Variables
    ├── app.js               # UI logic (Dragging, Validating, Fetching)
    └── samples/             # Auto-generated sample files for testing
```

---

## 🧪 Testing the Engine (Live Samples)

To quickly test the capabilities of the engine without needing to manually author plagiarism:
1. Run `python generate_samples.py`. This will create 4 DOCX files in `frontend/samples/`:
   * `Sample_A_Original.docx` - A baseline document about AI.
   * `Sample_B_CopyPaste.docx` - 100% plagiarized text.
   * `Sample_C_Paraphrased.docx` - Heavy semantic paraphrasing.
   * `Sample_D_Different.docx` - Completely unrelated document regarding photosynthesis.
2. Open the Web UI and hit the **Load Samples** button next to the browse file button. This will automatically inject the samples into the queue and trigger the engine.

---

*This project was engineered for scalable, high-accuracy deployment capable of assessing both structured digital documents and raw physical submissions.*
