"""
text_extractor.py
-----------------
Handles text extraction from PDFs (typed + scanned/handwritten) and Word documents.
- Typed PDFs: PyPDF2
- Scanned/Handwritten PDFs: pdf2image + OpenCV preprocessing + Tesseract OCR
- Word (.docx): python-docx
"""

import os
import io
import numpy as np
import cv2
import PyPDF2
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
import docx

# ─── Tesseract Path (Windows) ───────────────────────────────────────────────
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

OCR_THRESHOLD = 80  # characters; below this we fall back to OCR


# ─── Image Preprocessing ─────────────────────────────────────────────────────

def preprocess_image(pil_img: Image.Image) -> Image.Image:
    """
    Enhance a PIL image for better OCR accuracy:
      1. Convert to grayscale
      2. Upscale if too small
      3. Deskew
      4. Denoise
      5. Adaptive thresholding (better than Otsu for handwriting)
    """
    img = np.array(pil_img)

    # Grayscale
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Upscale for small images
    h, w = img.shape
    if max(h, w) < 1500:
        scale = max(2, int(1500 / max(h, w)))
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # Deskew
    img = _deskew(img)

    # Denoise
    img = cv2.fastNlMeansDenoising(img, None, h=15, templateWindowSize=7, searchWindowSize=21)

    # Adaptive threshold (works better than Otsu for mixed/handwritten content)
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=10
    )

    # Morphological closing to connect broken strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return Image.fromarray(img)


def _deskew(img: np.ndarray) -> np.ndarray:
    """Detect and correct skew angle using Hough lines."""
    try:
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                                minLineLength=100, maxLineGap=10)
        if lines is None:
            return img
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if -45 < angle < 45:
                    angles.append(angle)
        if not angles:
            return img
        median_angle = np.median(angles)
        if abs(median_angle) < 0.5:
            return img
        h, w = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return img


# ─── Extraction Functions ─────────────────────────────────────────────────────

def extract_from_pdf_bytes(file_bytes: bytes, filename: str) -> dict:
    """
    Extract text from a PDF given its raw bytes.
    Returns: {"text": str, "method": "digital"|"ocr", "pages": int}
    """
    result = {"text": "", "method": "digital", "pages": 0, "filename": filename}

    # --- Try digital text extraction first ---
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        result["pages"] = len(reader.pages)
        digital_text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                digital_text += t + "\n"

        if len(digital_text.strip()) >= OCR_THRESHOLD:
            result["text"] = _clean_text(digital_text)
            result["method"] = "digital"
            return result
    except Exception as e:
        print(f"[WARN] PyPDF2 failed for {filename}: {e}")

    # --- Fall back to OCR ---
    result["method"] = "ocr"
    try:
        pages = convert_from_bytes(file_bytes, dpi=250)
        result["pages"] = len(pages)
        ocr_text = ""
        for page_img in pages:
            enhanced = preprocess_image(page_img)
            # Try both LSTM and legacy engines for robustness
            txt = pytesseract.image_to_string(
                enhanced,
                config="--oem 1 --psm 3 -l eng"
            )
            ocr_text += txt + "\n"
        result["text"] = _clean_text(ocr_text)
    except Exception as e:
        print(f"[ERROR] OCR failed for {filename}: {e}")
        result["text"] = ""

    return result


def extract_from_docx_bytes(file_bytes: bytes, filename: str) -> dict:
    """Extract text from a .docx Word document given its raw bytes."""
    result = {"text": "", "method": "docx", "pages": 0, "filename": filename}
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        result["text"] = _clean_text("\n".join(paragraphs))
        result["pages"] = max(1, len(paragraphs) // 30)
    except Exception as e:
        print(f"[ERROR] DOCX extraction failed for {filename}: {e}")
    return result


def extract_text(file_bytes: bytes, filename: str) -> dict:
    """Route extraction based on file extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return extract_from_pdf_bytes(file_bytes, filename)
    elif ext in (".docx", ".doc"):
        return extract_from_docx_bytes(file_bytes, filename)
    else:
        return {"text": "", "method": "unsupported", "pages": 0, "filename": filename}


# ─── Text Cleaning ────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Normalize whitespace, remove control characters, collapse blank lines."""
    import re
    # Remove non-printable except newlines
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
