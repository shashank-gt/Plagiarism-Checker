import os
import PyPDF2
import pandas as pd
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import subprocess

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
folder_path = "submissions"
OUTPUT_CSV = "plagiarism_results.csv"
HIGH_SIM_THRESHOLD = 80.0  # percent

def preprocess_image(pil_img):
    img = np.array(pil_img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img.shape
    if max(h, w) < 1000:
        scale = int(1000 / max(h, w)) + 1
        img = cv2.resize(img, (w*scale, h*scale), interpolation=cv2.INTER_LINEAR)
    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(img)

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    if len(text.strip()) < 50:
        ocr_text = extract_text_using_ocr(file_path)
        if ocr_text.strip():
            return ocr_text
    return text

def extract_text_using_ocr(file_path):
    ocr_text = ""
    try:
        pages = convert_from_path(file_path, dpi=200)
        for page in pages:
            proc = preprocess_image(page)
            txt = pytesseract.image_to_string(proc, config='--oem 1 --psm 3')
            ocr_text += txt + "\n"
    except Exception as e:
        print(f"OCR failed for {file_path}: {e}")
    return ocr_text

def load_pdfs_from_folder(folder):
    if not os.path.exists(folder):
        print(f"\nFolder '{folder}' not found. Please create it and add PDFs.")
        return {}
    files = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    if not files:
        print(f"\nNo PDF files found in '{folder}' folder.")
        return {}
    texts = {}
    print("\n[INFO] Extracting text from PDFs...")
    for file in tqdm(sorted(files)):
        texts[file] = extract_text_from_pdf(os.path.join(folder, file))
    return texts

def compute_similarity(texts):
    print("\n[INFO] Calculating similarity...")
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts.values())
    similarity = cosine_similarity(tfidf_matrix)
    return similarity

def save_results(files, similarity):
    results = []
    print("\n[INFO] Generating report...")
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            sim_percent = round(similarity[i][j] * 100, 2)
            results.append({
                "File 1": files[i],
                "File 2": files[j],
                "Similarity (%)": sim_percent
            })
    df = pd.DataFrame(results)
    df.sort_values(by="Similarity (%)", ascending=False, inplace=True)

    high_similar = df[df["Similarity (%)"] >= HIGH_SIM_THRESHOLD]
    summary_lines = []
    if not high_similar.empty:
        summary_lines.append("Highly Similar Submissions (Possible Plagiarism Detected):")
        for _, row in high_similar.iterrows():
            summary_lines.append(f"{row['File 1']} and {row['File 2']} -> {row['Similarity (%)']}%")
    else:
        summary_lines.append("No major plagiarism detected (all below 80%).")

    df.to_csv(OUTPUT_CSV, index=False)
    with open(OUTPUT_CSV, "a", encoding="utf-8") as f:
        f.write("\n\nSummary:\n")
        for line in summary_lines:
            f.write(line + "\n")

    print(f"\n[SUCCESS] Results saved to: {OUTPUT_CSV}")
    print("\n===== SUMMARY =====")
    for line in summary_lines:
        print(line)

    try:
        os.startfile(OUTPUT_CSV)
    except Exception:
        try:
            subprocess.run(["open", OUTPUT_CSV])
        except:
            try:
                subprocess.run(["xdg-open", OUTPUT_CSV])
            except:
                pass

def main():
    texts = load_pdfs_from_folder(folder_path)
    if not texts:
        return
    similarity = compute_similarity(texts)
    save_results(list(texts.keys()), similarity)

if __name__ == "__main__":
    main()
