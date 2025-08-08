import os
from tqdm import tqdm
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

# Set up source folder
SOURCE_DIR = os.path.join("data", "reports_corrupted")

# Optional: Set path to tesseract binary if not in PATH
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Linux
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows

def ocr_pdf_to_text(pdf_path: str) -> str:
    """
    Convert PDF to images, then apply OCR to extract text.
    Returns the extracted text as a string.
    """
    text = ""
    try:
        # Convert PDF to list of images (one per page)
        images = convert_from_path(pdf_path, dpi=300)
        print("Converted pages to images")
        for img in tqdm(images, desc=f"Extracting text from {pdf_path}"):
            page_text = pytesseract.image_to_string(img)
            text += page_text + "\n"
    except Exception as e:
        print(f"Failed to OCR {pdf_path}: {e}")
    return text

def process_all_pdfs(source_dir: str):
    for root, _, files in os.walk(source_dir):
        for file in tqdm(files, desc=f"Processing PDFs in {root}"):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                txt_path = os.path.join(root, file.replace(".pdf", ".txt"))

                # if os.path.exists(txt_path):
                #     continue  # Skip already processed

                print(f"OCR processing: {pdf_path}")
                text = ocr_pdf_to_text(pdf_path)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Saved OCR output: {txt_path}")

if __name__ == "__main__":
    process_all_pdfs(SOURCE_DIR)
