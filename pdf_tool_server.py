# pdf_tool_server.py
import os
import shutil
import io
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
from PIL import Image
import pytesseract
from shared_models import ProcessRequest, ProcessResponse 
import logging

# --- Configuration ---
PDF_DIR = "./pdfs"
IMG_DIR = os.path.join(PDF_DIR, "images")
THUMB_DIR = os.path.join(PDF_DIR, "thumbs")
DEBUG_DIR = os.path.join(PDF_DIR, "debug")
for d in (PDF_DIR, IMG_DIR, THUMB_DIR, DEBUG_DIR): os.makedirs(d, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(DEBUG_DIR, "pdf_tool.log"), 
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def log_workflow(event: str, **kwargs):
    logging.info(f"{event}: " + ", ".join(f"{k}={v}" for k,v in kwargs.items()))


app = FastAPI()


def extract_images(pdf_path: str) -> dict:
    doc = fitz.open(pdf_path)
    images = {}
    for i, page in enumerate(doc, start=1):
        img_list = []
        for img_index, img in enumerate(page.get_images(full=True)):
            base = doc.extract_image(img[0])
            ext = base['ext']; data = base['image']
            fname = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page{i}_img{img_index}.{ext}"
            fpath = os.path.join(IMG_DIR, fname)
            with open(fpath, "wb") as f: f.write(data)
            thumb = Image.open(io.BytesIO(data))
            thumb.thumbnail((640, 640))
            tfname = f"{os.path.splitext(fname)[0]}_thumb.jpg"
            tpath = os.path.join(THUMB_DIR, tfname)
            thumb.save(tpath, format='JPEG', quality=75)
            img_list.append(tpath)
        images[i] = img_list
    return images


def extract_tables(pdf_path: str) -> dict:
    tables = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            md_list = []
            for table in page.extract_tables():
                df = pd.DataFrame(table[1:], columns=table[0])
                md_list.append(df.to_markdown(index=False))
            tables[page_num] = md_list
    return tables
    

import subprocess
def convert_pdf(pdf_path: str) -> (str, dict, dict):
    # Log markitdown availability
    logging.info(f"markitdown exist or not? {shutil.which('markitdown')}")
    log_workflow("convert_pdf:start", pdf=os.path.basename(pdf_path))
    
    md_path = pdf_path.replace('.pdf', '.md')
    md_text = ''
    # 1. Try MarkItDown with -o to ensure full output
    if shutil.which('markitdown'):
        res = subprocess.run(
            ["markitdown", pdf_path, "-o", md_path],
            capture_output=True
        )
        if res.returncode == 0 and os.path.exists(md_path):
            with open(md_path, 'r', encoding='utf-8', errors='ignore') as f:
                md_text = f.read()
    
    # 2. Extract images and tables
    images = extract_images(pdf_path)
    tables = extract_tables(pdf_path)
    doc = fitz.open(pdf_path)
    pages_md = []
    basename = os.path.basename(pdf_path)
    
    for i, page in enumerate(doc, start=1):
        text_md = ''
        # 2a. Attempt to split original MD by page marker
        if md_text:
            parts = md_text.split(f"<!-- Page {i} -->")
            if len(parts) > 1:
                text_md = parts[1].split("<!-- Page")[0]
                # Save debug original MD per page
                open(os.path.join(DEBUG_DIR, f"{basename}_page{i}_md.md"), 'w').write(text_md)
        
        # 2b. Fallback to PyMuPDF text extraction if no MD content
        if not text_md.strip():
            text_md = page.get_text("text")
        
        # 2c. Only if still too short, do OCR
        if len(text_md.strip()) < 50:
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # Save debug image
            dbg_img = os.path.join(DEBUG_DIR, f"{basename}_page{i}_dbg.png")
            img.save(dbg_img)
            # OCR
            text_md = pytesseract.image_to_string(img, lang="chi_tra+eng")
            print(text_md)
            open(os.path.join(DEBUG_DIR, f"{basename}_page{i}_ocr.txt"), 'w').write(text_md)
            # OCR bounding boxes
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)
            data.to_csv(os.path.join(DEBUG_DIR, f"{basename}_page{i}_ocr_boxes.csv"), index=False)
        
        # 3. Append page marker, text, and tables
        part = f"<!-- Page {i} -->\n" + text_md
        for tbl in tables.get(i, []):
            part += "\n" + tbl
        pages_md.append(part)
    
    # Write final combined markdown
    final = "\n\n".join(pages_md)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(final)
    log_workflow("convert_pdf:done", md_path=basename + '.md')
    return md_path, images, tables


@app.post("/process_pdf", response_model=ProcessResponse)
async def process_pdf_endpoint(req: ProcessRequest):
    """Takes a PDF path and processes it into Markdown, images, and tables."""
    if not os.path.exists(req.pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    try:
        md_path, images, tables = convert_pdf(req.pdf_path)
        return ProcessResponse(md_path=md_path, images=images, tables=tables)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)