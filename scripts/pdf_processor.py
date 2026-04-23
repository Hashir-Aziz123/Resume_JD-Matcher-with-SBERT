import pdfplumber
import fitz  # PyMuPDF
import re
import logging
from pathlib import Path

# Configure logging for pipeline monitoring
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def clean_text(text: str) -> str:
    """
    Sanitizes raw PDF text. Order of operations is critical here.
    """
    if not text:
        return ""

    # 1. Lowercase early to simplify regex matching
    text = text.lower()

    # 2. Strip PII (Personally Identifiable Information)
    # Remove URLs and Emails
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove Phone Numbers (Catches standard NA and international formats)
    text = re.sub(r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', ' ', text)

    # 3. Aggressive Character Filtering
    # We keep lowercase letters, numbers, spaces, and specifically +, #, and . 
    # This ensures "c++", "c#", and "node.js" survive the purge. Everything else becomes a space.
    text = re.sub(r'[^a-z0-9\s\+\#\.]', ' ', text)

    # 4. Whitespace Normalization
    # Collapse tabs, newlines, and multiple spaces into a single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def extract_text(pdf_path: str) -> str:
    """
    Two-stage PDF extraction cascade.
    Attempts pdfplumber first, falls back to PyMuPDF on failure or low word count.
    """
    filepath = Path(pdf_path)
    if not filepath.exists():
        logging.error(f"File not found: {pdf_path}")
        return ""

    raw_text = ""

    # STAGE 1: The pdfplumber Engine
    try:
        with pdfplumber.open(filepath) as pdf:
            # Extract text from all pages, filtering out None values
            pages = [page.extract_text() for page in pdf.pages if page.extract_text() is not None]
            raw_text = " ".join(pages)

        word_count = len(raw_text.split())
        
        # If we got a healthy amount of text, assume standard layout and return
        if word_count > 100:
            logging.info(f"pdfplumber extraction successful [{word_count} words]: {filepath.name}")
            return clean_text(raw_text)
        else:
            logging.warning(f"pdfplumber yielded only {word_count} words. Triggering PyMuPDF fallback for {filepath.name}")

    except Exception as e:
        logging.warning(f"pdfplumber failed ({e}). Triggering PyMuPDF fallback for {filepath.name}")

    # STAGE 2: The PyMuPDF (fitz) Fallback
    try:
        raw_text = ""
        doc = fitz.open(filepath)
        
        for page in doc:
            # 'blocks' extraction natively guesses columns and paragraphs
            blocks = page.get_text("blocks")
            
            # Sort blocks by vertical position (y0), then horizontal (x0)
            # This attempts to preserve logical reading order in chaotic layouts
            blocks.sort(key=lambda b: (b[1], b[0]))
            
            for b in blocks:
                # b[6] represents the block type. Type 0 is text.
                if b[6] == 0:  
                    raw_text += b[4] + " "

        word_count = len(raw_text.split())
        logging.info(f"PyMuPDF fallback finished [{word_count} words]: {filepath.name}")
        return clean_text(raw_text)

    except Exception as e:
        logging.error(f"Complete pipeline failure on {filepath.name}. Error: {e}")
        return ""
