from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import sys
import os
import logging
import fitz  # PyMuPDF

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.pdf_processor import clean_text
from scripts.matcher import ResumeMatcher

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Executes exactly once when the server boots. 
    Loads the heavy PyTorch and spaCy models into RAM.
    """
    logging.info("Igniting the API Server...")
    logging.info("Loading ResumeMatcher Engine into memory. Please wait...")
    ml_models["matcher"] = ResumeMatcher()
    logging.info("Engine is hot. Ready for traffic.")
    yield
    ml_models.clear()

# Initialize the API
app = FastAPI(
    title="Resume Matcher AI Core",
    description="Semantic Matching Engine for Resumes and Job Descriptions",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Security note: In production, change this to your actual React URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MatchRequest(BaseModel):
    resume_text: str
    jd_text: str

# --- Helper Function for PDF Extraction ---
def extract_and_clean_pdf(file_bytes: bytes) -> str:
    """
    Reads PDF bytes from RAM, sorts the text blocks spatially, 
    and passes it through the exact Phase 1 training regex cleaner.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    raw_text = ""
    
    for page in doc:
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))
        for b in blocks:
            if b[6] == 0:  
                raw_text += b[4] + " "
                
    return clean_text(raw_text)

# --- API Endpoints ---

@app.get("/api/health")
async def health_check():
    """Simple ping to check if the backend is alive."""
    return {"status": "online", "model_loaded": "matcher" in ml_models}

@app.post("/api/match")
async def match_resume(request: MatchRequest):
    """
    The core inference endpoint. 
    Expects a JSON payload with 'resume_text' and 'jd_text'.
    """
    try:
        engine = ml_models.get("matcher")
        if not engine:
            raise HTTPException(status_code=503, detail="Model engine is offline.")
            
        result = engine.calculate_match(request.resume_text, request.jd_text)
        return result
        
    except Exception as e:
        logging.error(f"Inference crash: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during matching.")

@app.post("/api/match-pdf")
async def match_resume_pdf(
    resume_file: UploadFile = File(...), 
    jd_text: str = Form(...)
):
    """
    Accepts a single PDF file and JD text.
    """
    try:
        if not resume_file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF.")

        file_bytes = await resume_file.read()
        final_resume_text = extract_and_clean_pdf(file_bytes)
        final_jd_text = clean_text(jd_text)

        if len(final_resume_text.strip()) < 10:
            raise HTTPException(
                status_code=422, 
                detail="Extraction failed. Document may be an image-only PDF."
            )

        engine = ml_models.get("matcher")
        if not engine:
            raise HTTPException(status_code=503, detail="Model engine is offline.")
            
        result = engine.calculate_match(final_resume_text, final_jd_text)
        return result

    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"PDF Processing crash: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during PDF parsing.")

@app.post("/api/match-bulk")
async def match_bulk_resumes(
    resumes: List[UploadFile] = File(...),
    jd_text: str = Form(...)
):
    """
    Accepts multiple PDF resumes and a single Job Description.
    Returns a sorted array of candidate scores (leaderboard).
    """
    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="Job Description cannot be empty.")
    
    if not resumes:
        raise HTTPException(status_code=400, detail="No resumes uploaded.")

    engine = ml_models.get("matcher")
    if not engine:
        raise HTTPException(status_code=503, detail="Model engine is offline.")

    final_jd_text = clean_text(jd_text)
    results = []

    for file in resumes:
        if not file.filename.lower().endswith(".pdf"):
            results.append({
                "filename": file.filename,
                "error": "Invalid file type. Only PDFs are allowed."
            })
            continue
            
        try:
            file_bytes = await file.read()
            final_resume_text = extract_and_clean_pdf(file_bytes)
            
            if len(final_resume_text.strip()) < 10:
                results.append({
                    "filename": file.filename,
                    "error": "Extraction failed. Document may be an image-only PDF."
                })
                continue
            
            # The JD embedding will be automatically cached after the first iteration
            match_data = engine.calculate_match(final_resume_text, final_jd_text)
            
            # If the length guard in matcher.py caught it, it returns an error dict
            if "error" in match_data:
                results.append({
                    "filename": file.filename,
                    "error": match_data["message"]
                })
            else:
                results.append({
                    "filename": file.filename,
                    "match_data": match_data
                })
            
        except Exception as e:
            logging.error(f"Bulk processing error on {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "error": "Internal server error during processing."
            })
            
    # Sort successful matches descending by overall score
    successful_matches = [r for r in results if "error" not in r]
    failed_matches = [r for r in results if "error" in r]
    
    successful_matches.sort(
        key=lambda x: x["match_data"]["overall_match_percentage"], 
        reverse=True
    )

    return {
        "jd_processed": True,
        "total_candidates": len(resumes),
        "ranked_results": successful_matches,
        "failed_files": failed_matches
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)