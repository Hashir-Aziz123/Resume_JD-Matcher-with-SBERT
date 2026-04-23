from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import logging

import fitz  # PyMuPDF
from fastapi import File, UploadFile, Form
from scripts.pdf_processor import clean_text


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

@app.post("/api/match")
async def match_resume(request: MatchRequest):
    """
    The core inference endpoint. 
    Expects a JSON payload with 'resume_text' and 'jd_text'.
    Returns the final semantic score and gap report.
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

@app.get("/api/health")
async def health_check():
    """Simple ping to check if the backend is alive."""
    return {"status": "online", "model_loaded": "matcher" in ml_models}

@app.post("/api/match-pdf")
async def match_resume_pdf(
    resume_file: UploadFile = File(...), 
    jd_text: str = Form(...)
):
    """
    Accepts a raw PDF file and JD text. 
    Uses in-memory PyMuPDF and the exact Phase 1 regex cleaner 
    to guarantee identical data distribution for the SBERT model.
    """
    try:
        if not resume_file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF.")

        # Read the file bytes directly into RAM (Zero Disk I/O)
        file_bytes = await resume_file.read()
        
        # Open the PDF from the memory stream using PyMuPDF (fitz)
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        raw_text = ""
        
        # Execute your exact block sorting logic
        for page in doc:
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: (b[1], b[0]))
            for b in blocks:
                if b[6] == 0:  
                    raw_text += b[4] + " "

        # THE CRITICAL STEP: Sanitize using your Phase 1 function
        final_resume_text = clean_text(raw_text)
        
        # We MUST also clean the JD text so it matches the training logic!
        final_jd_text = clean_text(jd_text)

        if len(final_resume_text.strip()) < 10:
            raise HTTPException(
                status_code=422, 
                detail="Extraction failed. Document may be an image-only PDF."
            )

        # Pass the mathematically identical text to the deep learning engine
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
