# ATS Semantic Engine

## Overview

A high-performance inference pipeline for evaluating candidate resumes against job descriptions. This system utilizes a hybrid deterministic-semantic architecture, combining rigid taxonomy-based extraction with deep learning document embeddings. The objective is to mitigate the flaws of traditional Applicant Tracking Systems (ATS) while actively penalizing keyword-stuffing exploits.

## System Architecture

### Frontend (Client)

* **Stack:** React, Vite, Tailwind v4.
* **Functionality:** Provides a dark-mode, command-center interface for submitting target parameters. Handles `multipart/form-data` for bulk PDF inference and JSON payloads for raw text evaluation.

### Backend (API)

* **Stack:** FastAPI, Python, Uvicorn.
* **Lifecycle:** Utilizes asynchronous lifespan management to load heavy PyTorch transformer models into RAM/VRAM at boot, preventing cold-start latency during requests.
* **Endpoints:**
* `POST /api/match-bulk`: Processes arrays of PDF files.
* `POST /api/match`: Processes raw text strings.



### Document Parsing

* **Engine:** PyMuPDF (`fitz`).
* **Implementation:** Executes spatial $(y, x)$ coordinate sorting during text extraction. This prevents multi-column layouts and sidebars from breaking the chronological reading order of the document.

## Inference Pipeline

The engine evaluates candidates across two distinct parallel tracks:

### 1. Deterministic Track (Hard Skills)

Utilizes `spaCy` (`en_core_web_sm`) alongside a strictly defined technical taxonomy.

* **Whitelist:** Explicitly extracts defined multi-word technical skills (e.g., "Random Forest", "Lean 4", "Docker") without token fragmentation.
* **Blacklist:** Aggressively filters corporate jargon and generic verbs (e.g., "build", "synergize", "monitor") to prevent artificial score inflation.

### 2. Semantic Track (Deep Learning)

Utilizes a fine-tuned `all-MiniLM-L6-v2` SBERT model to map documents into a 384-dimensional latent space.

* **Dual Embeddings:** Generates two vectors per document:
1. *Identity Embedding:* Scaffolded with a `SKILLS:` prefix to force transformer self-attention on technical keywords.
2. *Narrative Embedding:* Raw prose to evaluate natural language structure and professional context.


* **Cosine Similarity:** Calculates the angular distance between the candidate vectors and the target job description vectors to determine conceptual alignment.

## Scoring & Anti-Cheat Logic

The final match percentage is not a raw semantic output. It is calculated via a weighted fusion algorithm protected by statistical guardrails.

$$Base\ Score = (Normalized\ Semantic\ Confidence \times 0.7) + (Hard\ Skill\ Match\ Rate \times 0.3)$$

To prevent manipulation, the system implements a penalty multiplier based on two factors:

1. **Keyword Density:** Evaluates the ratio of matched skills to total word count using sub-linear scaling ($\sqrt{N}$) to normalize across varying document lengths.
2. **Coherence Gap:** The mathematical delta between the Identity Embedding and Narrative Embedding. A negative gap indicates unnatural text flow.

If extreme keyword density or a negative coherence gap is detected, the system applies a 50% penalty to the final score.

## Local Setup

**Prerequisites:** Python 3.10+, Node.js 18+

1. **Backend Initialization**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn main:app --reload --port 8000

```

2. **Model Deployment**
Ensure the fine-tuned SBERT model directory (`sbert_finetuned`) is placed within the `models/` directory at the project root.
3. **Frontend Initialization**

```bash
cd frontend
npm install
npm run dev

```
