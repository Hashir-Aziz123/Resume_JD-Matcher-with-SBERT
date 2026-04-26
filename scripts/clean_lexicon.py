import json
import os
import time
import logging
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

# 1. Load the .env file BEFORE doing anything else
load_dotenv()

# 2. Import the NEW SDK
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class SkillEvaluation(BaseModel):
    valid_tech_skills: List[str]

def clean_lexicon(input_path: str, output_path: str, batch_size: int = 100):
    logging.info(f"Loading corrupted lexicon from {input_path}...")
    
    with open(input_path, "r", encoding="utf-8") as f:
        raw_lexicon = json.load(f)
        
    all_keys = list(raw_lexicon.keys())
    logging.info(f"Total concepts to evaluate: {len(all_keys)}")

    # Initialize the new client. 
    # It will automatically find GEMINI_API_KEY in the environment now that dotenv is loaded.
    client = genai.Client()
    
    system_instruction = """
    You are an expert HR Technical Recruiter and Software Architect. 
    You will be given a list of terms scraped from the internet.
    Your job is to filter the list. ONLY return a term if it is a verifiable, 
    hard technical skill: a programming language, framework, software tool, 
    cloud service, or strict architectural standard (e.g., "python", "reactjs", "docker", "mvc").
    
    DO NOT include soft skills, general nouns, abstractions, or actions 
    (e.g., "user", "performance", "deployment", "agile", "fun", "history", "learning").
    """

    approved_skills = set()

    for i in range(0, len(all_keys), batch_size):
        batch = all_keys[i:i + batch_size]
        logging.info(f"Processing batch {i // batch_size + 1}/{(len(all_keys) // batch_size) + 1}...")
        
        prompt = f"{system_instruction}\n\nEvaluate the following terms:\n{json.dumps(batch)}"
        
        try:
            # New SDK syntax for generation and structured outputs
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=SkillEvaluation,
                    temperature=0.1
                ),
            )
            
            result = json.loads(response.text)
            approved_skills.update(result.get("valid_tech_skills", []))
            
            # Respect rate limits
            time.sleep(2)
            
        except Exception as e:
            logging.error(f"Batch failed. Skipping terms {i} to {i + batch_size}. Error: {e}")

    # Rebuild the dictionary with only the approved keys
    clean_lexicon = {k: raw_lexicon[k] for k in approved_skills if k in raw_lexicon}
    
    logging.info(f"Sterilization complete. Original count: {len(all_keys)}. New count: {len(clean_lexicon)}.")
    logging.info(f"Total junk entities nuked: {len(all_keys) - len(clean_lexicon)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_lexicon, f, indent=4)
        
    logging.info(f"Clean lexicon saved to {output_path}")

if __name__ == "__main__":
    # Make sure this points to wherever your actual file is located
    INPUT_FILE = "data/processed/tech_lexicon.json"
    OUTPUT_FILE = "data/processed/clean_tech_lexicon.json"
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Safety check so the script doesn't run if the key still isn't loaded
    if not os.environ.get("GEMINI_API_KEY"):
        logging.error("CRITICAL: GEMINI_API_KEY not found. Ensure your .env file is in the root directory.")
    else:
        clean_lexicon(INPUT_FILE, OUTPUT_FILE)