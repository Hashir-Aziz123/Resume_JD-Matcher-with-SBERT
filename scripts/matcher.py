import os
import sys
import torch
import math
import re
from sentence_transformers import SentenceTransformer, util
import logging
from functools import lru_cache

# Dynamically add project root to path so we can import the extractor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.skill_extractor import SkillExtractor

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class ResumeMatcher:
    def __init__(self):
        """
        Initializes the deterministic skill extractor and loads the 
        fine-tuned deep learning model into memory.
        """
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_path = os.path.join(base_dir, "models", "sbert_finetuned")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Fine-tuned model not found at {model_path}")

        logging.info("Spinning up the deterministic Skill Extractor...")
        self.extractor = SkillExtractor()
        
        logging.info("Loading the fine-tuned SBERT Transformer into memory...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_path, device=self.device)
        logging.info(f"ResumeMatcher online. Compute device: {self.device.upper()}")

    @lru_cache(maxsize=128)
    def _process_jd(self, jd_text: str):
        """
        Memoizes the Job Description processing. 
        Turns an O(N) NLP extraction and Transformer encoding into an O(1) RAM lookup.
        """
        logging.info("Cache Miss: Processing new Job Description...")
        
        # 1. Extract JD Skills
        jd_skills = self.extractor.extract_skills(jd_text)
        jd_skills_str = ", ".join(list(jd_skills))
        
        # 2. Format inputs
        full_jd_input = f"SKILLS: {jd_skills_str} | CONTEXT: {str(jd_text)[:1000]}"
        ctx_jd_input = str(jd_text)[:1000]
        
        # 3. Encode to Tensors
        embeddings = self.model.encode(
            [full_jd_input, ctx_jd_input], 
            convert_to_tensor=True
        )
        return jd_skills, embeddings[0], embeddings[1]

    def calculate_match(self, resume_text: str, jd_text: str) -> dict:
        """
        Passes the documents through both the deterministic and semantic pipelines.
        """
        if not resume_text or not jd_text:
            return {"error": "Both Resume and JD text must be provided."}

        # FINAL FIX: The UX & Math Guard (Minimum Document Length)
        # Using regex to prevent comma-splicing bypasses
        word_count = len(re.findall(r'\w+', resume_text))
        if word_count < 50:
            logging.warning(f"Rejected document: Too short ({word_count} words).")
            return {
                "error": "resume_too_short",
                "message": f"Your resume only contains {word_count} words. Please upload a more detailed document (minimum 50 words)."
            }

        # 1. Extract Resume Skills (Must be done per resume)
        resume_skills = self.extractor.extract_skills(resume_text)
        
        # 2. Fetch Cached JD Data (Instant O(1) lookup)
        jd_skills, jd_full_emb, jd_ctx_emb = self._process_jd(jd_text)
        
        # 3. Fast In-Memory Gap Math (Bypasses redundant NLP extraction)
        matched_skills = resume_skills.intersection(jd_skills)
        missing = jd_skills.difference(resume_skills)
        bonus = resume_skills.difference(jd_skills)
        match_rate = len(matched_skills) / len(jd_skills) if jd_skills else 0.0
        
        gap_report = {
            "match_rate": round(match_rate, 2),
            "matched": list(matched_skills),
            "missing": list(missing),
            "bonus": list(bonus)
        }

        # 4. Format Resume Inputs (Only injecting matched skills to prevent dilution)
        res_skills_str = ", ".join(matched_skills)
        full_res_input = f"SKILLS: {res_skills_str} | CONTEXT: {str(resume_text)[:1000]}"
        
        # Context-Only: Raw text, no prefix scaffolding to prevent Train-Serve Skew
        ctx_res_input = str(resume_text)[:1000]

        # 5. Tensor Embedding (Only processing the Resume now)
        res_embeddings = self.model.encode(
            [full_res_input, ctx_res_input], 
            convert_to_tensor=True
        )
        
        # 6. Symmetric Cosine Similarity
        sim_full = util.cos_sim(res_embeddings[0], jd_full_emb).item()
        sim_context = util.cos_sim(res_embeddings[1], jd_ctx_emb).item()

        # 7. Deterministic Score
        hard_skill_score = gap_report["match_rate"] * 100

        # 8. The Semantic Coherence Blend & Soft Scaler
        blended_semantic = (0.65 * sim_full) + (0.35 * sim_context)
        
        lower_bound = 0.30
        upper_bound = 0.80
        
        if 0.10 < blended_semantic < lower_bound:
            logging.debug(f"Distribution Drift Alert: Score {blended_semantic:.3f} hit the floor.")
            
        raw_scale = (blended_semantic - lower_bound) / (upper_bound - lower_bound)
        normalized_semantic = max(0.0, min(1.0, raw_scale)) * 100

        base_final_score = (normalized_semantic * 0.7) + (hard_skill_score * 0.3)

        # 9. Advanced Statistical Density Guard (Regex + Sqrt)
        safe_words = max(word_count, 1)
        matched_count = len(matched_skills)
        density = matched_count / math.sqrt(safe_words)
        
        penalty_multiplier = 1.0
        if density > 1.5:  
            penalty_multiplier = 0.50
        elif density > 1.2: 
            penalty_multiplier = 0.85

        final_score = base_final_score * penalty_multiplier

        return {
            "overall_match_percentage": round(final_score, 2),
            "semantic_confidence": round(blended_semantic, 4), 
            "coherence_gap": round(sim_full - sim_context, 4),
            "density_penalty_applied": penalty_multiplier < 1.0,
            "skill_gap_report": gap_report,
            "dealbreakers": self.extractor.check_dealbreakers(resume_text, jd_text)
        }

if __name__ == "__main__":
    # Test Harness to verify the cache and short document guard
    matcher = ResumeMatcher()
    
    print("\n" + "="*60)
    print("🔥 PRODUCTION BACKEND TEST SUITE 🔥")
    print("="*60)

    sample_jd = "Looking for a Senior Python Developer with AWS and Docker experience to build backend services."
    
    # Resume 1: Too short (will trigger the 50-word guard)
    sample_resume_short = "I am a Python Developer with AWS experience. Hire me."
    
    # Resume 2: Normal length, good match
    sample_resume_long = """
    I am an experienced Python Developer with over 5 years in the tech industry. 
    Throughout my career, I have specialized in building robust backend services, 
    designing scalable APIs, and managing cloud infrastructure. My primary cloud 
    provider is AWS, where I have deployed numerous serverless functions and EC2 
    instances. I also have deep expertise in containerizing applications using 
    Docker, orchestrating CI/CD pipelines, and ensuring high availability. I love 
    working on complex backend challenges and collaborating with cross-functional teams.
    """

    print("\n--- Request 1: The Haiku Resume ---")
    res_short = matcher.calculate_match(sample_resume_short, sample_jd)
    print(res_short)
    
    print("\n--- Request 2: The Real Candidate (Cache Miss expected) ---")
    res_long = matcher.calculate_match(sample_resume_long, sample_jd)
    print(f"Match Score: {res_long.get('overall_match_percentage')}%")
    
    print("\n--- Request 3: Another Real Candidate (Cache HIT expected) ---")
    res_cache = matcher.calculate_match(sample_resume_long + " Extra words.", sample_jd)
    print(f"Match Score: {res_cache.get('overall_match_percentage')}%")