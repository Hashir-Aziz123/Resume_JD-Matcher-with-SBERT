import spacy
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class SkillExtractor:
    def __init__(self, lexicon_path: str = "data/processed/tech_lexicon.json"):
        """
        Initializes the NLP model and loads the community-vetted SEDE tech lexicon.
        (ESCO has been removed to specialize this engine for Tech/Engineering).
        """
        logging.info("Loading spaCy NLP model...")
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        
        logging.info("Loading Stack Overflow Tech Lexicon...")
        try:
            with open(lexicon_path, "r", encoding="utf-8") as f:
                self.tech_aliases = json.load(f)
            logging.info(f"Loaded {len(self.tech_aliases)} standard tech concepts.")
        except Exception as e:
            logging.error(f"Lexicon missing or corrupted: {e}. Run build_lexicon.py first.")
            self.tech_aliases = {}

    def _lemmatize_text(self, text: str) -> str:
        doc = self.nlp(text.lower())
        return " ".join([token.lemma_ for token in doc if not token.is_space])

    def extract_skills(self, text: str) -> set:
        if not text:
            return set()

        processed_text = self._lemmatize_text(text)
        padded_text = f" {processed_text} " 
        extracted = set()

        # STRICT TECH OVERRIDE: Check the Stack Overflow skills exclusively
        for standard_skill, aliases in self.tech_aliases.items():
            if f" {standard_skill} " in padded_text:
                extracted.add(standard_skill)
                continue
            
            for alias in aliases:
                if f" {alias} " in padded_text:
                    extracted.add(standard_skill)
                    break

        return extracted

    def skill_gap_report(self, resume_text: str, jd_text: str) -> dict:
        resume_skills = self.extract_skills(resume_text)
        jd_skills = self.extract_skills(jd_text)

        matched = resume_skills.intersection(jd_skills)
        missing = jd_skills.difference(resume_skills)
        bonus = resume_skills.difference(jd_skills)
        
        match_rate = len(matched) / len(jd_skills) if jd_skills else 0.0

        return {
            "match_rate": round(match_rate, 2),
            "matched": list(matched),
            "missing": list(missing),
            "bonus": list(bonus)
        }

    def check_dealbreakers(self, resume_text: str, jd_text: str) -> list:
        dealbreakers = [
            "security clearance", "top secret", "ts/sci", "active clearance",
            "phd", "doctorate", "pmp", "md",
            "on-site only", "no remote", "relocation required",
            "us citizen", "no visa sponsorship"
        ]
        
        flags = []
        resume_lower = resume_text.lower()
        jd_lower = jd_text.lower()
        
        for keyword in dealbreakers:
            if keyword in jd_lower and keyword not in resume_lower:
                flags.append(keyword)
                
        return flags