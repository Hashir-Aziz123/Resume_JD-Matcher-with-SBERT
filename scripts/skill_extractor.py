import spacy
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class SkillExtractor:
    def __init__(self, lexicon_path: str = "data/processed/clean_tech_lexicon.json"):
        """
        Initializes the NLP model and loads the tech lexicon.
        Applies a stop-list and Maximal Munch sorting to prevent junk extraction.
        """
        logging.info("Loading spaCy NLP model...")
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        
        # The Domain Stop-List: The bouncer for StackOverflow's existential tagging crisis.
        self.junk_skills = {
            "user", "interface", "order", "warning", "warnings", "performance", "deployment", 
            "web", "learning", "stack", "project", "agile", "architecture", 
            "system", "application", "data", "error", "bug", "string", "list", "array",
            "fun", "questions", "history", "discussion", "behavior", "image", "disk", "jobs"
        }

        logging.info("Loading Tech Lexicon...")
        try:
            with open(lexicon_path, "r", encoding="utf-8") as f:
                raw_aliases = json.load(f)
            
            # Flatten and Sort by Length (Maximal Munch Preparation)
            self.search_patterns = []
            for standard_skill, aliases in raw_aliases.items():
                if standard_skill in self.junk_skills:
                    continue
                
                # Add the main skill
                self.search_patterns.append((standard_skill, standard_skill))
                
                # Add its aliases
                for alias in aliases:
                    if alias not in self.junk_skills:
                        self.search_patterns.append((alias, standard_skill))
            
            # Sort by the number of words in the search term, descending.
            # This ensures "user interface" is processed before "user" or "interface".
            self.search_patterns.sort(key=lambda x: len(x[0].split()), reverse=True)
            
            logging.info(f"Loaded and sorted {len(self.search_patterns)} strict tech patterns.")
        except Exception as e:
            logging.error(f"Lexicon missing or corrupted: {e}. Run build_lexicon.py first.")
            self.search_patterns = []

    def _lemmatize_text(self, text: str) -> str:
        doc = self.nlp(text.lower())
        return " ".join([token.lemma_ for token in doc if not token.is_space])

    def extract_skills(self, text: str) -> set:
        if not text:
            return set()

        # Pad the text so exact whole-word matching works on the edges
        processed_text = f" {self._lemmatize_text(text)} "
        extracted = set()

        # STRICT TECH OVERRIDE: Using Maximal Munch with Masking
        for search_term, standard_term in self.search_patterns:
            pattern = f" {search_term} "
            
            if pattern in processed_text:
                extracted.add(standard_term)
                # MASK the token so sub-words aren't double-counted
                # e.g., " user interface " becomes " [MATCHED] "
                processed_text = processed_text.replace(pattern, " [MATCHED] ")

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