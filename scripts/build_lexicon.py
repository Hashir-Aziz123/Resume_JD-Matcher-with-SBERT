import pandas as pd
import json
import re
import logging
from pathlib import Path
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def clean_tag(text: str) -> str:
    """Matches your exact model training distribution."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\+\#\.]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def build_lexicon(csv_path: str, output_path: str):
    logging.info(f"Loading raw SO synonyms from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        stop_words = set(stopwords.words('english'))
        # Add a few common false positives that slip past NLTK
        custom_stops = {"object", "time", "limit", "change", "features"}
        stop_words.update(custom_stops)
        
        lexicon = {}
        valid_pairs = 0

        for _, row in df.iterrows():
            alias_raw = row['SourceTagName']
            standard_raw = row['TargetTagName']

            alias_clean = clean_tag(alias_raw)
            standard_clean = clean_tag(standard_raw)

            # THE FIX: Stopword and Length Hygiene
            if not alias_clean or not standard_clean:
                continue
            if alias_clean in stop_words or standard_clean in stop_words:
                continue
            if len(alias_clean) < 2 or len(standard_clean) < 2:
                # Keep C and R, drop meaningless single letters
                if alias_clean not in {"c", "r"} and standard_clean not in {"c", "r"}:
                    continue

            if standard_clean not in lexicon:
                lexicon[standard_clean] = []
            
            if alias_clean not in lexicon[standard_clean] and alias_clean != standard_clean:
                lexicon[standard_clean].append(alias_clean)
                valid_pairs += 1

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(lexicon, f, indent=4)
            
        logging.info(f"Mapped {len(lexicon)} Standard Tech Skills to {valid_pairs} Aliases.")
        logging.info(f"Saved hygienic dictionary to {output_path}")

    except Exception as e:
        logging.error(f"Failed to build lexicon: {e}")

if __name__ == "__main__":
    build_lexicon("data/raw/tag_synonyms.csv", "data/processed/tech_lexicon.json")