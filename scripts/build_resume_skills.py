import os
import sys
import pandas as pd
from tqdm import tqdm
import time

# Dynamically add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.skill_extractor import SkillExtractor

def process_all_resumes():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_path = os.path.join(base_dir, "data", "raw", "Resume.csv")
    output_path = os.path.join(base_dir, "data", "processed", "resume_skills_extracted.csv")

    # Ensure processed directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading raw resume dataset from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("ERROR: Resume.csv not found.")
        return

    # Drop the known corrupted PDFs if they exist in the CSV
    df = df[~df['ID'].isin([15746146, 12632728, 14663897])]

    print("\nInitializing SkillExtractor (loading spaCy and ESCO)...")
    extractor = SkillExtractor()

    print(f"\nExtracting skills for all {len(df)} resumes...")
    
    # Store results
    extracted_skills_list = []
    
    start_time = time.time()
    
    for _, row in tqdm(df.iterrows(), total=len(df), unit="resume"):
        text = str(row['Resume_str'])
        skills_set = extractor.extract_skills(text)
        
        # Convert set to a comma-separated string for easy CSV storage
        extracted_skills_list.append(", ".join(list(skills_set)))

    df['Extracted_Skills'] = extracted_skills_list
    df['Skill_Count'] = df['Extracted_Skills'].apply(lambda x: len(x.split(", ")) if x else 0)

    # Save to processed folder
    df.to_csv(output_path, index=False)
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*50)
    print("✅ CORPUS EXTRACTION COMPLETE")
    print("="*50)
    print(f"Total Time: {elapsed_time:.2f} seconds")
    print(f"Saved to:   {output_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    process_all_resumes()