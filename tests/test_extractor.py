import os
import sys
import pandas as pd
import time
from tqdm import tqdm

# Dynamically add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.skill_extractor import SkillExtractor

def run_extraction_benchmark(sample_size=100):
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "Resume.csv")
    
    print("Loading Kaggle dataset for extraction benchmark...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: Could not find {csv_path}")
        return

    # Drop our known casualties just in case they get sampled
    df = df[~df['ID'].isin([15746146, 12632728, 14663897])]
    
    # Sample a random subset to profile
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    print("\nInitializing SkillExtractor (loading spaCy and 13k ESCO skills)...")
    init_start = time.time()
    extractor = SkillExtractor()
    init_time = time.time() - init_start
    print(f"Initialization took: {init_time:.2f} seconds\n")

    extraction_times = []
    skill_counts = []
    empty_extractions = 0

    print(f"Running extraction on {len(sample_df)} real resumes...")
    
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), unit="resume"):
        text = str(row['Resume_str'])
        
        start_time = time.time()
        skills = extractor.extract_skills(text)
        end_time = time.time()
        
        extraction_times.append(end_time - start_time)
        skill_counts.append(len(skills))
        
        if len(skills) == 0:
            empty_extractions += 1

    # Calculate metrics
    avg_time = sum(extraction_times) / len(extraction_times)
    avg_skills = sum(skill_counts) / len(skill_counts)
    max_skills = max(skill_counts)
    
    print("\n" + "="*50)
    print("🚀 EXTRACTION BENCHMARK RESULTS")
    print("="*50)
    print(f"Average Time per Resume:   {avg_time:.4f} seconds")
    print(f"Average Skills Extracted:  {avg_skills:.1f} skills")
    print(f"Max Skills in One Resume:  {max_skills} skills")
    print(f"Zero-Skill Resumes:        {empty_extractions}")
    print("="*50 + "\n")

    # Let's peek at the highest scoring resume to ensure it's not hallucinating
    max_idx = skill_counts.index(max_skills)
    max_text = str(sample_df.iloc[max_idx]['Resume_str'])
    print(f"🔎 SANITY CHECK: Top Extracted Resume ({max_skills} skills)")
    print(f"Category: {sample_df.iloc[max_idx]['Category']}")
    print(f"Extracted: {extractor.extract_skills(max_text)}")

if __name__ == "__main__":
    run_extraction_benchmark(100)