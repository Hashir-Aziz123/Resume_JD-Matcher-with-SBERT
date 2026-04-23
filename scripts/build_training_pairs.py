import os
import sys
import pandas as pd
import random
from tqdm import tqdm

# Dynamically add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.skill_extractor import SkillExtractor

def compute_jaccard(skills1: str, skills2: str) -> float:
    """Calculates the Jaccard similarity between two comma-separated skill strings."""
    s1 = set(skills1.split(', ')) if pd.notna(skills1) and skills1 else set()
    s2 = set(skills2.split(', ')) if pd.notna(skills2) and skills2 else set()
    
    if not s1 and not s2:
        return 0.0
    
    intersection = s1.intersection(s2)
    union = s1.union(s2)
    
    return len(intersection) / len(union) if len(union) > 0 else 0.0

def build_dataset():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    resume_csv = os.path.join(base_dir, "data", "processed", "resume_skills_extracted.csv")
    jd_csv = os.path.join(base_dir, "data", "raw", "postings.csv")
    output_path = os.path.join(base_dir, "data", "processed", "training_pairs.csv")

    print("Loading processed resumes...")
    try:
        resumes_df = pd.read_csv(resume_csv)
        # Drop resumes with 0 skills as they can't form meaningful positive pairs
        resumes_df = resumes_df[resumes_df['Skill_Count'] > 0].copy() 
    except FileNotFoundError:
        print("ERROR: resume_skills_extracted.csv not found. Run Phase 3 first.")
        return

    print("Loading LinkedIn Job Postings (this is a big file)...")
    try:
        jds_df = pd.read_csv(jd_csv, usecols=['description', 'title'])
        jds_df = jds_df.dropna(subset=['description'])
    except FileNotFoundError:
        print("ERROR: postings.csv not found. Check your data/raw directory.")
        return
    
    # We only need a subset of JDs to create enough pairs. 
    # Processing 2,500 JDs will take about ~3 minutes and give us millions of permutation options.
    sample_jds = jds_df.sample(n=2500, random_state=42).reset_index(drop=True)

    print("\nInitializing SkillExtractor for Job Descriptions...")
    extractor = SkillExtractor()
    
    jd_skills_list = []
    print("Extracting skills from 2,500 Job Descriptions...")
    for _, row in tqdm(sample_jds.iterrows(), total=len(sample_jds), unit="JD"):
        text = str(row['description'])
        skills_set = extractor.extract_skills(text)
        jd_skills_list.append(", ".join(list(skills_set)))
        
    sample_jds['Extracted_Skills'] = jd_skills_list
    # Filter out JDs that didn't yield any skills
    sample_jds = sample_jds[sample_jds['Extracted_Skills'] != ""]

    print("\nColliding Resumes and JDs to build balanced training pairs...")
    
    strong_positives = [] # Label 1.0 (Jaccard > 0.25)
    weak_positives = []   # Label 0.6 (Jaccard 0.10 - 0.25)
    negatives = []        # Label 0.0 (Jaccard < 0.10)
    
    # Target dataset size: 1500 of each class for a perfectly balanced 4500-pair dataset
    TARGET_PER_CLASS = 1500
    
    resume_records = resumes_df.to_dict('records')
    jd_records = sample_jds.to_dict('records')
    
    # Brute-force random collisions until buckets are full
    attempts = 0
    pbar = tqdm(total=TARGET_PER_CLASS * 3, desc="Generating Pairs")
    
    while True:
        attempts += 1
        res = random.choice(resume_records)
        jd = random.choice(jd_records)
        
        score = compute_jaccard(res['Extracted_Skills'], jd['Extracted_Skills'])
        
        pair_data = {
            "resume_text": res['Resume_str'],
            "jd_text": jd['description'],
            "resume_skills": res['Extracted_Skills'],
            "jd_skills": jd['Extracted_Skills'],
            "jaccard_score": round(score, 4),
            "label": 0.0
        }
        
        if score > 0.25 and len(strong_positives) < TARGET_PER_CLASS:
            pair_data["label"] = 1.0
            strong_positives.append(pair_data)
            pbar.update(1)
        elif 0.10 <= score <= 0.25 and len(weak_positives) < TARGET_PER_CLASS:
            pair_data["label"] = 0.6
            weak_positives.append(pair_data)
            pbar.update(1)
        elif score < 0.10 and len(negatives) < TARGET_PER_CLASS:
            pair_data["label"] = 0.0
            negatives.append(pair_data)
            pbar.update(1)
            
        # Break if all buckets are full
        if len(strong_positives) == TARGET_PER_CLASS and \
           len(weak_positives) == TARGET_PER_CLASS and \
           len(negatives) == TARGET_PER_CLASS:
            break
            
        # Safety valve so we don't get stuck in an infinite loop if thresholds are too strict
        if attempts > 1000000:
            print("\nWARNING: Hit collision cap. Stopping early.")
            break

    pbar.close()
    
    # Combine and shuffle
    all_pairs = strong_positives + weak_positives + negatives
    random.shuffle(all_pairs)
    
    final_df = pd.DataFrame(all_pairs)
    final_df.to_csv(output_path, index=False)
    
    print("\n" + "="*50)
    print("✅ DATASET CONSTRUCTION COMPLETE")
    print("="*50)
    print(f"Strong Positives (1.0): {len(strong_positives)}")
    print(f"Weak Positives (0.6):   {len(weak_positives)}")
    print(f"Negatives (0.0):        {len(negatives)}")
    print(f"Total Pairs Saved:      {len(final_df)}")
    print(f"Saved to:               {output_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    build_dataset()