import os
import pandas as pd
import numpy as np

def analyze_extraction_results():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_path = os.path.join(base_dir, "data", "processed", "resume_skills_extracted.csv")

    if not os.path.exists(csv_path):
        print(f"ERROR: Could not find {csv_path}. Did the extraction script finish?")
        return

    print("Loading processed dataset...")
    df = pd.read_csv(csv_path)

    # Global Statistics
    total_resumes = len(df)
    zero_skill_resumes = len(df[df['Skill_Count'] == 0])
    avg_skills = df['Skill_Count'].mean()
    median_skills = df['Skill_Count'].median()
    max_skills = df['Skill_Count'].max()

    print("\n" + "="*50)
    print("🌍 GLOBAL EXTRACTION STATISTICS")
    print("="*50)
    print(f"Total Resumes Processed:  {total_resumes}")
    print(f"Global Average Skills:    {avg_skills:.2f}")
    print(f"Global Median Skills:     {median_skills:.0f}")
    print(f"Highest Skill Count:      {max_skills}")
    print(f"Zero-Skill Casualties:    {zero_skill_resumes} ({(zero_skill_resumes/total_resumes)*100:.2f}%)")
    print("="*50 + "\n")

    # Category Breakdown
    print("📊 PER-CATEGORY BREAKDOWN")
    print("-" * 65)
    print(f"{'CATEGORY':<25} | {'COUNT':<6} | {'AVG SKILLS':<10} | {'ZERO SKILL %':<12}")
    print("-" * 65)

    # Group by category and calculate metrics
    category_stats = df.groupby('Category').agg(
        Total_Resumes=('ID', 'count'),
        Avg_Skills=('Skill_Count', 'mean'),
        Zero_Skill_Count=('Skill_Count', lambda x: (x == 0).sum())
    ).reset_index()

    # Calculate the percentage of zero-skill resumes per category
    category_stats['Zero_Skill_Pct'] = (category_stats['Zero_Skill_Count'] / category_stats['Total_Resumes']) * 100

    # Sort by Average Skills descending to see which categories performed best
    category_stats = category_stats.sort_values('Avg_Skills', ascending=False)

    for _, row in category_stats.iterrows():
        cat = row['Category']
        count = row['Total_Resumes']
        avg = row['Avg_Skills']
        zero_pct = row['Zero_Skill_Pct']
        
        print(f"{cat:<25} | {count:<6} | {avg:<10.2f} | {zero_pct:>5.2f}%")

    print("-" * 65 + "\n")
    
    # Optional: Investigate the zero-skill resumes
    if zero_skill_resumes > 0:
        print("💡 TIP: You have resumes with zero skills. To inspect them, run:")
        print("df[df['Skill_Count'] == 0][['ID', 'Category', 'Resume_str']].head()")

if __name__ == "__main__":
    analyze_extraction_results()