import os
import sys
import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.pdf_processor import extract_text, clean_text

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_ROOT = os.path.join(BASE_DIR, "data", "raw", "data")
CSV_PATH = os.path.join(BASE_DIR, "data", "raw", "Resume.csv")
OUTPUT_REPORT = os.path.join(BASE_DIR, "tests", "failed_parses.csv")

def run_full_corpus_test():
    print(f"Loading ground truth CSV from: {CSV_PATH}")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print("ERROR: Resume.csv not found. Check your paths.")
        return

    # Gather all PDF files across all category subdirectories
    all_pdfs = []
    for root, _, files in os.walk(DATA_ROOT):
        for file in files:
            if file.endswith('.pdf'):
                all_pdfs.append(os.path.join(root, file))

    total_files = len(all_pdfs)
    print(f"Found {total_files} PDFs to process.")
    
    if total_files == 0:
        return

    failures = []
    THRESHOLD = 85.0

    # Process with a progress bar
    for file_path in tqdm(all_pdfs, desc="Parsing PDFs", unit="file"):
        file_name = os.path.basename(file_path)
        pdf_id = file_name.replace(".pdf", "")
        category = os.path.basename(os.path.dirname(file_path))

        # Fetch ground truth
        truth_row = df[df['ID'] == int(pdf_id)]
        if truth_row.empty:
            failures.append({
                "ID": pdf_id, "Category": category, 
                "Reason": "ID not in CSV", "Similarity": 0
            })
            continue
            
        raw_truth = str(truth_row.iloc[0]['Resume_str'])
        clean_truth = clean_text(raw_truth)

        # Run extraction
        our_clean_text = extract_text(file_path)
        word_count = len(our_clean_text.split())
        
        # Check for empty extraction (complete failure)
        if word_count == 0:
            failures.append({
                "ID": pdf_id, "Category": category, 
                "Reason": "Zero words extracted", "Similarity": 0
            })
            continue

        # Calculate metrics
        similarity = fuzz.ratio(our_clean_text, clean_truth)
        is_gibberish = (len(our_clean_text) / word_count) < 3

        # Log failures
        if similarity < THRESHOLD or word_count < 100 or is_gibberish:
            reason = "Low Similarity" if similarity < THRESHOLD else "Low Word Count" if word_count < 100 else "Gibberish Detected"
            failures.append({
                "ID": pdf_id, "Category": category, 
                "Reason": reason, "Similarity": round(similarity, 2)
            })

    # Write report
    pass_count = total_files - len(failures)
    pass_rate = (pass_count / total_files) * 100

    print("\n" + "="*50)
    print(f"📊 FULL CORPUS RESULTS: {pass_count}/{total_files} Passed ({pass_rate:.2f}%)")
    print("="*50 + "\n")

    if failures:
        # Save failures to CSV for easy review
        report_df = pd.DataFrame(failures)
        report_df.to_csv(OUTPUT_REPORT, index=False)
        print(f"⚠️ Wrote {len(failures)} failure records to {OUTPUT_REPORT}")
        print("Review this file to see which resume categories break the parser the most.")

if __name__ == "__main__":
    run_full_corpus_test()