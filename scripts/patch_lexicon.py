import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def patch_lexicon(file_path: str):
    logging.info(f"Loading clean lexicon from {file_path}...")
    
    with open(file_path, "r", encoding="utf-8") as f:
        lexicon = json.load(f)

    # 1. The Surgical Strike: Remove the CS 101 Syntax that survived
    cs_101_junk = [
        "if statement", "while loop", "for in loop", "switch statement", 
        "arrays", "boolean", "pointers", "file io", "printf"
    ]
    
    removed_count = 0
    for junk in cs_101_junk:
        if junk in lexicon:
            del lexicon[junk]
            removed_count += 1
            
    logging.info(f"Nuked {removed_count} CS 101 syntax keys.")

    # 2. The Modern Injection: Add the specific stack we actually care about
    modern_stack = {
        "pnpm": ["pnpm workspace", "performant npm"],
        "fastapi": ["fast api"],
        "streamlit": ["streamlit dashboard"],
        "xgboost": ["xgb"],
        "pytorch": ["torch", "libtorch"],
        "tensorflow": ["tf", "tensor flow"],
        "react native": ["react-native", "rn"],
        "docker": ["docker container", "dockerfile", "docker compose"],
        "kubernetes": ["k8s", "kube"],
        "next.js": ["nextjs", "next js"],
        "three.js": ["threejs", "three js", "three.js"],
        "sentence-transformers": ["sentence transformers", "sbert", "sentence-bert"],
        "tailwind css": ["tailwindcss", "tailwind"]
    }

    # Only add them if they aren't somehow already there
    added_count = 0
    for key, aliases in modern_stack.items():
        if key not in lexicon:
            lexicon[key] = aliases
            added_count += 1
        else:
            # If the key exists, just append any new aliases
            existing_aliases = set(lexicon[key])
            existing_aliases.update(aliases)
            lexicon[key] = list(existing_aliases)

    logging.info(f"Injected {added_count} modern stack concepts.")

    # Save the patched database
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(lexicon, f, indent=4)
        
    logging.info("Lexicon patched successfully. Ready for inference.")

if __name__ == "__main__":
    CLEAN_FILE = "data/processed/clean_tech_lexicon.json"
    
    if not os.path.exists(CLEAN_FILE):
        logging.error(f"Cannot find {CLEAN_FILE}. Did you run clean_lexicon.py first?")
    else:
        patch_lexicon(CLEAN_FILE)