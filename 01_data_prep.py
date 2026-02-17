import re
import os
from datasets import load_dataset
from tqdm import tqdm

# Setup paths
os.makedirs("data", exist_ok=True)
OUTPUT_FILE = "data/hindi_corpus_clean.txt"
MAX_SENTENCES = 2_000_000

def clean_and_split(text):
    # 1. Split document into sentences based on Hindi full stop (।) or newlines
    # This regex looks for "।" or newline characters to split chunks
    sentences = re.split(r'[।\n]+', text)
    
    cleaned_sentences = []
    for s in sentences:
        s = s.lower()
        # Remove brackets but keep text
        s = re.sub(r"[()]", " ", s)
        # Keep Hindi chars, digits, spaces
        s = re.sub(r"[^\u0900-\u097F0-9\s]", " ", s)
        # Normalize spaces
        s = re.sub(r"\s+", " ", s).strip()
        
        # Only keep if it has at least 3 words
        if len(s.split()) >= 3:
            cleaned_sentences.append(s)
            
    return cleaned_sentences

def main():
    print(f"Streaming dataset... Target: {MAX_SENTENCES} SENTENCES (not documents)")
    
    # Streaming the dataset
    dataset = load_dataset("ai4bharat/sangraha", data_dir="verified/hin", split="train", streaming=True)
    
    seen = set()
    sentence_count = 0
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # We use a progress bar that updates manually
        pbar = tqdm(total=MAX_SENTENCES)
        
        for example in dataset:
            # Get list of clean sentences from the document
            doc_sentences = clean_and_split(example["text"])
            
            for sent in doc_sentences:
                if sent in seen:
                    continue
                
                seen.add(sent)
                f.write(sent + "\n")
                sentence_count += 1
                pbar.update(1)
                
                # STOP precisely when we hit the limit
                if sentence_count >= MAX_SENTENCES:
                    break
            
            if sentence_count >= MAX_SENTENCES:
                break
                
        pbar.close()

    print(f"\n✅ Done! Saved {sentence_count} cleaned sentences to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()