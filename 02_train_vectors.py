import fasttext
import os
import time

# Setup paths
os.makedirs("models", exist_ok=True)
CORPUS_FILE = "data/hindi_corpus_clean.txt"
MODEL_PATH = "models/custom_hindi_vectors.bin"

def get_file_size(filepath):
    """Returns file size in MB"""
    if not os.path.exists(filepath):
        return 0
    return os.path.getsize(filepath) / (1024 * 1024)

def train():
    file_size = get_file_size(CORPUS_FILE)
    print(f"--- Initialization ---")
    print(f"Corpus File: {CORPUS_FILE}")
    print(f"Corpus Size: {file_size:.2f} MB")
    
    print("\n--- Starting FastText Training ---")
    print("NOTE: You should see a progress bar below. If not, it may update in chunks.")
    
    start_time = time.time()
    
    # verbose=2 forces the C++ library to print its internal progress bar
    # thread=4 uses 4 CPU cores (adjust based on your machine for speed)
    model = fasttext.train_unsupervised(
        CORPUS_FILE, 
        model='skipgram', 
        dim=100, 
        verbose=2,
        thread=4 
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n✅ Training Complete!")
    print(f"Time taken: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model

def compare(my_model):
    test_words = ["प्रधानमंत्री", "क्रिकेट", "शिक्षा", "मोबाइल"]
    
    print("\n--- Qualitative Comparison ---")
    print(f"Checking {len(test_words)} test words against vocabulary...")
    
    for word in test_words:
        print(f"\nTarget: {word}")
        if word in my_model.words:
            neighbors = my_model.get_nearest_neighbors(word, k=3)
            # Formatting the output to be cleaner
            for score, neighbor in neighbors:
                print(f"  -> {neighbor} ({score:.4f})")
        else:
            print("  [x] Word not in vocabulary (Training data might be too small)")

if __name__ == "__main__":
    if not os.path.exists(CORPUS_FILE):
        print(f"Error: {CORPUS_FILE} not found. Please run the data prep script first.")
    else:
        model = train()
        compare(model)