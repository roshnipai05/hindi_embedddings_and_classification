import fasttext.util
import os
import shutil

# Config
TARGET_DIR = "models"
MODEL_FILENAME = "cc.hi.300.bin"
TARGET_PATH = os.path.join(TARGET_DIR, MODEL_FILENAME)

def main():
    # 1. Create models directory if it doesn't exist
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"Created directory: {TARGET_DIR}")

    # 2. Check if model already exists
    if os.path.exists(TARGET_PATH):
        print(f"Model already exists at {TARGET_PATH}")
        print("Skipping download.")
        return

    print("--- Downloading Common Crawl Hindi Model (approx 4GB) ---")
    print("This may take a while depending on your internet speed...")
    
    # Download 'hi' (Hindi) model to current directory
    # fasttext.util automatically handles the .gz decompression
    fasttext.util.download_model('hi', if_exists='ignore') 

    # 3. Move the model to the 'models/' folder
    source_path = MODEL_FILENAME # fasttext downloads to current dir
    
    if os.path.exists(source_path):
        print(f"Moving {source_path} to {TARGET_DIR}...")
        shutil.move(source_path, TARGET_PATH)
        print("Move complete!")
        
        # Cleanup: Remove the .gz file if it was left behind
        gz_file = f"{MODEL_FILENAME}.gz"
        if os.path.exists(gz_file):
            os.remove(gz_file)
            print("Cleaned up compressed archive.")
    else:
        print("Error: Download appeared to finish but file not found.")

if __name__ == "__main__":
    main()