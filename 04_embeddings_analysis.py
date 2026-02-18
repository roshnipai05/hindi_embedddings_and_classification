import fasttext
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib import font_manager
import os

# --- Configuration ---
# Get the absolute path to the font in your project folder
current_dir = os.path.dirname(os.path.abspath(__file__))
HINDI_FONT_PATH = os.path.join(current_dir, "fonts", "NotoSansDevanagari-Regular.ttf")

MY_MODEL_PATH = "models/custom_hindi_vectors.bin"
CC_MODEL_PATH = "models/cc.hi.300.bin"

# --- Helper Functions ---

def get_neighbors(model, word, k=10):
    """Returns top k nearest neighbors for a word."""
    if word not in model.words:
        return []
    return [w for sim, w in model.get_nearest_neighbors(word, k=k)]

def jaccard_similarity(list1, list2):
    """Calculates overlap between two lists of neighbors."""
    s1, s2 = set(list1), set(list2)
    if not s1 or not s2: return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))

def co_sim(model, w1, w2):
    """Calculates Cosine Similarity between two words."""
    if w1 not in model.words or w2 not in model.words:
        return 0.0
    v1 = model.get_word_vector(w1)
    v2 = model.get_word_vector(w2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def evaluate_group_similarity(model, groups):
    """Calculates average similarity within semantic clusters."""
    results = {}
    for group_name, words in groups.items():
        sims = []
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                sim = co_sim(model, words[i], words[j])
                if sim != 0: sims.append(sim)
        results[group_name] = np.mean(sims) if sims else 0.0
    return results

def get_hindi_font():
    """Tries to find a valid Hindi font on the system."""
    # Common Hindi fonts on Windows & Linux
    candidates = [
        "C:\\Windows\\Fonts\\Nirmala.ttf",      # Standard Windows 10/11
        "C:\\Windows\\Fonts\\nirmala.ttf",      # Lowercase check
        "C:\\Windows\\Fonts\\Mangal.ttf",       # Older Windows
        "C:\\Windows\\Fonts\\Arial Unicode.ttf",# Universal
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf", # Linux
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf" # Mac
    ]
    
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def plot_pca_comparison(model_name, model, words_dict):
    """Visualizes 2D projection of word clusters with safe font loading."""
    print(f"Generating plot for {model_name}...")
    
    all_words = []
    labels = []
    colors = []
    color_map = {'politics': 'red', 'sports': 'blue', 'economy': 'green', 'movies': 'purple'}
    
    # 1. Collect vectors
    for category, words in words_dict.items():
        for w in words:
            if w in model.words:
                all_words.append(model.get_word_vector(w))
                labels.append(w)
                colors.append(color_map.get(category, 'gray'))
    
    if not all_words: 
        print("No words found in model to plot.")
        return

    # 2. Reduce dimensions (PCA)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    result = pca.fit_transform(np.array(all_words))
    
    # 3. Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(result[:, 0], result[:, 1], c=colors, s=100, edgecolors='k')
    
    # 4. Safe Labeling
    font_path = get_hindi_font()
    if font_path:
        prop = font_manager.FontProperties(fname=font_path)
        print(f"Using font: {font_path}")
        for i, word in enumerate(labels):
            plt.annotate(word, xy=(result[i, 0], result[i, 1]), 
                         xytext=(5, 2), textcoords='offset points',
                         fontproperties=prop, fontsize=12)
    else:
        print("⚠️ Warning: No Hindi font found. Plotting without labels to prevent crash.")
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, label=k, markersize=10)
                       for k, c in color_map.items()]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.title(f"Semantic Clusters: {model_name}")
    plt.grid(True, alpha=0.3)
    plt.show()

# --- Main Execution ---

def main():
    # 1. Load Models
    if not os.path.exists(CC_MODEL_PATH):
        print(f"❌ Error: Could not find {CC_MODEL_PATH}")
        print("Please download it from https://fasttext.cc/docs/en/crawl-vectors.html")
        return

    print("Loading Custom Model... (Fast)")
    my_model = fasttext.load_model(MY_MODEL_PATH)
    
    print("Loading Common Crawl Model... (Slow ~1-2 mins)")
    cc_model = fasttext.load_model(CC_MODEL_PATH)

    # --- Test Data ---
    test_words = ["प्रधानमंत्री", "क्रिकेट", "शिक्षा", "दिल्ली", "विज्ञान"]
    
    analogies = [
        ("राजा", "पुरुष", "स्त्री"),   # King - Man + Woman -> Queen
        ("भारत", "दिल्ली", "जापान"),   # India - Delhi + Japan -> Tokyo
        ("क्रिकेट", "बल्ला", "फुटबॉल") # Cricket - Bat + Football -> Ball?
    ]
    
    semantic_groups = {
        "sports": ["क्रिकेट", "फुटबॉल", "मैच", "खिलाड़ी", "टूर्नामेंट", "विकेट"],
        "politics": ["राजनीति", "सरकार", "चुनाव", "संसद", "नेता", "मंत्री"],
        "economy": ["अर्थव्यवस्था", "व्यापार", "बैंक", "बाजार", "निवेश", "वित्त"],
        "movies": ["फिल्म", "सिनेमा", "अभिनेता", "निर्देशक", "गीत", "कलाकार"]
    }

    # --- 2. Qualitative: Neighbor Comparison ---
    print(f"\n{'='*20} 1. NEIGHBOR OVERLAP {'='*20}")
    print(f"{'Word':<15} | {'My Model (Top 3)':<35} | {'CC Model (Top 3)':<35} | {'Overlap'}")
    print("-" * 105)

    for word in test_words:
        my_n = get_neighbors(my_model, word, k=10)
        cc_n = get_neighbors(cc_model, word, k=10)
        overlap = jaccard_similarity(my_n, cc_n)
        
        my_str = ", ".join(my_n[:3])
        cc_str = ", ".join(cc_n[:3])
        print(f"{word:<15} | {my_str:<35} | {cc_str:<35} | {overlap:.2f}")

    # --- 3. Qualitative: Analogy Test ---
    print(f"\n{'='*20} 2. ANALOGY TEST {'='*20}")
    for a, b, c in analogies:
        print(f"Query: {a} - {b} + {c} = ?")
        
        # My Model
        my_res = "N/A"
        if all(w in my_model.words for w in [a,b,c]):
            # get_analogies returns [(score, word)]
            results = my_model.get_analogies(wordA=b, wordB=a, wordC=c, k=1)
            if results: my_res = f"{results[0][1]} ({results[0][0]:.2f})"
            
        # CC Model
        cc_res = "N/A"
        if all(w in cc_model.words for w in [a,b,c]):
            results = cc_model.get_analogies(wordA=b, wordB=a, wordC=c, k=1)
            if results: cc_res = f"{results[0][1]} ({results[0][0]:.2f})"

        print(f"  My Model: {my_res}")
        print(f"  CC Model: {cc_res}")
        print("-" * 40)

    # --- 4. Quantitative: Cluster Tightness ---
    print(f"\n{'='*20} 3. CLUSTER TIGHTNESS (Avg Cos Sim) {'='*20}")
    
    my_group_scores = evaluate_group_similarity(my_model, semantic_groups)
    cc_group_scores = evaluate_group_similarity(cc_model, semantic_groups)
    
    print(f"{'Category':<15} | {'My Model':<10} | {'CC Model':<10}")
    print("-" * 40)
    for cat in semantic_groups.keys():
        s1 = my_group_scores.get(cat, 0)
        s2 = cc_group_scores.get(cat, 0)
        print(f"{cat:<15} | {s1:.4f}     | {s2:.4f}")

    # --- 5. Visual: PCA Plot ---
    print("\nGenerating Plots...")
    plot_pca_comparison("My Custom Model (Sangraha)", my_model, semantic_groups)
    plot_pca_comparison("Common Crawl Model (Official)", cc_model, semantic_groups)

if __name__ == "__main__":
    main()