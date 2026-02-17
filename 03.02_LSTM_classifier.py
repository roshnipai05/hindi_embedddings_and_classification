import pandas as pd
import numpy as np
import tensorflow as tf
import fasttext

# --- TENSORFLOW 2.16+ COMPATIBLE IMPORTS ---
from keras.src.legacy.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional, Dropout
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Config
DATA_PATH = "data/bbc_hindi_news.csv"
VECTOR_MODEL_PATH = "models/custom_hindi_vectors.bin"
MAX_NB_WORDS = 20000
MAX_LEN = 200
EMBED_DIM = 100

def main():
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_PATH)
        # Handle inconsistent column names
        cols = df.columns.tolist()
        print(f"Loaded columns: {cols}")
        
        # Heuristic to find text/label columns
        text_col = next((c for c in cols if 'text' in c.lower() or 'content' in c.lower()), cols[1])
        label_col = next((c for c in cols if 'category' in c.lower() or 'topic' in c.lower()), cols[0])
        
        # Clean nulls and ensure string type
        df = df.dropna(subset=[text_col, label_col])
        X_raw = df[text_col].astype(str).tolist()
        Y_raw = df[label_col].astype(str).tolist()
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # 2. Tokenize
    print("Tokenizing text...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(X_raw)
    word_index = tokenizer.word_index
    X = pad_sequences(tokenizer.texts_to_sequences(X_raw), maxlen=MAX_LEN)

    # 3. Encode Labels
    le = LabelEncoder()
    Y_encoded = le.fit_transform(Y_raw)
    # Convert to one-hot encoding
    Y = tf.keras.utils.to_categorical(Y_encoded)
    num_classes = Y.shape[1]
    print(f"Classes detected: {le.classes_}")
    
    # 4. Load Pre-trained Vectors
    print("Loading custom FastText vectors...")
    ft_model = fasttext.load_model(VECTOR_MODEL_PATH)
    
    embedding_matrix = np.zeros((MAX_NB_WORDS, EMBED_DIM))
    hits = 0
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS: continue
        if word in ft_model.words:
            embedding_matrix[i] = ft_model.get_word_vector(word)
            hits += 1
            
    print(f"Embedding coverage: {hits}/{min(len(word_index), MAX_NB_WORDS)}")

    # 5. Build Robust Model (Regularized to prevent overfitting)
    model = Sequential()
    
    # Embedding Layer (Trainable=True allows fine-tuning)
    model.add(Embedding(MAX_NB_WORDS, 
                        EMBED_DIM, 
                        weights=[embedding_matrix], 
                        input_length=MAX_LEN, 
                        trainable=True))
    
    # SpatialDropout drops entire 1D feature maps (better for NLP)
    model.add(SpatialDropout1D(0.4))
    
    # Bidirectional LSTM with L2 Regularization (Weight Decay)
    # We reduce units to 64 to lower model capacity and prevent memorization
    model.add(Bidirectional(LSTM(64, 
                                 dropout=0.4,           # Input dropout
                                 recurrent_dropout=0.4, # Gate dropout
                                 kernel_regularizer=l2(0.001), # Penalize large weights
                                 recurrent_regularizer=l2(0.001))))
    
    # Dense layer with bottleneck and L2 regularization
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    
    # Final Dropout before classification
    model.add(Dropout(0.5))
    
    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    # 6. Callbacks for Smart Training
    
    # Reduce Learning Rate if validation loss plateaus
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                     patience=3, 
                                     verbose=1, 
                                     factor=0.5, 
                                     min_lr=0.00001)
    
    # Stop early if validation loss stops improving
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=6,
                                   restore_best_weights=True,
                                   verbose=1)

    # 7. Train
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    print("--- Starting Training (Max 30 Epochs) ---")
    model.fit(X_train, Y_train, 
              epochs=30, 
              batch_size=32, 
              validation_data=(X_test, Y_test),
              callbacks=[lr_reduction, early_stopping])

if __name__ == "__main__":
    main()