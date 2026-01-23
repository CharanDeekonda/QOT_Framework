import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle
import os
INPUT_FILE = "data/raw/network_data.csv"
OUTPUT_DIR = "data/processed/"
MAX_PATH_LENGTH = 10  
def load_and_process_data():
    print("🔄 Loading raw data...")
    df = pd.read_csv(INPUT_FILE)
    sequences = df['path_sequence'].apply(lambda x: [int(s) for s in x.split('-')]).tolist()
    X_padded = pad_sequences(sequences, maxlen=MAX_PATH_LENGTH, padding='post', value=0)
    y = df['GSNR_dB'].values
    X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)
    
    print(f"✅ Data Processed!")
    print(f"   - Training Shape: {X_train.shape}")
    print(f"   - Testing Shape:  {X_test.shape}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(os.path.join(OUTPUT_DIR, "processed_data.pkl"), "wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)
        
    print(f"   - Saved to: {os.path.join(OUTPUT_DIR, 'processed_data.pkl')}")

if __name__ == "__main__":
    load_and_process_data()