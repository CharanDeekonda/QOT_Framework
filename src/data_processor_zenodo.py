import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import pad_sequences 

INPUT_FILE = "data/raw/janos-us_300_200.ssv"
OUTPUT_DIR = "data/processed/"
MAX_PATH_LENGTH = 30 

def load_and_process_zenodo_data():
    print(f"hz Loading Zenodo dataset from {INPUT_FILE}...")
    
    sequences = []
    gsnr_values = []
    
    try:
        with open(INPUT_FILE, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ ERROR: File not found at {INPUT_FILE}")
        return
    print("hz Parsing semicolon-separated data...")
    success_count = 0

    for i, line in enumerate(lines[1:]):
        try:
            parts = line.strip().split(';')
            gsnr_str = parts[-1]
            if not gsnr_str: 
                gsnr_str = parts[-2]
            gsnr = float(gsnr_str)
            path_str = parts[9] 
            clean_path = path_str.replace('[', '').replace(']', '').replace(',', ' ')
            path_nodes = [int(x) for x in clean_path.split()]
            if len(path_nodes) > 0:
                sequences.append(path_nodes)
                gsnr_values.append(gsnr)
                success_count += 1
        except (ValueError, IndexError):
            continue

    print(f"   - Successfully parsed {success_count} valid lightpaths.")

    if success_count == 0:
        print("❌ CRITICAL ERROR: Still found 0 paths. The column index might be wrong.")
        print("   - Sample Row Split:", lines[1].strip().split(';'))
        return
    actual_max = max(len(seq) for seq in sequences)
    final_max_len = max(MAX_PATH_LENGTH, actual_max)
    print(f"hz Padding paths to length {final_max_len}...")
    
    X_padded = pad_sequences(sequences, maxlen=final_max_len, padding='post', value=0)
    y = np.array(gsnr_values)

    X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "processed_data.pkl")
    with open(save_path, "wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test, final_max_len), f)

    print(f"✅ Real Dataset Processed! Saved to {save_path}")

if __name__ == "__main__":
    load_and_process_zenodo_data()