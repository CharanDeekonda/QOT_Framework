import pickle
import numpy as np
import os
import tensorflow as tf
from model_builder import build_qot_model
import matplotlib.pyplot as plt
DATA_PATH = "data/processed/processed_data.pkl"
MODEL_SAVE_PATH = "models/saved_models/qot_model.keras"
NUM_EPOCHS = 10          # Reduced to 10 because we have huge data (it learns faster)
BATCH_SIZE = 64          # Increased batch size for speed
NUM_UNIQUE_LINKS = 300   # The Janos-US topology has roughly 300 links
MAX_PATH_LENGTH = 10     # Placeholder, will be overwritten by loaded data

def train_model():
    # 1. Load Data
    print("hz Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: File not found at {DATA_PATH}.")
        return

    # LOAD 5 ITEMS (New Logic)
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
        if len(data) == 5:
            X_train, X_test, y_train, y_test, loaded_max_len = data
            global MAX_PATH_LENGTH
            MAX_PATH_LENGTH = loaded_max_len
            print(f"   - Adapted MAX_PATH_LENGTH to: {MAX_PATH_LENGTH}")
        else:
            # Fallback for old data
            X_train, X_test, y_train, y_test = data

    print(f"   - Training Data Shape: {X_train.shape}")
    
    # 2. Build Model
    print("hz Building Model...")
    # We increase the embedding size slightly since the network is bigger
    model = build_qot_model(NUM_UNIQUE_LINKS, MAX_PATH_LENGTH)
    model.build(input_shape=(None, MAX_PATH_LENGTH))
    model.summary()

    # 3. Train Model
    print(f"hz Starting Training for {NUM_EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    # 4. Save Model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")

    # 5. Evaluate
    print("hz Evaluating on Test Data...")
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"   - Mean Absolute Error (MAE): {mae:.4f} dB")

    # Plot
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_model()