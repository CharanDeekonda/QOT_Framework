import pickle
import numpy as np
import os
import tensorflow as tf
from model_builder import build_qot_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

DATA_PATH = "data/processed/processed_data.pkl"
MODEL_SAVE_PATH = "models/saved_models/qot_model.keras"

# --- OPTIMIZED FOR HIGHER ACCURACY ---
NUM_EPOCHS = 30          # Increased from 10 to 30 to allow the bigger model to converge
BATCH_SIZE = 128         # Increased for better stability with the larger 64-unit LSTM
NUM_UNIQUE_LINKS = 300   
MAX_PATH_LENGTH = 10     

def train_model():
    # 1. Load Data
    print("hz Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: File not found at {DATA_PATH}.")
        return

    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
        if len(data) == 5:
            X_train, X_test, y_train, y_test, loaded_max_len = data
            global MAX_PATH_LENGTH
            MAX_PATH_LENGTH = loaded_max_len
            print(f"   - Adapted MAX_PATH_LENGTH to: {MAX_PATH_LENGTH}")
        else:
            X_train, X_test, y_train, y_test = data

    print(f"   - Training Data Shape: {X_train.shape}")
    
    # 2. Build Model (Now using your 64-unit LSTM architecture)
    print("hz Building Enhanced Model...")
    model = build_qot_model(NUM_UNIQUE_LINKS, MAX_PATH_LENGTH)
    model.build(input_shape=(None, MAX_PATH_LENGTH))
    model.summary()

    # 3. Train Model
    print(f"hz Starting Training for {NUM_EPOCHS} epochs...")
    # Added EarlyStopping to prevent overfitting if the model stops improving
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )
    
    # 4. Save Model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"✅ Enhanced Model saved to {MODEL_SAVE_PATH}")

    # 5. Evaluate
    print("hz Evaluating on Test Data...")
    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"   - Mean Absolute Error (MAE): {mae:.4f} dB")
    
    y_pred = model.predict(X_test)
    r2_acc = r2_score(y_test, y_pred)
    
    print(f"✅ Final Model Accuracy (R-Squared): {r2_acc * 100:.2f}%")
    
    # Plotting the Learning Curve
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Enhanced Model Learning Curve (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_model()