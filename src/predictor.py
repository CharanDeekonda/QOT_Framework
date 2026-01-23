import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH = "models/saved_models/qot_model.keras"
MAX_PATH_LENGTH = 10  
MODULATION_FORMATS = [
    (22.0, "64QAM"),  # Requires very clean signal (High GSNR)
    (18.0, "16QAM"),  # Requires clean signal
    (12.0, "8QAM"),   # Medium
    (9.0,  "QPSK"),   # Robust, but slower
    (0.0,  "BPSK")    # Very robust, very slow (fallback)
]

def load_prediction_model():
    print("hz Loading Trained AI Model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def recommend_modulation(gsnr_score):
    """
    Decides the best modulation format based on the predicted GSNR.
    """
    for threshold, format_name in MODULATION_FORMATS:
        if gsnr_score >= threshold:
            return format_name
    return "NO_CONNECTION (Signal too weak)"

def predict_path_quality(model, link_sequence):
    """
    End-to-end prediction for a single path.
    """
    sequence_padded = pad_sequences([link_sequence], 
                                    maxlen=MAX_PATH_LENGTH, 
                                    padding='post', 
                                    value=0)
    predicted_gsnr = model.predict(sequence_padded, verbose=0)[0][0]
    format_recommendation = recommend_modulation(predicted_gsnr)
    
    return predicted_gsnr, format_recommendation

if __name__ == "__main__":
    ai_model = load_prediction_model()
    
    if ai_model:
        print("\n--- 🚀 OPTICAL NETWORK PREDICTION SYSTEM ---")
        print("Enter a sequence of link IDs (e.g., '5 12 23') to test a path.")
        print("Type 'exit' to quit.\n")
        
        while True:
            user_input = input("Enter Path Sequence > ")
            
            if user_input.lower() == 'exit':
                break
            
            try:
                path_ids = [int(x) for x in user_input.split()]
                if len(path_ids) > MAX_PATH_LENGTH:
                    print(f"⚠️  Path too long! Max length is {MAX_PATH_LENGTH}.")
                    continue
                gsnr, mod_format = predict_path_quality(ai_model, path_ids)
                print("-" * 40)
                print(f"📡 Path: {path_ids}")
                print(f"📊 Predicted GSNR:      {gsnr:.4f} dB")
                print(f"✅ Recommended Format:  {mod_format}")
                print("-" * 40 + "\n")
                
            except ValueError:
                print("❌ Invalid input. Please enter numbers separated by space.")