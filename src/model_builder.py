import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

def build_qot_model(num_unique_links, max_path_length):
    """
    Enhanced LSTM-based QoT Prediction Model with higher capacity.
    """
    model = Sequential()
    
    # INCREASED: output_dim from 16 to 32 to capture more link-level features
    model.add(Embedding(input_dim=num_unique_links + 1, 
                        output_dim=32, 
                        input_length=max_path_length))
    
    # INCREASED: units from 32 to 64 to process longer, more complex path sequences
    model.add(LSTM(units=64, return_sequences=False))
    
    # STAYS: Dropout helps prevent overfitting during longer training sessions
    model.add(Dropout(0.2))
    
    # INCREASED: units from 16 to 32 for better non-linear approximation
    model.add(Dense(32, activation='relu'))
    
    # OUTPUT: Still predicting 1 GSNR score
    model.add(Dense(1, activation='linear'))
    
    # Using Adam optimizer as described in the methodology
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

if __name__ == "__main__":
    # Test with your adapted max_path_length of 42
    model = build_qot_model(num_unique_links=300, max_path_length=42)
    model.summary()
    print("✅ Enhanced Model Architecture built successfully!")