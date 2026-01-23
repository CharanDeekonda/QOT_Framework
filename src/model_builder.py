import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

def build_qot_model(num_unique_links, max_path_length):
    """
    Constructs the LSTM-based QoT Prediction Model.
    """
    model = Sequential()
    model.add(Embedding(input_dim=num_unique_links + 1, 
                        output_dim=16, 
                        input_length=max_path_length))
    model.add(LSTM(units=32, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

if __name__ == "__main__":
    model = build_qot_model(num_unique_links=50, max_path_length=10)
    model.summary()
    print("✅ Model Architecture built successfully!")