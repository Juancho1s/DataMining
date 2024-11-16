import serial
import json
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import TensorBoard
from time import sleep
import os
import datetime

model = Sequential([
    Dense(32, input_dim=3, activation='sigmoid'),  # Changed to sigmoid activation
    Dense(64, activation='sigmoid'),              # Changed to sigmoid activation
    Dense(64, activation='sigmoid'),              # Changed to sigmoid activation
    Dense(5, activation='sigmoid')                # Changed to sigmoid activation
])
model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

# Serial port setup
bluetooth_port = 'COM14'
baud_rate = 9600
ser = serial.Serial(bluetooth_port, baud_rate, timeout=1)

# Prediction function with dynamic scaling based on distance
def predict_speeds(current_direction, target_direction, distance):
    inputs = np.array([[current_direction, target_direction, distance]])
    predictions = model.predict(inputs)[0]
    
    # Apply scaling to provide more variation in wheel speeds
    # speed_factor = max(100, int(255 * distance / 100))  # Adjusts with distance
    
    speeds = {f"W{i+1}": round(predictions[i] * 150) for i in range(4)}
    speeds["D"] = round(predictions[-1] * 10)
    return speeds

try:
    while True:
        if ser.in_waiting > 0:
            incoming = ser.readline().decode('utf-8', errors='ignore').strip()
            
            if incoming.startswith('<') and incoming.endswith('>'):
                incoming = incoming[1:-1] 
                try:
                    data = json.loads(incoming)
                    distance = data.get("distance", 0)
                    current_direction = data.get("current_direction", 0)
                    target_direction = data.get("target_direction", 0)

                    # Optional: Skip real-time training or batch it for better performance
                    X_train = np.array([[current_direction, target_direction, distance]])
                    y_train = np.array([[120, 120, 120, 120, 0]])  # Placeholder labels
                    
                    model.fit(X_train, y_train, epochs=1, verbose=0)


                    speeds = predict_speeds(current_direction, target_direction, distance)
                    response = json.dumps(speeds)
                    ser.write((response + "\n").encode('utf-8'))

                    print(f"Distance: {distance} cm, Current Direction: {current_direction}, Target Direction: {target_direction}, Predicted Speeds: {speeds}")
                except json.JSONDecodeError:
                    print("Error decoding JSON")

except KeyboardInterrupt:
    print("Program terminated.")
    model.save(model_path)  # Save model on exit
finally:
    ser.close()
