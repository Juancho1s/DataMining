import serial
import json
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from time import sleep
import os
import datetime

# Define paths for saving and loading the model
MODEL_PATH = "./trained_model.keras"

# Load or create the model
def initialize_model():
    if os.path.exists(MODEL_PATH):
        try:
            print("Loading existing model...")
            return load_model(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
    print("Creating a new model...")
    model = Sequential([
        Dense(32, input_dim=3, activation='sigmoid'),
        Dense(64, activation='sigmoid'),
        Dense(64, activation='sigmoid'),
        Dense(5, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    model.save(MODEL_PATH)
    return model

model = initialize_model()

# Bluetooth Serial setup
bluetooth_port = 'COM12'  # Adjust this to match your Bluetooth port
baud_rate = 9600
ser = serial.Serial(bluetooth_port, baud_rate, timeout=1)

# Prediction function
def predict_speeds(inputs, explore=True):
    predictions = model.predict(inputs, verbose=0)[0]  # Get predictions
    if explore:
        predictions += np.random.normal(0, 0.1, size=predictions.shape)  # Add exploration noise
    predictions = np.clip(predictions, 0, 1)  # Ensure predictions are between 0 and 1

    # Compute average speed for the first four predictions (representing wheels)
    averageSpeed = sum(predictions[:4]) / 4  # Correctly average the first 4 predictions

    # Scale wheel speeds proportionally to averageSpeed (e.g., up to 255)
    max_wheel_speed = 200
    speeds = {f"W{i+1}": max(100, int(averageSpeed * max_wheel_speed)) for i in range(4)}

    # Direction is scaled to [0, 9]
    speeds["D"] = int(predictions[-1] * 9)

    return speeds, predictions



# Model update
def update_model(inputs, predictions, reward):
    targets = np.clip(predictions + 0.01 * reward * (predictions - 0.5), 0, 1)
    model.fit(inputs, np.array([targets]), verbose=0, callbacks=None)

# Reward calculation
def calculate_reward(distance, car_collision, speeds, current_direction, target_direction):
    car_collision = (distance < 8 and current_direction in [0, 6, 7])
    direction = speeds["D"]
    reward = -1.0 / (direction + 1) if car_collision else 0
    if not car_collision:
        if distance > 10 and distance < 150:
            reward += 1.0 * (direction + 1) / (distance + 1)
            direction_alignment = max(0, 1 - abs(target_direction - current_direction) / 9)
            reward += direction_alignment * 0.5
        # reward += (speeds["predictedEnginesForce"] / 150) * 0.1

    print(f"given Reward: {reward}")
    return reward

# Main loop
try:
    print("Bluetooth server running...")
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
                    car_collision = data.get("carCollision", False)

                    inputs = np.array([[current_direction, target_direction, distance]])
                    speeds, predictions = predict_speeds(inputs)
                    reward = calculate_reward(distance, car_collision, speeds, current_direction, target_direction)
                    update_model(inputs, predictions, reward)

                    response = json.dumps(speeds)
                    ser.write((response + "\n").encode('utf-8'))
                    print(f"Received: {data}, Sent: {speeds}")
                except json.JSONDecodeError:
                    print("Error decoding JSON")
                except Exception as e:
                    print(f"Error: {e}")
except KeyboardInterrupt:
    print("Shutting down...")
    model.save(MODEL_PATH)
    ser.close()
    print("Model saved and Bluetooth connection closed.")
