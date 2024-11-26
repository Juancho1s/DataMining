import asyncio
import websockets
import json
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import TensorBoard
import datetime
import os
import signal
import random


# Define the TensorBoard log directory
log_dir = f"logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Define paths for saving and loading the model
MODEL_PATH = "./trained_model.keras"

# Ensure the model file is created if it doesn't exist
if not os.path.exists(MODEL_PATH):
    print(f"Model file not found at {MODEL_PATH}. Creating a new model...")
    model = Sequential([
        Input(shape=(3,)),
        Dense(32, activation='sigmoid'),
        Dense(64, activation='sigmoid'),
        Dense(64, activation='sigmoid'),
        Dense(5, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    model.save(MODEL_PATH)  # Save the newly created model
else:
    try:
        print("Loading existing model...")
        model = load_model(MODEL_PATH)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    except Exception as e:
        print(f"Error loading model: {e}. Creating a new model...")
        model = Sequential([
            Input(shape=(3,)),
            Dense(32, activation='sigmoid'),
            Dense(64, activation='sigmoid'),
            Dense(64, activation='sigmoid'),
            Dense(5, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
        model.save(MODEL_PATH)


# Reinforcement learning parameters
learning_rate = 0.01
reward_positive_factor = 1.0
reward_negative_base = -1.0
exploration_noise_stddev = 0.1
multiplier_factors = {
    0: 2,    # forward
    1: 0.1,    # backward
    2: 0.3,    # trun right
    3: 0.3,    # turn left
    4: 0.1,    # left translation
    5: 0.1,    # right translation
    6: 0.8,   # left diagonal forward
    7: 0.8,   # right diagonal forward
    8: 0.1,    # left diagonal backward
    9: 0.1,    # right diagonal backward
}

# Prediction function with exploration
def predict_speeds(current_direction, target_direction, distance, explore=True):
    inputs = np.array([[current_direction, target_direction, distance]])
    predictions = model.predict(inputs, verbose=0)[0]

    if explore:
        # Add Gaussian noise for exploration
        predictions += np.random.normal(0, exploration_noise_stddev, size=predictions.shape)

    # Ensure outputs stay within valid range
    predictions = np.clip(predictions, 0, 1)

    speeds = {f"W{i+1}": round(predictions[i] * 150) for i in range(4)}  # Scale wheel speeds
    speeds["D"] = round(predictions[-1] * 9)  # Scale direction
    speeds["predictedEnginesForce"] = sum(speeds[f"W{i+1}"] for i in range(4)) / 4  # Average speed

    return speeds, predictions

# Update the model using better training logic
def update_model(inputs, predictions, reward):
    """
    Update the model based on inputs, predictions, and calculated reward.
    """
    targets = predictions + learning_rate * reward * (predictions - 0.5)
    targets = np.clip(targets, 0, 1)  # Ensure targets are valid

    # Fit the model and log metrics to TensorBoard
    model.fit(inputs, np.array([targets]), verbose=0, callbacks=[tensorboard_callback])


# Dynamic reward function
def calculate_reward(distance, car_collision, speeds, current_direction, target_direction):
    """
    Enhanced reward function considering distance, collisions, velocity, and direction alignment.

    Args:
        distance (float): Distance to the nearest obstacle.
        car_collision (bool): Whether the car has collided with an obstacle.
        speeds (dict): Predicted speeds and direction.
        current_direction (float): The current direction of the robot.
        target_direction (float): The target direction for the robot.

    Returns:
        float: Calculated reward value.
    """
    direction = speeds["D"]
    average_speed = speeds["predictedEnginesForce"]
    reward = 0

    if car_collision:
        # Apply a negative reward for collision, scaled by the direction multiplier
        reward = reward_negative_base / multiplier_factors[direction]
    else:
        # Positive reward for maintaining a safe distance
        if distance > 20:
            reward += reward_positive_factor * multiplier_factors[direction] * (1 / ((distance + 1)))

        # Additional reward for aligning with the target direction
        direction_alignment = max(0, 1 - abs(target_direction - current_direction) / 9)  # Scale alignment to [0, 1]
        reward += direction_alignment * 0.5  # Adjust weight as needed

        # Speed incentive: Encourage efficient but not excessive speeds
        reward += (average_speed / 150) * 0.1  # Scale average speed to a small contribution

    return reward



# WebSocket server logic
async def handle_client(websocket, path):
    async for message in websocket:
        try:
            # Parse the incoming JSON
            data = json.loads(message)
            distance = data.get("distance", 0)
            current_direction = data.get("current_direction", 0)
            target_direction = data.get("target_direction", 0)
            car_collision = data.get("carCollision", False)

            # Prepare inputs
            inputs = np.array([[current_direction, target_direction, distance]])

            # Make predictions
            speeds, predictions = predict_speeds(current_direction, target_direction, distance)

            # Calculate dynamic reward
            reward = calculate_reward(distance, car_collision, speeds, current_direction, target_direction)

            # Update the model with better training logic
            update_model(inputs, predictions, reward)

            if car_collision:
                print(f"Collision detected. Applied negative reward: {reward}")
            else:
                print(f"Smooth operation. Applied positive reward: {reward}")

            # Send response back to Unity
            response = json.dumps(speeds)
            await websocket.send(response)
            print(f"Processed: {data} -> {speeds}")

        except Exception as e:
            print(f"Error: {e}")

# Save model before shutdown
async def save_model_and_exit():
    try:
        print("Saving model before shutdown...")
        model.save(MODEL_PATH)
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {e}")
    finally:
        asyncio.get_event_loop().stop()

# Start the WebSocket server
start_server = websockets.serve(handle_client, "localhost", 8080)

async def main():
    server = await websockets.serve(handle_client, "localhost", 8080)
    print("WebSocket server is running...")
    try:
        await asyncio.Future()  # Keep running until interrupted
    finally:
        print("Shutting down server...")

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("Server shutdown requested via KeyboardInterrupt.")
