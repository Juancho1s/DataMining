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

# Define paths for saving and loading the model
MODEL_PATH = "./trained_model.keras"

# TensorBoard setup (logging disabled during runtime for speed)
log_dir = f"logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)

# Load or create the model
def initialize_model():
    if os.path.exists(MODEL_PATH):
        try:
            print("Loading existing model...")
            return load_model(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
    print(f"Creating a new model...")
    model = Sequential([
        Input(shape=(3,)),
        Dense(32, activation='sigmoid'),
        Dense(64, activation='sigmoid'),
        Dense(64, activation='sigmoid'),
        Dense(5, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    model.save(MODEL_PATH)
    return model

model = initialize_model()

# Reinforcement learning parameters
learning_rate = 0.01
reward_positive_factor = 1.0
reward_negative_base = -1.0
exploration_noise_stddev = 0.1
multiplier_factors = {
    0: 2, 1: 0.1, 2: 0.3, 3: 0.3,
    4: 0.1, 5: 0.1, 6: 0.8, 7: 0.8,
    8: 0.1, 9: 0.1,
}

# Agent-specific data storage
clients = {}

# Prediction function
def predict_speeds(inputs, explore=True):
    predictions = model.predict(inputs, verbose=0)[0]
    if explore:
        predictions += np.random.normal(0, exploration_noise_stddev, size=predictions.shape)
    predictions = np.clip(predictions, 0, 1)
    speeds = {
        f"W{i+1}": int(predictions[i] * 150) for i in range(4)
    }
    speeds["D"] = int(predictions[-1] * 9)
    speeds["predictedEnginesForce"] = sum(speeds[f"W{i+1}"] for i in range(4)) / 4
    return speeds, predictions

# Model update
def update_model(inputs, predictions, reward):
    targets = np.clip(predictions + learning_rate * reward * (predictions - 0.5), 0, 1)
    model.fit(inputs, np.array([targets]), verbose=0, callbacks=None)

# Reward calculation
def calculate_reward(distance, car_collision, speeds, current_direction, target_direction):
    direction = speeds["D"]
    reward = reward_negative_base / multiplier_factors[direction] if car_collision else 0
    if not car_collision:
        reward += reward_positive_factor * multiplier_factors[direction] / (distance + 1)
        direction_alignment = max(0, 1 - abs(target_direction - current_direction) / 9)
        reward += direction_alignment * 0.5
        reward += (speeds["predictedEnginesForce"] / 150) * 0.1

    print(reward)
    return reward

# WebSocket handler
async def handle_client(websocket, path):
    agent_id = None
    try:
        # Receive agent ID as the first message
        agent_id = await websocket.recv()
        if agent_id not in clients:
            clients[agent_id] = {"inputs": [], "predictions": [], "rewards": []}
        print(f"Agent {agent_id} connected.")

        async for message in websocket:
            try:
                data = json.loads(message)
                distance = data.get("distance", 0)
                current_direction = data.get("current_direction", 0)
                target_direction = data.get("target_direction", 0)
                car_collision = data.get("carCollision", False)

                inputs = np.array([[current_direction, target_direction, distance]])
                speeds, predictions = predict_speeds(inputs)
                reward = calculate_reward(distance, car_collision, speeds, current_direction, target_direction)

                # Update agent-specific data
                update_model(inputs, predictions, reward)
                clients[agent_id]["inputs"].append(inputs)
                clients[agent_id]["predictions"].append(predictions)
                clients[agent_id]["rewards"].append(reward)

                response = json.dumps(speeds)
                await websocket.send(response)
            except Exception as e:
                print(f"Error processing message from agent {agent_id}: {e}")

    except Exception as e:
        print(f"Error with agent {agent_id}: {e}")
    finally:
        if agent_id:
            print(f"Agent {agent_id} disconnected.")
            del clients[agent_id]

# Periodically save model
async def periodic_model_save(interval=60):
    while True:
        await asyncio.sleep(interval)
        try:
            model.save(MODEL_PATH)
            print("Model saved periodically.")
        except Exception as e:
            print(f"Error saving model: {e}")

# Server startup
async def main():
    print("WebSocket server is running...")
    server = await websockets.serve(handle_client, "localhost", 8080)
    loop = asyncio.get_event_loop()
    loop.create_task(periodic_model_save())
    await server.wait_closed()

# Graceful shutdown
def shutdown_handler():
    print("Shutting down gracefully...")
    model.save(MODEL_PATH)
    print("Model saved. Goodbye!")
    asyncio.get_event_loop().stop()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s, f: shutdown_handler())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        shutdown_handler()
