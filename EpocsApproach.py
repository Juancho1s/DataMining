import serial
import json
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
import os

# Paths and Constants
MODEL_PATH = "./new_trained_maped_model.keras"
MAP_SIZE = 20  # 20x20 grid
GRID_RESOLUTION = 10  # Each cell is 10 cm
BLUETOOTH_PORT = 'COM12'
BAUD_RATE = 9600
LOG_DIR = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
DIRECTION_OFFSETS = {
    0: (0, +1),   # Forward
    1: (0, -1),   # Backward
    2: (0, 0),    # Turn right
    3: (0, 0),    # Turn left
    4: (-1, 0),   # Translate left
    5: (+1, 0),   # Translate right
    6: (-1, +1),  # Diagonal up-left
    7: (+1, +1),  # Diagonal up-right
    8: (-1, -1),  # Diagonal down-left
    9: (+1, -1)   # Diagonal down-right
}

# Initialize Global Map and Robot State
map_grid = np.zeros((MAP_SIZE, MAP_SIZE), dtype=int)  # 0 = unexplored, 1 = free, -1 = obstacle
robot_position = (MAP_SIZE // 2, MAP_SIZE // 2)  # Start at center


def initialize_model():
    """Load or create a machine learning model."""
    if os.path.exists(MODEL_PATH):
        try:
            print("Loading existing model...")
            return load_model(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
    print("Creating a new model...")
    model = Sequential([
        Dense(32, input_dim=7, activation='sigmoid'),
        Dense(64, activation='sigmoid'),
        Dense(64, activation='sigmoid'),
        Dense(5, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    model.save(MODEL_PATH)
    return model


def update_map(position, distance, direction_code):
    """Updates the map based on robot's movement."""
    global map_grid
    x, y = position
    dx, dy = DIRECTION_OFFSETS.get(direction_code, (0, 0))
    steps = int(distance / GRID_RESOLUTION)
    for step in range(1, steps + 1):
        nx, ny = x + step * dx, y + step * dy
        if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
            map_grid[nx, ny] = 1  # Free space
        else:
            break
    # Mark obstacle
    nx, ny = x + steps * dx, y + steps * dy
    if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
        map_grid[nx, ny] = -1  # Obstacle


def get_surrounding_features(position):
    """Extract features of the surroundings for input."""
    x, y = position
    features = [
        map_grid[x + dx, y + dy] if 0 <= x + dx < MAP_SIZE and 0 <= y + dy < MAP_SIZE else -1
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ]
    return features


def calculate_reward(distance, collision, explored, alignment):
    """Compute the reward for the current state."""
    reward = 0
    if collision:
        reward -= 5  # Penalize collisions
    else:
        # Reward for aligning with target direction
        reward += max(0, 1 - abs(alignment) / 9) * 0.5

        # Reward for exploration
        reward += 1 if explored else -0.5

        # Bell-shaped reward based on distance
        if 10 < distance < 150:  # Only apply the reward within this range
            min_reward = 0
            max_reward = 1
            peak_distance = 75  # Distance where the reward is maximized
            width = 30  # Controls the spread of the reward
            reward += min_reward + (max_reward - min_reward) * np.exp(-((distance - peak_distance) ** 2) / (2 * width ** 2))
    
    return reward


def predict_speeds(inputs, explore=True):
    """Predict motor speeds and directions."""
    predictions = model.predict(inputs, verbose=0)[0]
    if explore:
        predictions += np.random.normal(0, 0.1, predictions.shape)
    predictions = np.clip(predictions, 0, 1)
    speeds = {f"W{i+1}": max(140, round((sum(predictions[0:3]) / 4 ) * 200)) for i in range(4)}
    speeds["D"] = round(predictions[4] * 9)  # Direction
    return speeds, predictions


def update_model(inputs, predictions, reward):
    """Update model with the reward."""
    targets = np.clip(predictions + 0.01 * reward * (predictions - 0.5), 0, 1)
    targets = np.array([targets])  # Reshape to (1, 5)
    model.fit(inputs, targets, verbose=0, callbacks=[tensorboard_callback])



def update_exploration_status(position):
    """Update exploration status of the current cell."""
    x, y = position
    if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE and map_grid[x, y] == 0:
        map_grid[x, y] = 1
        return True
    return False


# Initialize Model and TensorBoard
model = initialize_model()
tensorboard_callback = TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

# Bluetooth Serial Communication
ser = serial.Serial(BLUETOOTH_PORT, BAUD_RATE, timeout=1)

# Main Loop
try:
    print("Bluetooth server running...")
    while True:
        if ser.in_waiting > 0:
            incoming = ser.readline().decode('utf-8', errors='ignore').strip()

            if incoming.startswith('<') and incoming.endswith('>'):
                try:
                    data = json.loads(incoming[1:-1])
                    distance = data.get("distance", 0)
                    current_dir = data.get("current_direction", 0)
                    target_dir = data.get("target_direction", 0)
                    collision = distance < 5

                    explored = update_exploration_status(robot_position)
                    features = get_surrounding_features(robot_position)
                    inputs = np.array([[current_dir, target_dir, distance] + features])

                    speeds, predictions = predict_speeds(inputs)
                    alignment = target_dir - current_dir
                    reward = calculate_reward(distance, collision, explored, alignment)

                    update_model(inputs, predictions, reward)
                    update_map(robot_position, distance, speeds["D"])

                    ser.write((json.dumps(speeds) + "\n").encode('utf-8'))
                    print(f"Processed: {data} -> {speeds}")
                except Exception as e:
                    print(f"Error: {e}")
    
except KeyboardInterrupt:
    print("Shutting down...")
    model.save(MODEL_PATH)
    ser.close()
