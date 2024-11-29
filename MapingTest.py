import serial
import json
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from time import sleep
import os

# Define paths for saving and loading the model
MODEL_PATH = "./trained_maped_model.keras"

# Define map size and initialize
MAP_SIZE = 20  # Map will be MAP_SIZE x MAP_SIZE
GRID_RESOLUTION = 10  # Each grid cell corresponds to 10 cm
map_grid = np.zeros((MAP_SIZE, MAP_SIZE), dtype=int)  # 0 = unexplored, 1 = free, -1 = occupied

# Robot's position in the map (start at the center)
robot_position = (MAP_SIZE // 2, MAP_SIZE // 2)


def initialize_model():
    """Load or create the model."""
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


def update_map(robot_pos, distance, direction_code):
    """Updates the map based on the robot's position, distance, and direction code."""
    global map_grid
    x, y = robot_pos

    # Define direction lookup table for grid offsets
    DIRECTION_OFFSETS = {
        0: (0, +1),   # Forward
        1: (0, -1),   # Backward
        2: (0, 0),    # Turn right (no positional change)
        3: (0, 0),    # Turn left (no positional change)
        4: (-1, 0),   # Translate left
        5: (+1, 0),   # Translate right
        6: (-1, +1),  # Diagonal up-left
        7: (+1, +1),  # Diagonal up-right
        8: (-1, -1),  # Diagonal down-left
        9: (+1, -1)   # Diagonal down-right
    }

    # Get the (dx, dy) for the given direction code
    dx, dy = DIRECTION_OFFSETS.get(direction_code, (0, 0))

    # Scale the offsets by distance and grid resolution
    grid_dx = int((distance * dx) / GRID_RESOLUTION)
    grid_dy = int((distance * dy) / GRID_RESOLUTION)

    # Mark cells along the path as free until the obstacle
    for i in range(1, abs(grid_dx) + 1):
        nx, ny = x + i * int(np.sign(grid_dx)), y
        if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
            map_grid[nx, ny] = 1  # Free space

    for j in range(1, abs(grid_dy) + 1):
        nx, ny = x, y + j * int(np.sign(grid_dy))
        if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
            map_grid[nx, ny] = 1  # Free space

    # Mark the obstacle cell
    obstacle_x, obstacle_y = x + grid_dx, y + grid_dy
    if 0 <= obstacle_x < MAP_SIZE and 0 <= obstacle_y < MAP_SIZE:
        map_grid[obstacle_x, obstacle_y] = -1  # Obstacle



def print_map():
    """Prints the current map to the console."""
    for row in map_grid:
        print(' '.join(['O' if cell == -1 else '.' if cell == 1 else '#' for cell in row]))
    print("\n")


# Load the model
model = initialize_model()

# Bluetooth Serial setup
bluetooth_port = 'COM12'  # Adjust this to match your Bluetooth port
baud_rate = 9600
ser = serial.Serial(bluetooth_port, baud_rate, timeout=1)


def predict_speeds(inputs, explore=True):
    """Make predictions for wheel speeds and direction."""
    predictions = model.predict(inputs, verbose=0)[0]  # Get predictions
    if explore:
        predictions += np.random.normal(0, 0.1, size=predictions.shape)  # Add exploration noise
    predictions = np.clip(predictions, 0, 1)  # Ensure predictions are between 0 and 1

    # Compute average speed for the first four predictions (representing wheels)
    averageSpeed = sum(predictions[:4]) / 4  # Correctly average the first 4 predictions

    # Scale wheel speeds proportionally to averageSpeed (e.g., up to 255)
    max_wheel_speed = 200
    speeds = {f"W{i+1}": max(140, int(averageSpeed * max_wheel_speed)) for i in range(4)}

    # Direction is scaled to [0, 9]
    speeds["D"] = int(predictions[-1] * 9)

    return speeds, predictions


def update_model(inputs, predictions, reward):
    """Update the model based on the reward and predictions."""
    targets = np.clip(predictions + 0.01 * reward * (predictions - 0.5), 0, 1)
    model.fit(inputs, np.array([targets]), verbose=0, callbacks=None)


def get_surrounding_map_features(position):
    """Extracts surrounding map information for ML input."""
    x, y = position
    surroundings = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE:
            surroundings.append(map_grid[nx, ny])
        else:
            surroundings.append(-1)  # Out-of-bounds as obstacle
    return surroundings


def calculate_reward(distance, car_collision, speeds, current_direction, target_direction, explored_area):
    """Calculate the reward with exploration enforcement."""
    reward = 0
    direction = speeds["D"]

    if car_collision:
        reward -= 10  # Penalize collisions heavily
    else:
        if distance > 10 and distance < 150:
            reward += 1.0 * (direction + 1) / (distance + 1)
            direction_alignment = max(0, 1 - abs(target_direction - current_direction) / 9)
            reward += direction_alignment * 0.5

        # Encourage exploration
        if explored_area:
            reward += 5  # Reward for exploring new areas
        else:
            reward -= 0.5

    print(f"Calculated Reward: {reward}")
    return reward


def update_exploration_status(position):
    """Checks if the robot explored a new area and updates the map."""
    x, y = position
    if 0 <= x < MAP_SIZE and 0 <= y < MAP_SIZE:
        if map_grid[x, y] == 0:  # Unexplored area
            map_grid[x, y] = 1  # Mark as explored
            return True
    return False


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
                    car_collision = (distance < 5 and current_direction in [0, 6, 7])

                    # Update exploration status
                    explored_area = update_exploration_status(robot_position)

                    # Get map features for ML input
                    map_features = get_surrounding_map_features(robot_position)
                    inputs = np.array([[current_direction, target_direction, distance] + map_features])

                    # Predict and train model
                    speeds, predictions = predict_speeds(inputs)
                    reward = calculate_reward(distance, car_collision, speeds, current_direction, target_direction, explored_area)
                    update_model(inputs, predictions, reward)

                    # Update the map
                    update_map(robot_position, distance, speeds["D"])

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