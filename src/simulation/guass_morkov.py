import numpy as np
import matplotlib.pyplot as plt
import math
import random

# ==============================================================================
# MOBILITY MODEL (Function remains the same)
# ==============================================================================
def gauss_markov_mobility_step(x_pos, y_pos, speed, direction_rad, mean_speed, mean_direction_deg, alpha, bounds):
    """Calculates the next step for a single node using the Gauss-Markov model."""
    min_x, max_x, min_y, max_y = bounds
    rand_speed = np.random.normal(0, 1)
    rand_direction = np.random.normal(0, 1)
    speed = (alpha * speed) + ((1 - alpha) * mean_speed) + (math.sqrt(1 - alpha**2) * rand_speed)
    direction_rad = (alpha * direction_rad) + ((1 - alpha) * math.radians(mean_direction_deg)) + (math.sqrt(1 - alpha**2) * rand_direction)
    x_pos += speed * math.cos(direction_rad)
    y_pos += speed * math.sin(direction_rad)
    if x_pos > max_x:
        mean_direction_deg = 180.0
        x_pos = max_x
    elif x_pos < min_x:
        mean_direction_deg = 0.0
        x_pos = min_x
    if y_pos > max_y:
        mean_direction_deg = 270.0
        y_pos = max_y
    elif y_pos < min_y:
        mean_direction_deg = 90.0
        y_pos = min_y
    return x_pos, y_pos, speed, direction_rad, mean_direction_deg

# ==============================================================================
# MAIN SIMULATION SCRIPT
# ==============================================================================
if __name__ == '__main__':
    # --- Simulation Parameters ---
    NUM_END_USERS = 50 
    NUM_STATIC_SENSORS = 450
    SIMULATION_AREA = (0, 10000, 0, 10000)
    SIMULATION_TIME_STEPS = 900
    
    # --- Number of Iterations ---
    # This is the main loop for running the entire simulation multiple times.
    NUM_ITERATIONS = 100 
    
    # Gauss-Markov mobility model parameters
    ALPHA = 0.75
    MEAN_SPEED_MPS = 15
    
    # A list to store results from all runs for statistical analysis later
    all_runs_results = []

    # ==========================================================================
    # WHERE THE ITERATION LOOP IS ADDED
    # ==========================================================================
    for i in range(NUM_ITERATIONS):
        # The print statement now correctly shows the current iteration number
        print(f"--- Starting Simulation Run ({i + 1} of {NUM_ITERATIONS}) ---")
        
        # --- Sensor Deployment (re-initialized for each independent run) ---
        static_sensors = {
            'x': [random.uniform(SIMULATION_AREA[0], SIMULATION_AREA[1]) for _ in range(NUM_STATIC_SENSORS)],
            'y': [random.uniform(SIMULATION_AREA[2], SIMULATION_AREA[3]) for _ in range(NUM_STATIC_SENSORS)]
        }
        
        end_users = []
        for j in range(NUM_END_USERS):
            user = {
                'id': j,
                'x': random.uniform(SIMULATION_AREA[0], SIMULATION_AREA[1]),
                'y': random.uniform(SIMULATION_AREA[2], SIMULATION_AREA[3]),
                'speed': random.uniform(MEAN_SPEED_MPS * 0.5, MEAN_SPEED_MPS * 1.5),
                'direction': math.radians(random.uniform(0, 360)),
                'mean_direction': random.uniform(0, 360),
                'path_x': [],
                'path_y': []
            }
            end_users.append(user)

        # --- Main Simulation Loop (for time steps) ---
        for t in range(SIMULATION_TIME_STEPS):
            # In your project, you would add your DQ-Map logic here for each time step
            for user in end_users:
                new_x, new_y, new_speed, new_dir, new_mean_dir = gauss_markov_mobility_step(
                    user['x'], user['y'], user['speed'], user['direction'],
                    MEAN_SPEED_MPS, user['mean_direction'], ALPHA, SIMULATION_AREA
                )
                user['x'], user['y'], user['speed'], user['direction'], user['mean_direction'] = new_x, new_y, new_speed, new_dir, new_mean_dir
                user['path_x'].append(new_x)
                user['path_y'].append(new_y)
        
        print(f"--- Simulation Run {i + 1} Complete ---")
        # Optional: Store the final state of this run for later analysis
        all_runs_results.append(end_users)

    print("\nAll simulation iterations are complete.")
    
    # --- Plotting the results of the FINAL (100th) simulation run as an example ---
    print("Plotting the results of the final simulation run...")
    plt.figure(figsize=(10, 10))
    
    final_run_users = all_runs_results[-1] # Get the data from the last run
    
    plt.scatter(static_sensors['x'], static_sensors['y'], c='red', marker='^', label='Static Sensors')
    
    for i, user in enumerate(final_run_users):
        if i < 5: 
            plt.plot(user['path_x'], user['path_y'], linestyle='-', label=f'End-User {user["id"]+1} Path')

    plt.title(f'Simulation Snapshot (Run {NUM_ITERATIONS}/{NUM_ITERATIONS})')
    plt.xlabel('X-coordinate (m)')
    plt.ylabel('Y-coordinate (m)')
    plt.xlim(SIMULATION_AREA[0], SIMULATION_AREA[1])
    plt.ylim(SIMULATION_AREA[2], SIMULATION_AREA[3])
    plt.grid(True)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()