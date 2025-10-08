import numpy as np
import math
import random

# ==============================================================================
# HELPER & MOBILITY FUNCTIONS
# ==============================================================================
""" 
The main thing is it will give us the requests in requred format and 
the funciton which helps us to get the location of nearest location of static sensor
which will be used in db
"""
def gauss_markov_mobility_step(user, mean_speed, alpha, bounds):
    """Calculates the next step for a single user using the Gauss-Markov model and returns the updated user dictionary."""
    min_x, max_x, min_y, max_y = bounds
    x_pos, y_pos, speed, direction_rad, mean_direction_deg = \
        user['x'], user['y'], user['speed'], user['direction'], user['mean_direction']

    # Update speed and direction
    speed = (alpha * speed) + ((1 - alpha) * mean_speed) + \
            (math.sqrt(1 - alpha**2) * np.random.normal(0, 1))
    direction_rad = (alpha * direction_rad) + \
                    ((1 - alpha) * math.radians(mean_direction_deg)) + \
                    (math.sqrt(1 - alpha**2) * np.random.normal(0, 1))

    # Update position
    x_pos += speed * math.cos(direction_rad)
    y_pos += speed * math.sin(direction_rad)

    # Boundary handling
    if x_pos > max_x: mean_direction_deg = 180.0; x_pos = max_x
    elif x_pos < min_x: mean_direction_deg = 0.0; x_pos = min_x
    if y_pos > max_y: mean_direction_deg = 270.0; y_pos = max_y
    elif y_pos < min_y: mean_direction_deg = 90.0; y_pos = min_y
    
    # Update user dictionary with new state
    user.update({
        'x': x_pos, 'y': y_pos, 'speed': speed, 
        'direction': direction_rad, 'mean_direction': mean_direction_deg
    })
    return user

def find_nearest_sensor(coords, static_sensors):
    """Finds the nearest static sensor to a given coordinate. Returns the sensor's ID and distance."""
    min_dist = float('inf')
    nearest_sensor_id = -1
    
    for i in range(len(static_sensors['x'])):
        dist = math.sqrt((coords[0] - static_sensors['x'][i])**2 + (coords[1] - static_sensors['y'][i])**2)
        if dist < min_dist:
            min_dist = dist
            nearest_sensor_id = i
            
    return nearest_sensor_id, min_dist

def generate_user_requests(users, time_step):
    """
    Checks which users should make a request at the current time step and returns a list of request data.
    """
    requests_this_step = []
    for user in users:
        if time_step >= user['next_request_time']:
            num_dqs = random.randint(1, 5)
            dq_list = sorted(random.sample(range(1, 26), num_dqs)) # DQs from 1 to 25
            
            requests_this_step.append([
                user['id'],
                round(user['x'], 2),
                round(user['y'], 2),
                time_step,
                dq_list
            ])
            # Set the next request time for this user
            user['next_request_time'] += user['request_interval']
    return requests_this_step

# ==============================================================================
# NEW: MAIN SIMULATION WRAPPER FUNCTION
# ==============================================================================

def run_simulation(num_users, num_sensors, area, duration, mean_speed, alpha):
    """
    Runs the entire mobility and request generation simulation for a single iteration.
    Returns the collected requests and the final state of all nodes.
    """
    # --- 1. Initialization ---
    static_sensors = {
        'x': [random.uniform(area[0], area[1]) for _ in range(num_sensors)],
        'y': [random.uniform(area[2], area[3]) for _ in range(num_sensors)]
    }
    
    end_users = []
    for i in range(num_users):
        end_users.append({
            'id': f"user_{i+1}",
            'x': random.uniform(area[0], area[1]),
            'y': random.uniform(area[2], area[3]),
            'speed': random.uniform(mean_speed * 0.8, mean_speed * 1.2),
            'direction': math.radians(random.uniform(0, 360)),
            'mean_direction': random.uniform(0, 360),
            'request_interval': random.randint(4, 9),
            'next_request_time': random.uniform(0, 10)
        })

    all_generated_requests = []

    # --- 2. Main Simulation Loop ---
    for t in range(duration):
        # Update positions of all users
        for i in range(len(end_users)):
            end_users[i] = gauss_markov_mobility_step(end_users[i], mean_speed, alpha, area)

        # Generate DQ requests for this time step
        requests_this_step = generate_user_requests(end_users, t)
        if requests_this_step:
            all_generated_requests.extend(requests_this_step)
    
    # --- 3. Return Results ---
    return {
        "requests": all_generated_requests,
        "static_sensors": static_sensors,
        "final_user_state": end_users
    }

# ==============================================================================
# EXAMPLE OF HOW TO CALL THE SIMULATION
# ==============================================================================
if __name__ == '__main__':
    # --- Define Simulation Parameters ---
    SIM_PARAMS = {
        "num_users": 50,
        "num_sensors": 450,
        "area": (0, 10000, 0, 10000),
        "duration": 100,
        "mean_speed": 15,
        "alpha": 0.75
    }

    print("--- Starting Simulation ---")
    
    # --- Run the simulation and get the results ---
    simulation_output = run_simulation(**SIM_PARAMS)
    
    print("--- Simulation Complete ---")

    # --- Process and display the returned results ---
    total_requests = len(simulation_output["requests"])
    print(f"\nTotal Decision Queries Generated: {total_requests}")
    
    print("\nExample of the first 5 requests returned:")
    for request in simulation_output["requests"][:5]:
        print(f"  > {request}")

    # You can now pass 'simulation_output["requests"]' to your online phase logic.