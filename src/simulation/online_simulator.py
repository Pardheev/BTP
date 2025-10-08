
"""
now i am coding online simulator.py which will have the values like{# Simulation cost parameters

COST_SENSOR_FETCH_ENERGY = 10.0 

COST_CACHE_LOOKUP_ENERGY = 0.2

TIME_SENSOR_FETCH_MS = 50

TIME_CACHE_LOOKUP_MS = 5}

since in this we need to provide the data which will be required for plotting graphs.

(which i will mention in the following)

it needs to use the offline brain,it need to maintain a db(here db means kind of dictionary) having <<location,parameter>:AoI> when a parameter at that location is used then we the offline brain for the <parameter,AoI>(if we have loc,parameter in the db else direct fetch) and we will do what to so according to the brain.



we need to do 100 iterations(once for 50,100,150,200,250 users) and store the # of sensor access , energy consumed, vs time and vs users and send this data to the main.


"""

import numpy as np
import math
import random
import time
import json
from collections import defaultdict

# --- Import project-specific functions ---
# Ensure you have 'guass_morkov.py' in the same directory or accessible in your Python path
from guass_morkov import run_simulation, find_nearest_sensor 
from simulation.config import DECISION_RECIPES
from mdp.solver import get_or_generate_brain

# --- Simulation Cost & Time Parameters (as you defined) ---
COST_SENSOR_FETCH_ENERGY = 10.0
COST_CACHE_LOOKUP_ENERGY = 0.2
TIME_SENSOR_FETCH_MS = 50
TIME_CACHE_LOOKUP_MS = 5
MAX_AOI = 100  # This should match the max_aoi used in your MDP solver

def get_params_for_dq(dq_number):
    """Helper function to map a DQ number to its required parameters from the recipe book."""
    recipe_key = list(DECISION_RECIPES.keys())[dq_number % len(DECISION_RECIPES)]
    return DECISION_RECIPES[recipe_key]["parameters"]

def run_single_online_iteration(num_users, master_policies, param_classifications):
    """
    Runs ONE full online simulation iteration and returns the detailed performance metrics for that run.
    """
    # --- 1. Generate all simulation events (user movements and requests) ---
    simulation_events = run_simulation(
        num_users=num_users, num_sensors=450, area=(0, 10000, 0, 10000),
        duration=100, mean_speed=15, alpha=0.75
    )
    all_requests = simulation_events["requests"]
    static_sensors = simulation_events["static_sensors"]

    # --- 2. Initialize State for this Run ---
    db = {}  # The database: {(sensor_id, param_name): aoi}
    
    # Performance metrics for this single run
    total_sensor_accesses = 0
    total_energy_consumed = 0
    
    # Data for "vs time" plotting for this single run
    accesses_over_time = defaultdict(int)  # {time_step: num_accesses}
    energy_over_time = defaultdict(float)    # {time_step: energy_consumed}

    # --- 3. Process All Requests Chronologically ---
    for request in all_requests:
        user_id, user_x, user_y, time_step, dq_list = request
        user_coords = (user_x, user_y)
        nearest_sensor_id, _ = find_nearest_sensor(user_coords, static_sensors)
        
        for dq in dq_list:
            required_params = get_params_for_dq(dq)
            
            for param in required_params:
                db_key = (nearest_sensor_id, param)
                
                # Check if we have a record for this param at this location. If not, it's a "cold start" and we must fetch.
                if db_key not in db:
                    action = 1  # 1 = FETCH_SENSOR
                else:
                    # If a record exists, use the MDP policy to decide the action
                    current_aoi = db[db_key]
                    category = tuple(param_classifications.get(param)) # Convert list to tuple for key
                    
                    if category and category in master_policies:
                        policy = master_policies[category]
                        action_index = min(current_aoi - 1, len(policy) - 1)
                        action = policy[action_index]
                    else:
                        action = 1 # Default to fetching if param has no defined policy

                # --- Execute the action and update metrics ---
                if action == 1:  # FETCH
                    total_sensor_accesses += 1
                    total_energy_consumed += COST_SENSOR_FETCH_ENERGY
                    accesses_over_time[time_step] += 1
                    energy_over_time[time_step] += COST_SENSOR_FETCH_ENERGY
                    db[db_key] = 1  # Reset AoI
                else:  # CACHE
                    total_energy_consumed += COST_CACHE_LOOKUP_ENERGY
                    energy_over_time[time_step] += COST_CACHE_LOOKUP_ENERGY
                    db[db_key] = min(db.get(db_key, 0) + 1, MAX_AOI)

    return {
        "total_sensor_accesses": total_sensor_accesses,
        "total_energy_consumed": total_energy_consumed,
        "accesses_over_time": accesses_over_time,
        "energy_over_time": energy_over_time
    }

# ==============================================================================
# MAIN EXECUTION BLOCK - This orchestrates the entire experiment
# ==============================================================================
if __name__ == '__main__':
    # --- Load or Generate the "Brain" ---
    master_policies, param_classifications = get_or_generate_brain()

    # --- Define the Experiment Parameters ---
    user_counts_to_simulate = [50, 100, 150, 200, 250]
    iterations_per_count = 100  # As specified for statistical significance
    
    # This dictionary will store all final results, ready for plotting
    experiment_results = {
        "vs_users": {},
        "vs_time": {}
    }

    # --- Run the Full Experiment ---
    for num_users in user_counts_to_simulate:
        print(f"\n--- Starting Experiment for {num_users} Users ({iterations_per_count} iterations) ---")
        
        # Lists to store the totals from each of the 100 iterations
        run_totals = {"sensor_accesses": [], "energy_consumed": []}
        # Dicts to aggregate time-series data across the 100 iterations
        time_series_agg = {"accesses": defaultdict(list), "energy": defaultdict(list)}
        
        for i in range(iterations_per_count):
            print(f"  Running iteration {i+1}/{iterations_per_count}...")
            result = run_single_online_iteration(num_users, master_policies, param_classifications)
            
            # Store the totals for this run
            run_totals["sensor_accesses"].append(result["total_sensor_accesses"])
            run_totals["energy_consumed"].append(result["total_energy_consumed"])
            
            # Append the time-series data for this run
            for t, val in result["accesses_over_time"].items():
                time_series_agg["accesses"][t].append(val)
            for t, val in result["energy_over_time"].items():
                time_series_agg["energy"][t].append(val)

        # --- Aggregate and store the results for this user count ---
        # Data for "vs Users" plot
        experiment_results["vs_users"][num_users] = {
            "avg_sensor_accesses": np.mean(run_totals["sensor_accesses"]),
            "std_sensor_accesses": np.std(run_totals["sensor_accesses"]),
            "avg_energy_consumed": np.mean(run_totals["energy_consumed"]),
            "std_energy_consumed": np.std(run_totals["energy_consumed"])
        }
        
        # Data for "vs Time" plot
        experiment_results["vs_time"][num_users] = {
            "time_steps": sorted(time_series_agg["accesses"].keys()),
            "avg_accesses_over_time": [np.mean(time_series_agg["accesses"][t]) for t in sorted(time_series_agg["accesses"].keys())],
            "avg_energy_over_time": [np.mean(time_series_agg["energy"][t]) for t in sorted(time_series_agg["energy"].keys())]
        }
    
    print("\n--- All Experiments Complete ---")
    
    # Save the final, aggregated results to a file for plotting
    with open('simulation_results_for_plotting.json', 'w') as f:
        json.dump(experiment_results, f, indent=2)
    print("\nFinal results saved to 'simulation_results_for_plotting.json'")
    print("This file contains all the data you need for your graphs.")