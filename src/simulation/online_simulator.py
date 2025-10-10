# File: sem7/src/simulation/online_simulator.py

import numpy as np
import math
import random
from collections import defaultdict

# --- Import project-specific functions ---
from .guass_morkov import run_simulation, find_nearest_sensor 
from .config import DECISION_RECIPES

# --- Simulation Cost & Time Parameters ---
COST_SENSOR_FETCH_ENERGY = 10.0
COST_CACHE_LOOKUP_ENERGY = 0.2
TIME_SENSOR_FETCH_MS = 50
TIME_CACHE_LOOKUP_MS = 5
MAX_AOI = 100

def get_params_for_dq(dq_number):
    """Helper function to map a DQ number to its required parameters."""
    recipe_key = list(DECISION_RECIPES.keys())[dq_number % len(DECISION_RECIPES)]
    return DECISION_RECIPES[recipe_key]["parameters"]

def run_single_online_iteration(num_users, master_policies, param_classifications):
    """
    Runs ONE full online simulation iteration and returns the detailed performance metrics.
    """
    # --- 1. Generate all simulation events ---
    simulation_events = run_simulation(
        num_users=num_users, num_sensors=450, area=(0, 10000, 0, 10000),
        duration=100, mean_speed=15, alpha=0.75
    )
    all_requests = simulation_events["requests"]
    static_sensors = simulation_events["static_sensors"]

    # --- 2. Initialize State for this Run ---
    db = {}  # The database: {(sensor_id, param_name): aoi}

    # Performance metric counters
    total_sensor_accesses = 0
    total_energy_consumed = 0
    total_decisions_made = 0
    
    # NEW: List to store the AoI for every parameter used in a decision
    aoi_at_decision_time = []

    accesses_over_time = defaultdict(int)
    energy_over_time = defaultdict(float)

    # --- 3. Process All Requests Chronologically ---
    for request in all_requests:
        user_id, user_x, user_y, time_step, dq_list = request
        user_coords = (user_x, user_y)
        nearest_sensor_id, _ = find_nearest_sensor(user_coords, static_sensors)
        
        for dq in dq_list:
            total_decisions_made += 1
            required_params = get_params_for_dq(dq)
            
            for param in required_params:
                db_key = (nearest_sensor_id, param)
                
                # Default to a high AoI if the parameter has never been seen
                current_aoi = db.get(db_key, MAX_AOI)
                
                # ==========================================================
                # <--- NEW: RECORD THE AOI FOR QOS MEASUREMENT --- >
                # ==========================================================
                aoi_at_decision_time.append(current_aoi)

                # --- (The rest of the MDP logic is the same) ---
                if db_key not in db:
                    action = 1
                else:
                    category = tuple(param_classifications.get(param))
                    
                    if category and category in master_policies:
                        policy = master_policies[category]
                        action_index = min(current_aoi - 1, len(policy) - 1)
                        action = policy[action_index]
                    else:
                        action = 1

                if action == 1: # FETCH
                    total_sensor_accesses += 1
                    total_energy_consumed += COST_SENSOR_FETCH_ENERGY
                    accesses_over_time[time_step] += 1
                    energy_over_time[time_step] += COST_SENSOR_FETCH_ENERGY
                    db[db_key] = 1
                else: # CACHE
                    total_energy_consumed += COST_CACHE_LOOKUP_ENERGY
                    energy_over_time[time_step] += COST_CACHE_LOOKUP_ENERGY
                    db[db_key] = min(current_aoi + 1, MAX_AOI)

    # --- Calculate the final QoS metric ---
    avg_aoi = np.mean(aoi_at_decision_time) if aoi_at_decision_time else 0
    avg_energy_per_decision = total_energy_consumed / total_decisions_made if total_decisions_made > 0 else 0

    return {
        "avg_aoi_for_qos": avg_aoi, # <-- Your new QoS metric
        "total_sensor_accesses": total_sensor_accesses,
        "avg_energy_per_decision": avg_energy_per_decision,
        "total_energy_consumed": total_energy_consumed,
        "accesses_over_time": accesses_over_time,
        "energy_over_time": energy_over_time
    }