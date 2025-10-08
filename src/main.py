# File: sem7/src/main.py

import numpy as np
import json
from collections import defaultdict

# --- Import all necessary functions from your project modules ---
from mdp.solver import get_or_generate_brain
from simulation.online_simulator import run_single_online_iteration

def main():
    """
    Main orchestration script. Runs the entire pipeline:
    1. Offline Phase: Generates or loads the MDP policies ("the brain").
    2. Online Phase: Runs the full simulation experiment using the brain.
    3. Results: Saves the final aggregated data for plotting.
    """
    # ==========================================================================
    # PHASE 1: OFFLINE BRAIN GENERATION (or loading from file)
    # ==========================================================================
    print("--- Running Offline Phase ---")
    # This single function handles everything for the offline part
    master_policies, param_classifications = get_or_generate_brain()
    print("--- Offline Phase Complete ---")

    # ==========================================================================
    # PHASE 2: ONLINE SIMULATION EXPERIMENT
    # ==========================================================================
    print("\n--- Running Online Phase ---")
    
    # --- Define the Experiment Parameters ---
    user_counts_to_simulate = [50, 75, 100,125, 150, 175, 200]
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
            # Run one full simulation from the online_simulator module
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
        sorted_time_steps = sorted(time_series_agg["accesses"].keys())
        experiment_results["vs_time"][num_users] = {
            "time_steps": sorted_time_steps,
            "avg_accesses_over_time": [np.mean(time_series_agg["accesses"][t]) for t in sorted_time_steps],
            "avg_energy_over_time": [np.mean(time_series_agg["energy"][t]) for t in sorted_time_steps]
        }
    
    print("\n--- All Experiments Complete ---")
    
    # Save the final, aggregated results to a file for plotting
    with open('simulation_results_for_plotting.json', 'w') as f:
        json.dump(experiment_results, f, indent=2)
    print("\nFinal results saved to 'simulation_results_for_plotting.json'")
    print("This file contains all the data you need for your graphs. 📈")


if __name__ == "__main__":
    main()