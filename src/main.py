import numpy as np
import json
from collections import defaultdict
import pandas as pd # Import pandas for easier handling of complex averages

# --- Import all necessary functions from your project modules ---
from mdp.solver import get_or_generate_brain
from simulation.online_simulator import run_single_online_iteration
from simulation.guass_morkov import run_simulation # Need this to pre-run for request counts

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
    master_policies, param_classifications = get_or_generate_brain()
    print("--- Offline Phase Complete ---")

    # ==========================================================================
    # PHASE 2: ONLINE SIMULATION EXPERIMENT
    # ==========================================================================
    print("\n--- Running Online Phase ---")
    
    # --- Define the Experiment Parameters ---
    user_counts_to_simulate = [50, 75, 100, 125, 150, 175, 200] # UPDATED USER COUNTS
    iterations_per_count = 100  
    
    # This dictionary will store all final results, ready for plotting
    experiment_results = {
        "vs_users": {},
        "vs_time": {},
        "vs_time_requests": {} # NEW: For requests vs time
    }

    # --- Run the Full Experiment ---
    for num_users in user_counts_to_simulate:
        print(f"\n--- Starting Experiment for {num_users} Users ({iterations_per_count} iterations) ---")
        
        # Lists to store the totals from each of the 100 iterations
        run_totals = {
            "sensor_accesses": [],
            "energy_consumed": [],
            "avg_aoi": [], 
        }
        # Dicts to aggregate time-series data across the 100 iterations
        time_series_agg = {
            "accesses": defaultdict(list), 
            "energy": defaultdict(list),
            "requests": defaultdict(list) # NEW: Aggregate request counts
        }
        
        for i in range(iterations_per_count):
            # We need to run the guass_morkov simulation to get requests BEFORE running the online simulator
            # This is necessary because we need the raw request count for the "Requests vs Time" plot.
            simulation_events = run_simulation(
                num_users=num_users, num_sensors=450, area=(0, 10000, 0, 10000),
                duration=100, mean_speed=15, alpha=0.75
            )
            
            # Run one full simulation from the online_simulator module using the generated events
            result = run_single_online_iteration(
                num_users, master_policies, param_classifications, 
                simulation_events=simulation_events # Pass events directly
            )
            
            # --- Aggregate and store the request count data ---
            request_counts_in_run = result["requests_over_time"] # Now available from online_simulator
            for t, val in request_counts_in_run.items():
                time_series_agg["requests"][t].append(val)
            
            # Store the total metrics for this iteration
            run_totals["sensor_accesses"].append(result["total_sensor_accesses"])
            run_totals["energy_consumed"].append(result["total_energy_consumed"])
            run_totals["avg_aoi"].append(result["avg_aoi_for_qos"])
            
            # Append the time-series data for this run
            for t, val in result["accesses_over_time"].items():
                time_series_agg["accesses"][t].append(val)
            for t, val in result["energy_over_time"].items():
                time_series_agg["energy"][t].append(val)
            
            # print(f"  Iteration {i+1}/{iterations_per_count} complete.") # Keep simulation progress visible

        # --- Aggregate and store the results for this user count ---
        # Data for "vs Users" plot (Sensor Access, Energy, AoI)
        experiment_results["vs_users"][num_users] = {
            "avg_sensor_accesses": np.mean(run_totals["sensor_accesses"]),
            "std_sensor_accesses": np.std(run_totals["sensor_accesses"]),
            "avg_energy_consumed": np.mean(run_totals["energy_consumed"]),
            "std_energy_consumed": np.std(run_totals["energy_consumed"]),
            "avg_aoi_for_qos": np.mean(run_totals["avg_aoi"]),
            "std_aoi_for_qos": np.std(run_totals["avg_aoi"])
        }
        
        # Data for "vs Time" plot (Sensor Access, Energy)
        sorted_time_steps = sorted(time_series_agg["accesses"].keys())
        experiment_results["vs_time"][num_users] = {
            "time_steps": sorted_time_steps,
            "avg_sensor_accesses_over_time": [np.mean(time_series_agg["accesses"][t]) for t in sorted_time_steps],
            "avg_energy_over_time": [np.mean(time_series_agg["energy"][t]) for t in sorted_time_steps]
        }
        
        # Data for "Requests vs Time" plot
        sorted_req_time_steps = sorted(time_series_agg["requests"].keys())
        experiment_results["vs_time_requests"][num_users] = {
            "time_steps": sorted_req_time_steps,
            "avg_requests_over_time": [np.mean(time_series_agg["requests"][t]) for t in sorted_req_time_steps]
        }
    
    print("\n--- All Experiments Complete ---")
    
    # Save the final, aggregated results to a file for plotting
    with open('simulation_results_for_plotting.json', 'w') as f:
        json.dump(experiment_results, f, indent=2)
    print("\nFinal results saved to 'simulation_results_for_plotting.json'")
    print("This file contains all the data you need for your graphs. 📈")


if __name__ == "__main__":
    # Ensure necessary modules are imported correctly when running main
    from simulation.guass_morkov import run_simulation # Required by main() for event generation
    main()