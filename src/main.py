import numpy as np
import pandas as pd
import json

# Import the new function from your solver
from mdp.solver import generate_and_save_master_policies 
from simulation.config import DECISION_RECIPES
from analysis.scoring import (
    calculate_volatility_score,
    get_criticality_scores,
    categorize_scores,
)

def generate_mock_data(parameters):
    """A helper function to create mock data for the simulation."""
    print("\nStep 1: Generating mock data for all parameters...")
    # (This function is unchanged)
    time_series_data = {p: np.random.normal(50, 10, 1000) for p in parameters}
    accident_df_data = {p: np.random.rand(500) for p in parameters}
    accident_df_data['Accident_Severity'] = np.random.randint(0, 3, 500)
    accident_data = pd.DataFrame(accident_df_data)
    print("Mock data generated.")
    return time_series_data, accident_data

def main():
    """Main orchestration script for the offline policy generation phase."""
    print("--- Starting Phase 1: Offline Policy Generation ---")

    all_params_set = {param for recipe in DECISION_RECIPES.values() for param in recipe['parameters']}
    ALL_PARAMETERS = sorted(list(all_params_set))
    print(f"\nSuccessfully identified {len(ALL_PARAMETERS)} unique parameters.")

    time_series_data, accident_data = generate_mock_data(ALL_PARAMETERS)
    
    # --- Part A & B: Analyze and Categorize Parameters ---
    print("\nStep 2 & 3: Analyzing and Categorizing Parameters...")
    volatility_scores = {p: calculate_volatility_score(time_series_data[p]) for p in ALL_PARAMETERS}
    criticality_scores = get_criticality_scores(ALL_PARAMETERS, accident_data)
    volatility_categories = categorize_scores(volatility_scores)
    criticality_categories = categorize_scores(criticality_scores)
    param_classifications = {p: (volatility_categories[p], criticality_categories[p]) for p in ALL_PARAMETERS}
    print("Parameter Classifications Complete.")

    # --- Part C: Define the Alpha-Beta Grid ---
    ALPHA_BETA_GRID = {
        ('High', 'High'):   (3.0, 1.5), ('High', 'Medium'):   (2.8, 0.8), ('High', 'Low'):   (2.5, 0.3),
        ('Medium', 'High'): (2.0, 1.2), ('Medium', 'Medium'): (1.8, 0.5), ('Medium', 'Low'): (1.5, 0.2),
        ('Low', 'High'):    (1.5, 1.0), ('Low', 'Medium'):    (1.2, 0.4), ('Low', 'Low'):    (1.1, 0.1),
    }

    # ==============================================================================
    # <--- REFACTORED SECTION: CALL SOLVER TO GENERATE AND SAVE POLICIES --- >
    # ==============================================================================
    
    # Call the function from solver.py to generate and save the master policies
    generate_and_save_master_policies(ALPHA_BETA_GRID, save_path="master_policies.json")

    # --- Save the parameter classifications (this logic stays in main.py) ---
    print("\nStep 5: Saving parameter classifications...")
    with open('param_classifications.json', 'w') as f:
        json.dump(param_classifications, f, indent=2)
    print("  - Saved 'param_classifications.json' 💾")
    print("\nOffline phase is complete. The brain is ready for the online simulator.")

if __name__ == "__main__":
    main()