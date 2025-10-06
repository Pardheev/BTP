import numpy as np
import pandas as pd

# Import functions from your new modules
from simulation.config import DECISION_RECIPES
from analysis.scoring import (
    calculate_volatility_score,
    get_criticality_scores,
    categorize_scores,
)
from mdp.solver import solve_mdp

def generate_mock_data(parameters):
    """A helper function to create mock data for the simulation."""
    print("\nStep 1: Generating mock data for all parameters...")
    time_series_data = {p: np.random.normal(50, 10, 1000) for p in parameters}
    
    accident_df_data = {p: np.random.rand(500) for p in parameters}
    accident_df_data['Accident_Severity'] = np.random.randint(0, 3, 500)
    accident_data = pd.DataFrame(accident_df_data)
    
    print("Mock data generated.")
    return time_series_data, accident_data

def main():
    """Main orchestration script for the offline policy generation phase."""
    print("--- Starting Phase 1: Offline Policy Generation ---")

    # Dynamically create the parameter list from the recipes
    all_params_set = {param for recipe in DECISION_RECIPES.values() for param in recipe['parameters']}
    ALL_PARAMETERS = sorted(list(all_params_set))
    print(f"\nSuccessfully identified {len(ALL_PARAMETERS)} unique parameters.")

    # Generate mock data
    time_series_data, accident_data = generate_mock_data(ALL_PARAMETERS)
    
    # --- Part A: Calculate Scores ---
    print("\nStep 2: Calculating Volatility and Criticality scores...")
    volatility_scores = {p: calculate_volatility_score(time_series_data[p]) for p in ALL_PARAMETERS}
    criticality_scores = get_criticality_scores(ALL_PARAMETERS, accident_data)
    
    # --- Part B: Categorize Parameters ---
    print("\nStep 3: Categorizing parameters...")
    volatility_categories = categorize_scores(volatility_scores)
    criticality_categories = categorize_scores(criticality_scores)
    param_classifications = {p: (volatility_categories[p], criticality_categories[p]) for p in ALL_PARAMETERS}
    
    print("Parameter Classifications Complete:")
    for i, (param, cats) in enumerate(param_classifications.items()):
        if i < 5: print(f"  - {param}: (Volatility: {cats[0]}, Criticality: {cats[1]})")

    # --- Part C: Generate the Master Policies ---
    print("\nStep 4: Generating the 9 master policies...")
    ALPHA_BETA_GRID = {
        ('High', 'High'):   (3.0, 1.5), ('High', 'Medium'):   (2.8, 0.8), ('High', 'Low'):   (2.5, 0.3),
        ('Medium', 'High'): (2.0, 1.2), ('Medium', 'Medium'): (1.8, 0.5), ('Medium', 'Low'): (1.5, 0.2),
        ('Low', 'High'):    (1.5, 1.0), ('Low', 'Medium'):    (1.2, 0.4), ('Low', 'Low'):    (1.1, 0.1),
    }

    master_policies = {}
    for category, params in ALPHA_BETA_GRID.items():
        alpha, beta = params
        print(f"  - Solving MDP for category {category} with (alpha={alpha}, beta={beta})...")
        policy = solve_mdp(alpha=alpha, beta=beta)
        master_policies[category] = policy
    
    print("\n--- Phase 1 Complete! ---")
    print(f"{len(master_policies)} master policies have been generated.")
    # In the next phase, you would save or use these policies.
    # For now, we can just print a confirmation.

if __name__ == "__main__":
    main()