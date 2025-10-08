import numpy as np
import json
import os
import pandas as pd

# This file now needs to know about the analysis and config to be self-contained
from analysis.scoring import (
    calculate_volatility_score,
    get_criticality_scores,
    categorize_scores,
)
from simulation.config import DECISION_RECIPES

def solve_mdp(alpha, beta, resource_cost=50, max_aoi=100, gamma=0.95, epsilon=1e-4):
    """Solves for the optimal caching policy using Value Iteration. (Unchanged)"""
    # ... (code for this function is the same as before) ...
    states = np.arange(1, max_aoi + 1)
    num_states = len(states)
    V = np.zeros(num_states)
    
    while True:
        delta = 0
        for s_idx, s_aoi in enumerate(states):
            cost_cache = beta * (s_aoi ** alpha)
            next_s_cache_idx = min(s_idx + 1, num_states - 1)
            v_cache = cost_cache + gamma * V[next_s_cache_idx]
            cost_fetch = resource_cost
            next_s_fetch_idx = 0
            v_fetch = cost_fetch + gamma * V[next_s_fetch_idx]
            v_s_old = V[s_idx]
            V[s_idx] = min(v_cache, v_fetch)
            delta = max(delta, abs(v_s_old - V[s_idx]))
        if delta < epsilon:
            break
            
    policy = np.zeros(num_states, dtype=int)
    for s_idx, s_aoi in enumerate(states):
        cost_cache = beta * (s_aoi ** alpha)
        next_s_cache_idx = min(s_idx + 1, num_states - 1)
        v_cache = cost_cache + gamma * V[next_s_cache_idx]
        cost_fetch = resource_cost
        next_s_fetch_idx = 0
        v_fetch = cost_fetch + gamma * V[next_s_fetch_idx]
        policy[s_idx] = 1 if v_fetch < v_cache else 0
            
    return policy

# ==============================================================================
# <--- NEW MASTER FUNCTION TO GET OR GENERATE THE BRAIN --- >
# ==============================================================================
def get_or_generate_brain(policy_path="master_policies.json", classification_path="param_classifications.json"):
    """
    Checks if policy and classification files exist. If yes, loads them.
    If not, it runs the entire offline generation process and saves the files.
    
    Returns:
        tuple: (master_policies, param_classifications)
    """
    print("--- Checking for existing brain files... ---")
    
    # Check if both files exist
    if os.path.exists(policy_path) and os.path.exists(classification_path):
        print("Brain files found! Loading from cache... 🧠")
        with open(policy_path, 'r') as f:
            master_policies_str_keys = json.load(f)
            master_policies = {eval(k): np.array(v) for k, v in master_policies_str_keys.items()}
        with open(classification_path, 'r') as f:
            param_classifications = json.load(f)
        
        print("Brain loaded successfully.")
        return master_policies, param_classifications

    # --- If files don't exist, generate everything ---
    print("Brain files not found. Starting offline generation process...")

    # 1. Get all parameters
    all_params_set = {param for recipe in DECISION_RECIPES.values() for param in recipe['parameters']}
    ALL_PARAMETERS = sorted(list(all_params_set))

    # 2. Generate mock data (or load real data)
    time_series_data = {p: np.random.normal(50, 10, 1000) for p in ALL_PARAMETERS}
    accident_df_data = {p: np.random.rand(500) for p in ALL_PARAMETERS}
    accident_df_data['Accident_Severity'] = np.random.randint(0, 3, 500)
    accident_data = pd.DataFrame(accident_df_data)
    
    # 3. Analyze and categorize parameters
    print("Analyzing and categorizing parameters...")
    volatility_scores = {p: calculate_volatility_score(time_series_data[p]) for p in ALL_PARAMETERS}
    criticality_scores = get_criticality_scores(ALL_PARAMETERS, accident_data)
    param_classifications = {p: (categorize_scores(volatility_scores)[p], categorize_scores(criticality_scores)[p]) for p in ALL_PARAMETERS}

    # 4. Generate master policies
    print("Generating master policies...")
    ALPHA_BETA_GRID = {
        ('High', 'High'):   (3.0, 1.5), ('High', 'Medium'):   (2.8, 0.8), ('High', 'Low'):   (2.5, 0.3),
        ('Medium', 'High'): (2.0, 1.2), ('Medium', 'Medium'): (1.8, 0.5), ('Medium', 'Low'): (1.5, 0.2),
        ('Low', 'High'):    (1.5, 1.0), ('Low', 'Medium'):    (1.2, 0.4), ('Low', 'Low'):    (1.1, 0.1),
    }
    master_policies = {}
    for category, params in ALPHA_BETA_GRID.items():
        alpha, beta = params
        policy = solve_mdp(alpha=alpha, beta=beta)
        master_policies[category] = policy
        
    # 5. Save the generated brain to files for next time
    print("Saving brain files... 💾")
    with open(classification_path, 'w') as f:
        json.dump(param_classifications, f, indent=2)
    
    policies_to_save = {str(key): policy.tolist() for key, policy in master_policies.items()}
    with open(policy_path, 'w') as f:
        json.dump(policies_to_save, f, indent=2)
        
    print("Brain generation and saving complete.")
    return master_policies, param_classifications