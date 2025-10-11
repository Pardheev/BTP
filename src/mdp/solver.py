import numpy as np
import json
import os
import pandas as pd
from collections import defaultdict
import math # Import math for simple utilities

# --- Import project-specific functions ---
from analysis.scoring import (
    calculate_volatility_score,
    get_criticality_scores,
    categorize_scores,
)
from simulation.config import DECISION_RECIPES

# ==============================================================================
# HELPER FUNCTIONS FOR ALGORITHM 2 (Category-Based MDP Policy Generation)
# ==============================================================================

def safe_average(scores):
    """Calculates average, safely handling empty lists."""
    return sum(scores) / len(scores) if scores else 0.0

def scale_score(score, min_score, max_score, min_range, max_range):
    """
    Linearly scales a single score from its original min/max range 
    to a new target range (e.g., scaling criticality scores to Beta values).
    
    Equivalent to the Algorithm 2 step: 'Scale(...)'.
    """
    # Check if the score range is zero (all scores are identical)
    if max_score == min_score:
        # Return the bottom of the target range, or the desired default.
        return min_range 
        
    # Linearly scale the score to the new range
    # new_value = new_min + (old_value - old_min) * (new_max - new_min) / (old_max - old_min)
    return min_range + (score - min_score) * (max_range - min_range) / (max_score - min_score)

# ==============================================================================
# MDP SOLVER (Original code preserved)
# ==============================================================================

def solve_mdp(alpha, beta, resource_cost=50, max_aoi=100, gamma=0.95, epsilon=1e-4):
    """Solves for the optimal caching policy using Value Iteration."""
    # ... (code for solve_mdp remains as it was) ...
    states = np.arange(1, max_aoi + 1)
    num_states = len(states)
    V = np.zeros(num_states)
    
    while True:
        delta = 0
        for s_idx, s_aoi in enumerate(states):
            # Cost function: Cost(s_AoI) = beta * (s_AoI)**alpha
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
# MASTER FUNCTION TO GET OR GENERATE THE BRAIN (Modified for Algorithm 2)
# ==============================================================================
def get_or_generate_brain(policy_path="master_policies.json", classification_path="param_classifications.json"):
    """
    Checks if policy and classification files exist. If yes, loads them.
    If not, it runs the entire offline generation process and saves the files.
    
    Returns:
        tuple: (master_policies, param_classifications)
    """
    print("--- Checking for existing brain files... ---")
    
    # Check if both files exist (Loading logic remains the same)
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
    # This section now corresponds to Algorithm 1 (Offline Parameter Characterization)
    print("Analyzing and categorizing parameters...")
    volatility_scores = {p: calculate_volatility_score(time_series_data[p]) for p in ALL_PARAMETERS} # S_vol
    criticality_scores = get_criticality_scores(ALL_PARAMETERS, accident_data) # S_crit
    
    # Generate the classification map C_map
    vol_class = categorize_scores(volatility_scores)
    crit_class = categorize_scores(criticality_scores)
    param_classifications = {p: (vol_class[p], crit_class[p]) for p in ALL_PARAMETERS} 

    # --- START OF ALGORITHM 2 LOGIC ---
    print("Generating master policies dynamically...")

    # 4a. Define Target Output Ranges (alpha_min/max, beta_min/max)
    # Based on typical MDP applications and the prior hardcoded grid values (1.1 to 3.0 for alpha, 0.1 to 1.5 for beta)
    ALPHA_RANGE = (1.1, 3.0) 
    BETA_RANGE = (0.1, 1.5)

    # 4b. Find Global Min/Max boundaries of the raw scores for normalization
    all_vol_scores = list(volatility_scores.values())
    all_crit_scores = list(criticality_scores.values())
    min_vol, max_vol = min(all_vol_scores) if all_vol_scores else 0, max(all_vol_scores) if all_vol_scores else 0
    min_crit, max_crit = min(all_crit_scores) if all_crit_scores else 0, max(all_crit_scores) if all_crit_scores else 0

    # 4c. Group parameters by their unique category tuple (C_map[p] = c)
    # Categories now acts as the set of 9 unique tuples (Algorithm 2, step: Let Categories be...)
    categories_to_params = defaultdict(list)
    for param, category_tuple in param_classifications.items():
        categories_to_params[category_tuple].append(param)
        
    master_policies = {}

    # 4d. Loop through unique categories (Algorithm 2, step: FOR each category c)
    for category_tuple, param_list in categories_to_params.items():
        
        # --- Algorithm 2, Step: Calculate the average scores (Average) ---
        avg_vol_for_category = safe_average([volatility_scores[p] for p in param_list])
        avg_crit_for_category = safe_average([criticality_scores[p] for p in param_list])
        
        # --- Algorithm 2, Step: Scale the average scores (Scale) ---
        alpha_c = scale_score(
            avg_vol_for_category, min_vol, max_vol, *ALPHA_RANGE
        )
        beta_c = scale_score(
            avg_crit_for_category, min_crit, max_crit, *BETA_RANGE
        )

        # --- Algorithm 2, Step: Solve the MDP using Value Iteration (SolveMDP) ---
        # Note: The MDP solver implicitly uses the derived alpha/beta to define the cost function:
        # Cost(s_AoI) <- beta_c * (s_AoI)**alpha_c
        policy = solve_mdp(alpha=alpha_c, beta=beta_c)
        
        # --- Algorithm 2, Step: Store the resulting optimal policy ---
        master_policies[category_tuple] = policy
        
    # 5. Save the generated brain to files for next time
    print("Saving brain files... 💾")
    with open(classification_path, 'w') as f:
        json.dump(param_classifications, f, indent=2)
    
    policies_to_save = {str(key): policy.tolist() for key, policy in master_policies.items()}
    with open(policy_path, 'w') as f:
        json.dump(policies_to_save, f, indent=2)
        
    print("Brain generation and saving complete.")
    return master_policies, param_classifications