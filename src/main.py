import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ==============================================================================
# --- PART A: SCORE CALCULATION FUNCTIONS ---
# ==============================================================================

def calculate_volatility_score(time_series_data):
    """Calculates a volatility score for a given time-series."""
    data = np.array(time_series_data)
    if len(data) < 2:
        return 0.0
    # Calculate percentage change, add epsilon to avoid division by zero
    percent_change = np.abs((data[1:] - data[:-1]) / (data[:-1] + 1e-9))
    return np.mean(percent_change)

def get_criticality_scores(parameters, accident_data):
    """
    Trains a model to predict accident severity and returns the feature importance
    for each parameter, which we use as the Criticality Score.
    """
    print("Training ML model to determine criticality scores...")
    
    # Separate features (X) from the target (y)
    X = accident_data[parameters]
    y = accident_data['Accident_Severity']
    
    # Train a Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Get feature importances and map them to parameter names
    importances = model.feature_importances_
    criticality_scores = {param: score for param, score in zip(parameters, importances)}
    
    print("ML model trained. Criticality scores extracted.")
    return criticality_scores

# ==============================================================================
# --- PART B: CATEGORIZATION FUNCTION ---
# ==============================================================================

def categorize_scores(scores_dict):
    """Ranks scores and bins them into High, Medium, and Low categories."""
    sorted_params = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
    n = len(sorted_params)
    
    categories = {}
    for i, (param, score) in enumerate(sorted_params):
        if i < n / 3:
            categories[param] = 'High'
        elif i < 2 * n / 3:
            categories[param] = 'Medium'
        else:
            categories[param] = 'Low'
    return categories

# ==============================================================================
# --- PART C: MDP SOLVER FUNCTION ---
# ==============================================================================

def solve_mdp(alpha, beta, resource_cost=50, max_aoi=100, gamma=0.95, epsilon=1e-4):
    """Solves for the optimal caching policy for a given cost function using Value Iteration."""
    states = np.arange(1, max_aoi + 1)
    num_states = len(states)
    V = np.zeros(num_states)
    
    while True:
        delta = 0
        for s_idx, s_aoi in enumerate(states):
            # Cost of action 0 (USE_CACHE)
            cost_cache = beta * (s_aoi ** alpha)
            next_s_cache_idx = min(s_idx + 1, num_states - 1)
            v_cache = cost_cache + gamma * V[next_s_cache_idx]

            # Cost of action 1 (FETCH_SENSOR)
            cost_fetch = resource_cost
            next_s_fetch_idx = 0  # AoI resets to 1 (index 0)
            v_fetch = cost_fetch + gamma * V[next_s_fetch_idx]

            v_s_old = V[s_idx]
            V[s_idx] = min(v_cache, v_fetch)
            delta = max(delta, abs(v_s_old - V[s_idx]))

        if delta < epsilon:
            break
            
    # Derive the final policy from the converged value function
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
# --- MAIN EXECUTION SCRIPT ---
# ==============================================================================

if __name__ == "__main__":
    print("--- Starting Phase 1: Offline Policy Generation ---")

    # --- SETUP: Define Parameters and Mock Data ---
    ALL_PARAMETERS = [
        'imminent_collision_risk', 'pedestrian_detection', 'real-time_traffic_density',
        'safe_intervehicular_dist', 'road_surface_condition', 'sudden_weather_event',
        'general_weather_forecast', 'driver_drowsiness_level', 'road_closure_alert',
        'pothole_location', 'road_geometry_sharp_curve', 'speed_limit_info'
    ]
    
    # --- MOCK DATA GENERATION (Replace with your real data loading) ---
    print("\nStep 1: Generating mock data...")
    # Generate mock time-series data with different volatilities
    TIME_SERIES_DATA = {
        'imminent_collision_risk': np.random.normal(100, 20, 1000), # High vol
        'real-time_traffic_density': np.random.normal(50, 5, 1000), # Medium vol
        'pothole_location': np.sin(np.arange(1000) * 0.01) + 10 # Low vol
    }
    # Add mock data for all other parameters
    for p in ALL_PARAMETERS:
        if p not in TIME_SERIES_DATA:
            TIME_SERIES_DATA[p] = np.random.normal(10, 1, 1000)

    # Generate mock accident data
    data = {p: np.random.rand(500) for p in ALL_PARAMETERS}
    data['Accident_Severity'] = np.random.randint(0, 3, 500)
    # Make some features more important
    data['Accident_Severity'] = data['Accident_Severity'] + (2 * data['imminent_collision_risk']).astype(int)
    data['Accident_Severity'] = np.clip(data['Accident_Severity'], 0, 2)
    ACCIDENT_DATA = pd.DataFrame(data)
    print("Mock data generated.")
    
    # --- PART A: CALCULATE SCORES ---
    print("\nStep 2: Calculating Volatility and Criticality scores...")
    volatility_scores = {p: calculate_volatility_score(TIME_SERIES_DATA[p]) for p in ALL_PARAMETERS}
    criticality_scores = get_criticality_scores(ALL_PARAMETERS, ACCIDENT_DATA)
    
    # --- PART B: CATEGORIZE PARAMETERS ---
    print("\nStep 3: Categorizing parameters...")
    volatility_categories = categorize_scores(volatility_scores)
    criticality_categories = categorize_scores(criticality_scores)
    
    param_classifications = {p: (volatility_categories[p], criticality_categories[p]) for p in ALL_PARAMETERS}
    
    print("Parameter Classifications Complete:")
    for param, cats in param_classifications.items():
        print(f"  - {param}: (Volatility: {cats[0]}, Criticality: {cats[1]})")

    # --- PART C: GENERATE THE 9 MASTER POLICIES ---
    print("\nStep 4: Generating the 9 master policies...")
    
    # This is your 3x3 matrix of (alpha, beta) pairs
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
    
    # You would now save `param_classifications` and `master_policies` to files
    # to be used by your online simulation.
    # For example:
    # np.save('results/policies/master_policies.npy', master_policies)
    
    print("\nExample Policy for High Volatility, High Criticality (first 20 states):")
    print(master_policies[('High', 'High')][:20])