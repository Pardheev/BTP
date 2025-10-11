import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def calculate_volatility_score(time_series_data):
    """
    Calculates a volatility score for a given time-series (equivalent to the step: score_v).
    Uses Mean Absolute Percentage Change.
    """
    data = np.array(time_series_data)
    if len(data) < 2:
        return 0.0
    # Calculate Mean Absolute Percentage Change
    percent_change = np.abs((data[1:] - data[:-1]) / (data[:-1] + 1e-9))
    return np.mean(percent_change)

def get_criticality_scores(parameters, accident_data):
    """
    Trains a model to predict accident severity and returns feature importances (equivalent
    to the step: AnalyzeAccidentData(D_Accident, P)).
    """
    print("Training ML model to determine criticality scores...")
    # Use all unique parameters (P) for training features (X)
    X = accident_data[parameters]
    y = accident_data['Accident_Severity']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    criticality_scores = {param: score for param, score in zip(parameters, importances)}
    print("ML model trained. Criticality scores extracted.")
    return criticality_scores

def categorize_scores(scores_dict):
    """
    Ranks scores and bins them into High, Medium, and Low categories 
    (equivalent to the steps: C_volatility and C_criticality).
    """
    # Rank parameters based on their score (e.g., Volatility or Criticality)
    sorted_params = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
    n = len(sorted_params)
    categories = {}
    
    # Assign categories based on equal thirds of the ranked list
    for i, (param, score) in enumerate(sorted_params):
        if i < n / 3:
            categories[param] = 'High'
        elif i < 2 * n / 3:
            categories[param] = 'Medium'
        else:
            categories[param] = 'Low'
    return categories

# --- New Master Function aligned with Algorithm 1 ---
def perform_offline_parameter_characterization(all_parameters, time_series_data, accident_data):
    """
    Orchestrates the entire offline parameter characterization process (Algorithm 1).
    
    Args:
        all_parameters (list): The set of all unique decision parameters (P).
        time_series_data (dict): Historical time-series data (D_TS).
        accident_data (DataFrame): Historical accident data (D_Accident).
        
    Returns:
        tuple: (C_map, S_vol, S_crit) where C_map is the final classification.
    """
    # 1. Initialization is implicit, P is passed as all_parameters
    
    # 2. Calculate Volatility Scores (S_vol) -> S_vol
    S_vol = {
        p: calculate_volatility_score(time_series_data[p]) 
        for p in all_parameters
    }
    
    # 3. Calculate Criticality Scores (S_crit) -> S_crit
    S_crit = get_criticality_scores(all_parameters, accident_data)
    
    # 4. Categorize Volatility Scores -> C_volatility
    C_volatility = categorize_scores(S_vol)
    
    # 5. Categorize Criticality Scores -> C_criticality
    C_criticality = categorize_scores(S_crit)
    
    # 6. Combine to form the final Classification Map (C_map)
    C_map = {
        p: (C_volatility[p], C_criticality[p])
        for p in all_parameters
    }
    
    # 7. Return all required outputs
    return C_map, S_vol, S_crit