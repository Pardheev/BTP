import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def calculate_volatility_score(time_series_data):
    """Calculates a volatility score for a given time-series."""
    data = np.array(time_series_data)
    if len(data) < 2:
        return 0.0
    percent_change = np.abs((data[1:] - data[:-1]) / (data[:-1] + 1e-9))
    return np.mean(percent_change)

def get_criticality_scores(parameters, accident_data):
    """
    Trains a model to predict accident severity and returns feature importances.
    """
    print("Training ML model to determine criticality scores...")
    X = accident_data[parameters]
    y = accident_data['Accident_Severity']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    criticality_scores = {param: score for param, score in zip(parameters, importances)}
    print("ML model trained. Criticality scores extracted.")
    return criticality_scores

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