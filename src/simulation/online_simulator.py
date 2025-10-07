import random
import numpy as np
from .config import DECISION_RECIPES

class UserRequest:
    """Represents a single user's request for a safety decision."""
    def __init__(self, user_id, decision_query, from_loc, to_loc):
        self.user_id = user_id
        self.decision_query = decision_query
        self.from_loc = from_loc
        self.to_loc = to_loc
        self.required_params = DECISION_RECIPES[decision_query]['parameters']

class SensorDB:
    """Simulates a cloud database holding sensor data points and their age."""
    def __init__(self):
        # The DB stores data as: {(location, param_name): {"value": val, "aoi": age}}
        self.db = {}

    def get_data(self, location, param_name):
        """Retrieves data from the DB. Returns None if it doesn't exist."""
        return self.db.get((location, param_name))

    def update_data(self, location, param_name, new_value, new_aoi):
        """Updates or adds a new data point to the DB."""
        self.db[(location, param_name)] = {"value": new_value, "aoi": new_aoi}
        
    def increment_aoi_all(self):
        """Increments the Age of Information for every item in the database by 1."""
        for key in self.db:
            self.db[key]['aoi'] += 1

class OnlineSimulator:
    """Manages the real-time simulation of the caching agent."""
    def __init__(self, master_policies, param_classifications, resource_cost=50):
        self.master_policies = master_policies
        self.param_classifications = param_classifications
        self.resource_cost = resource_cost
        self.db = SensorDB()
        self.total_resource_cost = 0
        self.total_aoi_at_decision = 0
        self.decision_count = 0

    def _get_policy_for_param(self, param_name):
        """Retrieves the correct MDP policy for a given parameter."""
        classification = self.param_classifications[param_name]
        return self.master_policies[classification]

    def _make_caching_decision(self, location, param_name):
        """The core logic of the agent."""
        policy = self._get_policy_for_param(param_name)
        
        # Check if data exists in the DB
        data_point = self.db.get_data(location, param_name)
        
        if data_point is None:
            # Data doesn't exist, must fetch
            action = 1 
            current_aoi = 0 # No data means we consider it infinitely old
        else:
            current_aoi = data_point['aoi']
            # Get the action from the pre-computed policy
            # Policy is 0-indexed, AoI is 1-indexed
            action_index = min(current_aoi - 1, len(policy) - 1)
            action = policy[action_index]

        # Execute the chosen action
        if action == 1: # FETCH_SENSOR
            print(f"  - (Action: FETCH) Fetching fresh data for '{param_name}' at location '{location}'.")
            self.total_resource_cost += self.resource_cost
            new_value = np.random.normal(50, 10) # Simulate getting new data
            self.db.update_data(location, param_name, new_value, new_aoi=1)
            self.total_aoi_at_decision += 1
        else: # USE_CACHE
            print(f"  - (Action: CACHE) Using cached data for '{param_name}' (AoI: {current_aoi}) at location '{location}'.")
            self.total_aoi_at_decision += current_aoi
            
        self.decision_count += 1

    def run_time_step(self, user_requests):
        """Simulates a single time step t."""
        print(f"\n--- Simulating Time Step ---")
        
        # 1. Increment AoI for all existing data in the DB from the previous step
        self.db.increment_aoi_all()

        # 2. Aggregate all unique parameter requests for the current time step
        unique_param_requests = set()
        for req in user_requests:
            # For simplicity, we'll assume a single location for the "from-to" journey
            # In a real scenario, you'd sample points along the route
            location = req.from_loc 
            for param in req.required_params:
                unique_param_requests.add((location, param))
        
        print(f"Aggregated {len(unique_param_requests)} unique parameter requests for this time step.")

        # 3. For each unique request, let the agent make a decision
        for location, param_name in unique_param_requests:
            self._make_caching_decision(location, param_name)
            
    def get_metrics(self):
        """Returns the final simulation metrics."""
        avg_aoi = self.total_aoi_at_decision / self.decision_count if self.decision_count > 0 else 0
        return {
            "Total Resource Cost": self.total_resource_cost,
            "Average AoI at Decision Time": avg_aoi
        }

# --- Helper function to generate random user requests ---
def generate_user_requests(num_users, locations, decision_types):
    """Generates a list of N random user requests."""
    requests = []
    for i in range(num_users):
        user_id = f"user_{i+1}"
        decision = random.choice(decision_types)
        from_loc = random.choice(locations)
        to_loc = random.choice(locations)
        while from_loc == to_loc:
            to_loc = random.choice(locations)
        
        requests.append(UserRequest(user_id, decision, from_loc, to_loc))
    return requests