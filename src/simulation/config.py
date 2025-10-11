# File: sem7/src/simulation/config.py

# This dictionary defines the parameters required for each type of decision query.
DECISION_RECIPES = {
    "find_shortest_path": {
        "description": "Finds the geographically shortest route.",
        "parameters": ["road_segment_length", "road_closure"]
    },
    "find_fastest_path": {
        "description": "Finds the route with the minimum travel time.",
        "parameters": ["traffic_congestion", "average_speed", "road_closure"]
    },
    "find_parking": {
        "description": "Finds the nearest available parking slot.",
        "parameters": ["parking_slot_availability"]
    },
    "find_less_crowded_shop": {
        "description": "Finds a nearby shop with the lowest footfall.",
        "parameters": ["shop_footfall_count"]
    },
    "find_cooler_path": {
        "description": "Finds a route that minimizes heat exposure.",
        "parameters": ["ambient_temperature", "road_surface_temperature", "road_shade_coverage"]
    },
    "find_available_ebike": {
        "description": "Finds a docking station with an available e-bike.",
        "parameters": ["ebike_availability_count"]
    },
    "check_road_water_level": {
        "description": "Checks for waterlogging on the route.",
        "parameters": ["road_water_level"]
    },
    "check_pothole_severity": {
        "description": "Assesses the road surface quality for major potholes.",
        "parameters": ["pothole_severity_score"]
    }
}

# Other simulation constants can also be placed here
SIM_DURATION = 100
LOCATIONS = ["Lanka", "Godowlia", "Assi Ghat", "BHU Campus", "Cantt Station"]
ENERGY_PER_FETCH = 50 # in nJ