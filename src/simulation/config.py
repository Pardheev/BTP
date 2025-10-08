#  simulation/config.py

DECISION_RECIPES = {
    # --- A smaller, more focused set of DQs for comparability ---

    # Critical Safety (4 unique params)
    "imminent_collision_warning_forward": {
        "parameters": ["forward_vehicle_distance", "relative_speed", "own_vehicle_speed"],
        "sensor_accesses": 3, "procedure_type": "Rule-Based", "procedure_logic": "..."
    },
    "emergency_braking_assist": {
        "parameters": ["forward_object_distance", "relative_speed", "own_vehicle_speed"],
        "sensor_accesses": 3, "procedure_type": "Rule-Based", "procedure_logic": "..."
    },

    # Maneuvering (5 unique params)
    "safe_lane_change_advisory": {
        "parameters": ["rear_vehicle_distance", "side_vehicle_distance", "relative_speed_rear"],
        "sensor_accesses": 3, "procedure_type": "Mathematical", "procedure_logic": "..."
    },
    "curve_speed_warning": {
        "parameters": ["road_geometry_curvature", "own_vehicle_speed", "road_surface_condition"],
        "sensor_accesses": 3, "procedure_type": "Mathematical", "procedure_logic": "..."
    },
    
    # Situational Awareness (6 unique params)
    "traffic_level_assessment": {
        "parameters": ["average_speed_segment", "vehicle_count_segment"],
        "sensor_accesses": 2, "procedure_type": "Rule-Based", "procedure_logic": "..."
    },
     "road_condition_report": {
        "parameters": ["surface_moisture_level", "surface_temperature", "pothole_density"],
        "sensor_accesses": 3, "procedure_type": "Rule-Based", "procedure_logic": "..."
    },
    "accident_ahead_warning": {
        "parameters": ["v2x_accident_alert", "distance_to_accident"],
        "sensor_accesses": 2, "procedure_type": "Rule-Based", "procedure_logic": "..."
    },
}


# ==============================================================================
# DECISION RECIPE BOOK
# This dictionary maps a high-level decision to the parameters required
# and the procedure to generate the decision.
# ==============================================================================

# DECISION_RECIPES = {
#     # --- 1. Collision Avoidance & Critical Safety ---
#     "imminent_collision_warning_forward": {
#         "parameters": ["forward_vehicle_distance", "relative_speed", "own_vehicle_speed"],
#         "sensor_accesses": 3,
#         "procedure_type": "Rule-Based",
#         "procedure_logic": "Time_To_Collision = params['forward_vehicle_distance'] / params['relative_speed']; return 'CRITICAL_WARNING' if Time_To_Collision < 2.0 else 'SAFE'"
#     },
#     "emergency_braking_assist": {
#         "parameters": ["forward_object_distance", "relative_speed", "driver_brake_pedal_status"],
#         "sensor_accesses": 3,
#         "procedure_type": "Rule-Based",
#         "procedure_logic": "Time_To_Collision = params['forward_object_distance'] / params['relative_speed']; return 'APPLY_BRAKES' if Time_To_Collision < 1.5 and params['driver_brake_pedal_status'] == 'NOT_PRESSED' else 'MONITOR'"
#     },
#     "rear_end_collision_warning": {
#         "parameters": ["rear_vehicle_distance", "relative_speed_rear"],
#         "sensor_accesses": 2,
#         "procedure_type": "Rule-Based",
#         "procedure_logic": "Time_To_Impact = params['rear_vehicle_distance'] / params['relative_speed_rear']; return 'REAR_IMPACT_WARNING' if Time_To_Impact < 2.0 else 'SAFE'"
#     },
#     "cross_traffic_alert": {
#         "parameters": ["side_traffic_left_detected", "side_traffic_right_detected", "own_vehicle_speed"],
#         "sensor_accesses": 3,
#         "procedure_type": "Rule-Based",
#         "procedure_logic": "return 'CROSS_TRAFFIC_ALERT' if (params['side_traffic_left_detected'] or params['side_traffic_right_detected']) and params['own_vehicle_speed'] < 5 else 'CLEAR'"
#     },

#     # --- 2. Pedestrian & VRU Safety ---
#     "pedestrian_in_path_warning": {
#         "parameters": ["pedestrian_detected", "pedestrian_distance", "pedestrian_trajectory"],
#         "sensor_accesses": 3,
#         "procedure_type": "Rule-Based",
#         "procedure_logic": "return 'PEDESTRIAN_ALERT' if params['pedestrian_detected'] and params['pedestrian_distance'] < 20 else 'CLEAR'"
#     },
#     "cyclist_approaching_alert": {
#         "parameters": ["cyclist_detected_side", "cyclist_distance", "cyclist_relative_speed"],
#         "sensor_accesses": 3,
#         "procedure_type": "Rule-Based",
#         "procedure_logic": "return 'CYCLIST_ALERT' if params['cyclist_detected_side'] and params['cyclist_distance'] < 15 else 'CLEAR'"
#     },

#     # --- 3. Driver Assistance & Maneuvering ---
#     "safe_lane_change_advisory": {
#         "parameters": ["rear_vehicle_distance", "side_vehicle_distance", "relative_speed_rear"],
#         "sensor_accesses": 3,
#         "procedure_type": "Mathematical",
#         "procedure_logic": "Risk = (10 / params['rear_vehicle_distance']) + (20 / params['side_vehicle_distance']) + (0.5 * params['relative_speed_rear']); return 'UNSAFE' if Risk > 5.0 else 'SAFE'"
#     },
#     "curve_speed_warning": {
#         "parameters": ["road_geometry_curvature", "current_speed", "road_surface_condition"],
#         "sensor_accesses": 3,
#         "procedure_type": "Mathematical",
#         "procedure_logic": "friction = 0.7 if params['road_surface_condition'] == 'DRY' else 0.4; Max_Speed = sqrt(params['road_geometry_curvature'] * 9.81 * friction); return 'SLOW_DOWN' if params['current_speed'] > Max_Speed else 'SAFE_SPEED'"
#     },
#     "safe_following_distance_status": {
#         "parameters": ["forward_vehicle_distance", "current_speed"],
#         "sensor_accesses": 2,
#         "procedure_type": "Rule-Based",
#         "procedure_logic": "Required_Dist = params['current_speed'] * 2.0; return 'TOO_CLOSE' if params['forward_vehicle_distance'] < Required_Dist else 'SAFE_DISTANCE'"
#     },
#     "overtaking_assistant": {
#         "parameters": ["oncoming_traffic_distance", "oncoming_traffic_speed", "own_vehicle_speed"],
#         "sensor_accesses": 3,
#         "procedure_type": "Mathematical",
#         "procedure_logic": "Time_To_Oncoming = params['oncoming_traffic_distance'] / (params['own_vehicle_speed'] + params['oncoming_traffic_speed']); return 'OVERTAKE_UNSAFE' if Time_To_Oncoming < 10.0 else 'OVERTAKE_SAFE'"
#     },
#     "stop_sign_red_light_warning": {
#         "parameters": ["upcoming_signal_status", "distance_to_signal", "current_speed"],
#         "sensor_accesses": 3,
#         "procedure_type": "Rule-Based",
#         "procedure_logic": "return 'BRAKE_NOW' if params['upcoming_signal_status'] in ['RED', 'STOP'] and (params['distance_to_signal'] / params['current_speed']) < 4.0 else 'PROCEED_CAUTIOUSLY'"
#     },
#     "parking_assist_guidance": {
#         "parameters": ["rear_obstacle_distance", "side_obstacle_distance", "steering_angle"],
#         "sensor_accesses": 3,
#         "procedure_type": "Rule-Based",
#         "procedure_logic": "return 'STOP' if params['rear_obstacle_distance'] < 0.3 or params['side_obstacle_distance'] < 0.3 else 'CONTINUE_MANEUVER'"
#     },

#     # --- 4. Situational Awareness ---
#     "traffic_level_assessment": {
#         "parameters": ["average_speed_segment", "vehicle_count_segment"],
#         "sensor_accesses": 2,
#         "procedure_type": "Rule-Based",
#         "procedure_logic": "if params['average_speed_segment'] < 20: return 'HEAVY_TRAFFIC'; elif params['average_speed_segment'] < 50: return 'MODERATE_TRAFFIC'; else: return 'LIGHT_TRAFFIC'"
#     },
#     "road_condition_report": {
#         "parameters": ["surface_moisture_level", "surface_temperature", "pothole_density"],
#         "sensor_accesses": 3,
#         "procedure_type": "Rule-Based",
#         "procedure_logic": "if params['surface_moisture_level'] > 0.8 and params['surface_temperature'] < 2: return 'ICY_ROAD_WARNING'; elif params['pothole_density'] > 5: return 'POOR_ROAD_SURFACE'; else: return 'GOOD_ROAD_CONDITION'"
#     },
#     "weather_summary": {
#         "parameters": ["temperature", "humidity", "precipitation_intensity", "visibility_distance"],
#         "sensor_accesses": 4,
#         "procedure_type": "Formatted String",
#         "procedure_logic": "return f'Temp: {params[\"temperature\"]}C, Visibility: {params[\"visibility_distance\"]}m'"
#     },
#     "accident_ahead_warning": {
#         "parameters": ["v2x_accident_alert", "distance_to_accident"],
#         "sensor_accesses": 2,
#         "procedure_type": "Rule-Based",
#         "procedure_logic": "return 'ACCIDENT_AHEAD' if params['v2x_accident_alert'] else 'CLEAR'"
#     },
#     "emergency_vehicle_approaching": {
#         "parameters": ["v2x_emergency_vehicle_alert", "emergency_vehicle_direction"],
#         "sensor_accesses": 2,
#         "procedure_type": "Rule-Based",
#         "procedure_logic": "return f'EMERGENCY_VEHICLE_APPROACHING_FROM_{params[\"emergency_vehicle_direction\"]}' if params['v2x_emergency_vehicle_alert'] else 'CLEAR'"
#     },

#     # --- 5. Driver State Monitoring ---
#     "driver_drowsiness_alert": {
#         "parameters": ["eye_blink_rate", "head_pose_angle", "steering_wheel_movement_variance"],
#         "sensor_accesses": 3,
#         "procedure_type": "Classifier",
#         "procedure_logic": "return Drowsiness_Classifier.predict(params)"
#     },
#     "driver_distraction_warning": {
#         "parameters": ["gaze_direction", "time_gaze_off_road"],
#         "sensor_accesses": 2,
#         "procedure_type": "Rule-Based",
#         "procedure_logic": "return 'DRIVER_DISTRACTED' if params['time_gaze_off_road'] > 2.5 else 'ATTENTIVE'"
#     },

#     # --- 6. Vehicle Health ---
#     "tire_pressure_status": {
#         "parameters": ["tire_pressure_FL", "tire_pressure_FR", "tire_pressure_RL", "tire_pressure_RR"],
#         "sensor_accesses": 4,
#         "procedure_type": "Rule-Based",
#         "procedure_logic": "return 'LOW_TIRE_PRESSURE' if min(params.values()) < 30 else 'OK'"
#     },
#     "engine_health_status": {
#         "parameters": ["engine_temp", "oil_pressure", "error_codes"],
#         "sensor_accesses": 3,
#         "procedure_type": "Rule-Based",
#         "procedure_logic": "return 'CHECK_ENGINE' if params['engine_temp'] > 105 or params['oil_pressure'] < 10 or params['error_codes'] > 0 else 'OK'"
#     },
#     "hard_maneuver_event": {
#         "parameters": ["acceleration_g_force", "steering_angle_rate"],
#         "sensor_accesses": 2,
#         "procedure_type": "Rule-Based",
#         "procedure_logic": "return 'AGGRESSIVE_MANEUVER' if abs(params['acceleration_g_force']) > 0.5 or abs(params['steering_angle_rate']) > 80 else 'NORMAL'"
#     },

#     # --- 7. Eco-Driving & Efficiency ---
#     "optimal_speed_for_fuel_efficiency": {
#         "parameters": ["engine_rpm", "current_gear", "road_gradient"],
#         "sensor_accesses": 3,
#         "procedure_type": "Mathematical",
#         "procedure_logic": "Efficiency_Score = calculate_efficiency(params); return 'ADJUST_SPEED' if Efficiency_Score < 0.8 else 'EFFICIENT_DRIVING'"
#     },
#     "coasting_advisory": {
#         "parameters": ["distance_to_red_light", "time_to_green_light"],
#         "sensor_accesses": 2,
#         "procedure_type": "Rule-Based",
#         "procedure_logic": "return 'ADVISE_COASTING' if params['distance_to_red_light'] < 200 and params['time_to_green_light'] > 5 else 'MAINTAIN_SPEED'"
#     },
#     "idle_start_stop_status": {
#         "parameters": ["vehicle_speed", "time_stopped"],
#         "sensor_accesses": 2,
#         "procedure_type": "Rule-Based",
#         "procedure_logic": "return 'ENGINE_STOP' if params['vehicle_speed'] == 0 and params['time_stopped'] > 3 else 'ENGINE_RUN'"
#     },
# }