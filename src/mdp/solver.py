import numpy as np

def solve_mdp(alpha, beta, resource_cost=50, max_aoi=100, gamma=0.95, epsilon=1e-4):
    """Solves for the optimal caching policy using Value Iteration."""
    states = np.arange(1, max_aoi + 1)
    num_states = len(states)
    V = np.zeros(num_states)
    
    while True:
        delta = 0
        for s_idx, s_aoi in enumerate(states):
            # Value of action 0 (USE_CACHE)
            cost_cache = beta * (s_aoi ** alpha)
            next_s_cache_idx = min(s_idx + 1, num_states - 1)
            v_cache = cost_cache + gamma * V[next_s_cache_idx]

            # Value of action 1 (FETCH_SENSOR)
            cost_fetch = resource_cost
            next_s_fetch_idx = 0  # AoI resets to 1
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