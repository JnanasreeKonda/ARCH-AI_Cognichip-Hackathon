import random


def propose_design(history):
    """
    Intelligent design space exploration agent.
    
    Strategy:
    1. Start with baseline designs
    2. Explore around best-performing designs
    3. Balance exploration (trying new params) vs exploitation (refining good designs)
    4. Search 2D space: PAR ∈ {1,2,4,8,16,32}, BUFFER_DEPTH ∈ {256,512,1024,2048}
    """
    
    # Define search space
    PAR_OPTIONS = [1, 2, 4, 8, 16, 32]
    BUFFER_DEPTH_OPTIONS = [256, 512, 1024, 2048]
    
    # Filter valid history entries
    valid = [h for h in history if h[1].get("total_cells") is not None]
    
    # Phase 1: Initial exploration (first 4 iterations)
    if len(valid) < 4:
        # Sample diverse starting points
        if len(valid) == 0:
            return {"PAR": 2, "BUFFER_DEPTH": 1024}  # Baseline
        elif len(valid) == 1:
            return {"PAR": 4, "BUFFER_DEPTH": 1024}  # Moderate parallelism
        elif len(valid) == 2:
            return {"PAR": 1, "BUFFER_DEPTH": 512}   # Minimal area
        else:
            return {"PAR": 8, "BUFFER_DEPTH": 2048}  # High performance
    
    # Phase 2: Exploitation - explore around best design
    # Find best design by objective function
    best = min(valid, key=lambda x: x[1].get("objective", float('inf')))
    best_params = best[0]
    best_par = best_params["PAR"]
    best_buffer = best_params.get("BUFFER_DEPTH", 1024)
    
    # Get index in options
    try:
        par_idx = PAR_OPTIONS.index(best_par)
        buffer_idx = BUFFER_DEPTH_OPTIONS.index(best_buffer)
    except ValueError:
        # If not in list, use closest
        par_idx = 2  # Default to PAR=4
        buffer_idx = 2  # Default to 1024
    
    # 70% exploitation: Try neighbors of best design
    # 30% exploration: Random search
    if random.random() < 0.7:
        # Exploitation: explore neighbors in parameter space
        # Try adjacent PAR values
        neighbor_pars = []
        if par_idx > 0:
            neighbor_pars.append(PAR_OPTIONS[par_idx - 1])
        if par_idx < len(PAR_OPTIONS) - 1:
            neighbor_pars.append(PAR_OPTIONS[par_idx + 1])
        neighbor_pars.append(best_par)  # Also consider same PAR
        
        # Try adjacent BUFFER_DEPTH values
        neighbor_buffers = []
        if buffer_idx > 0:
            neighbor_buffers.append(BUFFER_DEPTH_OPTIONS[buffer_idx - 1])
        if buffer_idx < len(BUFFER_DEPTH_OPTIONS) - 1:
            neighbor_buffers.append(BUFFER_DEPTH_OPTIONS[buffer_idx + 1])
        neighbor_buffers.append(best_buffer)  # Also consider same buffer
        
        # Generate all neighbor combinations
        neighbors = [(p, b) for p in neighbor_pars for b in neighbor_buffers]
        
        # Filter out already tried combinations
        tried = {(h[0]["PAR"], h[0].get("BUFFER_DEPTH", 1024)) for h in valid}
        untried_neighbors = [(p, b) for p, b in neighbors if (p, b) not in tried]
        
        if untried_neighbors:
            # Choose random untried neighbor
            par, buffer = random.choice(untried_neighbors)
            return {"PAR": par, "BUFFER_DEPTH": buffer}
    
    # Exploration: Random search in untried space
    tried = {(h[0]["PAR"], h[0].get("BUFFER_DEPTH", 1024)) for h in valid}
    untried = [(p, b) for p in PAR_OPTIONS for b in BUFFER_DEPTH_OPTIONS if (p, b) not in tried]
    
    if untried:
        par, buffer = random.choice(untried)
        return {"PAR": par, "BUFFER_DEPTH": buffer}
    
    # All combinations tried - do local refinement around best
    # Try slight variations
    return {"PAR": best_par, "BUFFER_DEPTH": best_buffer}
