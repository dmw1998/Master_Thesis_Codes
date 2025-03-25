# Multilevel Estimator Implementation for Toy Reliability Experiment

import numpy as np

def rRMSE(p_hat):
    """
    Calculate relative Root Mean Squared Error (rRMSE) against reference value.
    
    Args:
        p_hat (array-like): Array of estimated probability values
        
    Returns:
        float: Relative RMSE compared to reference value 7.23e-5
    """
    p_hat = np.array(p_hat)
    reference = 7.23e-05
    squared_errors = (p_hat - reference) ** 2
    return np.sqrt(np.mean(squared_errors)) / reference

def sample_new_G(G, G_l, N, l, c_l, gamma=0.5):
    """
    Generate new samples using MCMC with adaptive correlation structure.
    
    Implements Markov Chain Monte Carlo sampling with controlled autocorrelation
    for multilevel subset simulation.
    
    Args:
        G (array): Current true limit state samples
        G_l (array): Noisy evaluations from previous level
        N (int): Target sample count
        l (int): Current simulation level
        c_l (float): Current threshold value
        gamma (float): Autocorrelation decay factor
        
    Returns:
        tuple: Updated (G_l, G) arrays with new samples
    """
    N0 = len(G_l)
    
    # Generate new samples using correlated MCMC chains
    for i in range(N - N0):
        # Create correlated proposal using AR(1) process
        G_new = 0.4 * G[i] + np.sqrt(1 - 0.4**2) * np.random.normal(0, 1)
        
        # Add uniform noise scaled by level
        kappa_new = np.random.uniform(-1, 1)
        G_l_new = G_new + kappa_new * gamma ** l
        
        # Acceptance/rejection based on threshold
        if G_l_new <= c_l:
            G_l = np.append(G_l, G_l_new)
            G = np.append(G, G_new)
        else:
            # Reuse previous sample if rejection occurs
            G_l = np.append(G_l, G_l[i])
            G = np.append(G, G[i])
            
    return G_l, G

def mle(N, L_b=10, y_L=-3.8, p0=0.25, gamma=0.5, L=7):
    """
    Multilevel Estimation core algorithm for rare event probability calculation.
    
    Args:
        N (int): Base samples per level
        L_b (int): Burn-in samples per chain
        y_L (float): Final failure threshold
        p0 (float): Conditional probability per level
        gamma (float): Autocorrelation decay factor
        L (int): Maximum number of levels
        
    Returns:
        tuple: (failure probability estimate, total computational cost)
    """
    N0 = int(N * p0)  # Samples surviving each level
    
    # Level 1 Initialization
    G = np.random.normal(0, 1, N)
    kappa = np.random.uniform(-1, 1, N)
    G_l = G + kappa * gamma
    cost = N * gamma ** (-2)  # Initial computational cost
    
    # Determine first threshold
    c_l = np.percentile(G_l, 100 * p0)
    
    # Early exit if threshold below final target
    if c_l <= y_L:
        mask = G_l <= y_L
        return np.mean(mask) / N, cost
    
    # Filter samples for next level
    mask = G_l <= c_l
    G_l = G_l[mask][:N0]
    G = G[mask][:N0]
    
    dinominator = 1  # NOTE: Typo preserved from original (should be 'denominator')
    
    # Level 2 Processing (no burn-in)
    G_l, G = sample_new_G(G, G_l, N, 2, c_l, gamma=gamma)
    cost += (N - N0) * gamma ** (-4)
    c_l_1 = c_l  # Previous level threshold
    
    # Update threshold
    c_l = np.percentile(G_l, 100 * p0)
    
    if c_l <= y_L:
        mask = G_l <= y_L
        return p0 * np.mean(mask) / dinominator, cost
    
    # Filter and process subsequent levels
    mask = G_l <= c_l
    G_l = G_l[mask][:N0]
    G = G[mask][:N0]
    
    # Burn-in handling for levels > 2
    N += L_b * N0
    for l in range(3, L):
        # Generate new samples with MCMC
        G_l, G = sample_new_G(G, G_l, N, l, c_l, gamma=gamma)
        cost += (N - N0) * gamma ** (-2 * l)
        
        # Discard burn-in samples
        G_l = G_l[int(N0*L_b):]
        G = G[int(N0*L_b):]
        c_l_1 = c_l
        
        # Update threshold
        c_l = np.percentile(G_l, 100 * p0)
        
        if c_l <= y_L:
            mask = G_l <= y_L
            return p0 ** (l-1) * np.sum(mask) / N / dinominator, cost

        # Filter and prepare for next level
        mask = G_l <= c_l
        G_l = G_l[mask][:N0]
        G = G[mask][:N0]
        
        # Update denominator with conditional probability
        G_l_1, _ = sample_new_G(G, G_l, N, l, c_l, gamma=gamma)
        cost += (N - N0) * gamma ** (-2 * l)
        G_l_1 = G_l_1[int(N0*L_b):]
        mask = G_l_1 <= c_l_1
        dinominator *= np.mean(mask)
        
    # Final level processing
    G_l, _ = sample_new_G(G, G_l, N, L, c_l, gamma=gamma)
    cost += (N - N0) * gamma ** (-2 * L)
    G_l = G_l[int(N0*L_b):]
    mask = G_l <= y_L

    # Final probability calculation
    G_l_1, _ = sample_new_G(G, G_l, N, L, y_L, gamma=gamma)
    cost += (N - N0) * gamma ** (-2 * L)
    G_l_1 = G_l_1[int(N0*L_b):]
    mask_1 = G_l_1 <= c_l
    dinominator *= np.mean(mask_1)
    
    return p0 ** (L-1) * np.mean(mask) / dinominator, cost

def run_simulation(args):
    """
    Parallel execution wrapper for MLE algorithm.
    
    Args:
        args (tuple): (sample size N, random seed)
        
    Returns:
        tuple: (failure probability estimate, computational cost)
    """
    N, seed = args
    np.random.seed(seed)
    return mle(N)

if __name__ == "__main__":
    # Parallel execution and result collection
    import multiprocessing as mp
    from tqdm import tqdm
    
    np.random.seed(16)  # Master seed for reproducibility
    error_list = []
    cost_list = []

    # Sample size progression for convergence study
    sample_sizes = [2000, 5000, 7000, 10000, 20000, 50000, 70000, 100000, 150000]
    
    for N in sample_sizes:
        failure_probs = []
        costs = []
        seeds = np.random.randint(100, 10000, 100)  # Worker seeds
        
        # Parallel execution with progress bar
        with mp.Pool(processes=8) as pool:
            args = [(N, seed) for seed in seeds]
            results = pool.imap(run_simulation, args)
            
            # Collect results with progress tracking
            for p_f, cost in tqdm(results, total=100, desc=f"Processing N={N}"):
                failure_probs.append(p_f)
                costs.append(cost)
        
        # Calculate performance metrics
        avg_prob = np.mean(failure_probs)
        avg_cost = np.mean(costs)
        rel_error = rRMSE(failure_probs)
        
        # Store and display results
        cost_list.append(avg_cost)
        error_list.append(rel_error)
        print(f"N={N}: p={avg_prob:.2e}, cost={avg_cost:.2e}, error={rel_error:.2e}\n")
        
        # Save incremental results
        np.save("MLE_cost_list.npy", cost_list)
        np.save("MLE_error_list.npy", error_list)