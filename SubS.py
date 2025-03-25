# Subset Simulation Implementation with Fixed Precision Levels

import numpy as np

# Global precision parameter for noise scaling
FIXED_PRECISION_LEVEL = 5  

def rRMSE(p_hat):
    """
    Calculate relative Root Mean Squared Error (rRMSE) against reference value.
    
    Args:
        p_hat (array-like): Array of estimated probability values
        
    Returns:
        float: Relative RMSE compared to reference value 7.23e-5
    """
    reference = 7.23e-05
    squared_errors = (np.array(p_hat) - reference) ** 2
    return np.sqrt(np.mean(squared_errors)) / reference

def sample_new_G(G_l, G, N, c_l, gamma=0.5):
    """
    Generate correlated samples using MCMC with fixed precision noise.
    
    Implements Modified Metropolis algorithm with AR(1) proposal distribution
    and fixed precision parameterization.
    
    Args:
        G_l (array): Current noisy evaluations (G_l <= c_l)
        G (array): True limit state samples
        N (int): Target sample count
        c_l (float): Current threshold value
        gamma (float): Base noise scaling factor
        
    Returns:
        tuple: Updated (G_l, G) arrays with new samples
    """
    N0 = len(G_l)
    rho = 0.8  # AR(1) correlation coefficient
    
    for _ in range(N - N0):
        # Generate correlated proposal
        G_new = rho * G[_] + np.sqrt(1 - rho**2) * np.random.normal(0, 1)
        
        # Add fixed-precision uniform noise
        kappa_new = np.random.uniform(-1, 1)
        G_l_new = G_new + kappa_new * gamma ** FIXED_PRECISION_LEVEL
        
        # Acceptance/rejection with sample recycling
        if G_l_new <= c_l:
            G_l = np.append(G_l, G_l_new)
            G = np.append(G, G_new)
        else:
            # Maintain chain length with previous sample
            G_l = np.append(G_l, G_l[_])
            G = np.append(G, G[_])
            
    return G_l, G

def subset_simulation(N, y_L=-3.8, p0=0.1, gamma=0.5, L=5):
    """
    Subset Simulation core algorithm for rare event probability estimation.
    
    Args:
        N (int): Samples per simulation level
        y_L (float): Final failure threshold
        p0 (float): Conditional probability target per level
        gamma (float): Noise scaling factor
        L (int): Maximum number of subsets
        
    Returns:
        tuple: (failure probability estimate, total computational cost)
    """
    N0 = int(N * p0)  # Surviving samples per level
    cost = 0

    # Level 1: Initial sampling
    G = np.random.normal(0, 1, N)
    kappa = np.random.uniform(-1, 1, N)
    G_l = G + kappa * gamma ** FIXED_PRECISION_LEVEL
    cost += N * gamma ** (-2 * FIXED_PRECISION_LEVEL)

    # Determine first threshold
    c_l = np.percentile(G_l, 100 * p0)
    mask = G_l <= c_l
    G_l, G = G_l[mask][:N0], G[mask][:N0]

    # Multi-level processing
    for l in range(2, L):
        # MCMC sampling phase
        G_l, G = sample_new_G(G_l, G, N, c_l, gamma)
        cost += (N - N0) * gamma ** (-2 * FIXED_PRECISION_LEVEL)
        
        # Update threshold
        c_l = np.percentile(G_l, 100 * p0)

        # Early termination check
        if c_l <= y_L:
            mask = G_l <= y_L
            return p0 ** (l-1) * np.mean(mask), cost

        # Shuffle to break correlation before resampling
        indices = np.random.permutation(len(G_l))
        G_l, G = G_l[indices], G[indices]
        
        # Resample for next level
        mask = G_l <= c_l
        G_l, G = G_l[mask][:N0], G[mask][:N0]

    # Final level processing
    G_l, G = sample_new_G(G_l, G, N, c_l, gamma)
    cost += (N - N0) * gamma ** (-2 * FIXED_PRECISION_LEVEL)
    mask = G_l <= y_L
    return p0 ** (L-1) * np.mean(mask), cost

def run_simulation(args):
    """
    Parallel execution wrapper for subset simulation.
    
    Args:
        args (tuple): (sample size N, random seed)
        
    Returns:
        tuple: (failure probability estimate, computational cost)
    """
    N, seed = args
    np.random.seed(seed)
    return subset_simulation(N)

if __name__ == "__main__":
    # Convergence Analysis and Result Collection
    import multiprocessing
    from tqdm import tqdm

    np.random.seed(135)  # Master seed for reproducibility
    error_list = []
    cost_list = []

    # Sample size progression for convergence study
    sample_sizes = [1000, 5000, 10000, 30000, 50000, 80000, 100000]
    
    for N in sample_sizes:
        failure_probs = []
        costs = []
        seeds = np.random.randint(100, 10000, 100)  # Worker seeds

        # Single-process execution for debugging (processes=1)
        with multiprocessing.Pool(1) as pool:
            args = [(N, seed) for seed in seeds]
            results = list(tqdm(pool.imap(run_simulation, args),
                          total=100, desc=f"Processing N={N}"))

        # Collect results
        failure_probs = [res[0] for res in results]
        costs = [res[1] for res in results]

        # Calculate performance metrics
        avg_prob = np.mean(failure_probs)
        rel_error = rRMSE(failure_probs)
        total_cost = np.mean(costs)
        
        print(f"N={N}: p={avg_prob:.2e}, error={rel_error:.2e}, cost={total_cost:.2e}\n")
        
        # Store results
        error_list.append(rel_error)
        cost_list.append(total_cost)
        np.save("SubS_error_list5.npy", error_list)
        np.save("SubS_cost_list5.npy", cost_list)