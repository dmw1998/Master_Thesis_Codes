# Subset Simulation with Selective Refinement and Fixed Precision Implementation

import numpy as np
from tqdm import tqdm
import multiprocessing

# Global fixed precision parameter for noise scaling
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

def sample_new_G(G_l, G, N, l, c_l, gamma=0.5):
    """
    Generate correlated samples using MCMC with fixed precision noise.
    
    Implements Modified Metropolis algorithm with AR(1) proposal distribution
    for subset simulation with selective refinement.
    
    Args:
        G_l (array): Current noisy evaluations (G_l <= c_l)
        G (array): True limit state samples
        N (int): Target sample count
        l (int): Current simulation level
        c_l (float): Current threshold value
        gamma (float): Base noise scaling factor
        
    Returns:
        tuple: Updated (G_l, G) arrays with new samples
    """
    N0 = len(G_l)
    rho = 0.8  # AR(1) correlation coefficient
    
    # MCMC sampling with correlation preservation
    for i in range(N - N0):
        # Generate correlated proposal using autoregressive process
        G_new = rho * G[i] + np.sqrt(1 - rho**2) * np.random.normal(0, 1)
        
        # Add fixed-precision uniform noise
        kappa_new = np.random.uniform(-1, 1)
        G_l_new = G_new + kappa_new * gamma ** FIXED_PRECISION_LEVEL

        # Acceptance/rejection with sample recycling
        if G_l_new <= c_l:
            G_l = np.append(G_l, G_l_new)
            G = np.append(G, G_new)
        else:
            # Maintain chain length with previous sample
            G_l = np.append(G_l, G_l[i])
            G = np.append(G, G[i])
    
    return G_l, G

def subset_simulation_y_l(N, y_L=-3.8, gamma=0.5, L=5):
    """
    Subset Simulation with predefined thresholds for selective refinement.
    
    Args:
        N (int): Samples per simulation level
        y_L (float): Final failure threshold
        gamma (float): Noise scaling factor
        L (int): Number of simulation levels
        
    Returns:
        tuple: (failure probability estimate, array of computational costs per level)
    """
    # Predefined failure thresholds for each level
    y = [-1.3, -2.0, -2.8, -3.3, y_L]
    cost = np.zeros(L)
    
    # Level 0 initialization
    G = np.random.normal(0, 1, N)
    kappa = np.random.uniform(-1, 1, N)
    G_l = G + kappa * gamma ** FIXED_PRECISION_LEVEL
    cost[0] = N * gamma ** (-2 * FIXED_PRECISION_LEVEL)
    
    # Initial probability estimation
    mask = G_l <= y[0]
    p_f = mask.mean()

    # Multi-level processing
    for l in range(1, L):
        # Resample from conditional distribution
        G_l, G = G_l[mask][:1], G[mask][:1]

        # Generate new samples with MCMC
        G_l, G = sample_new_G(G_l, G, N, l, y[l-1], gamma=gamma)
        cost[l] = (N - 1) * gamma ** (-2 * FIXED_PRECISION_LEVEL)
        
        # Update probability estimate
        mask = G_l <= y[l]
        p_f *= mask.mean()
        
    return p_f, cost

def run_simulation(args):
    """
    Parallel execution wrapper for subset simulation.
    
    Args:
        args (tuple): (sample size N, random seed)
        
    Returns:
        tuple: (failure probability estimate, total computational cost)
    """
    N, seed = args
    np.random.seed(seed)
    return subset_simulation_y_l(N)

if __name__ == "__main__":
    # Convergence Analysis and Performance Benchmarking
    error_list = []
    cost_list = []
    np.random.seed(136)  # Master seed for reproducibility

    # Sample size progression for convergence study
    sample_sizes = [1000, 5000, 10000, 30000, 50000, 80000, 100000]
    
    for N in sample_sizes:
        failure_probs = []
        costs = []
        seeds = np.random.randint(10, 1000, 100)  # Worker seeds

        # Parallel execution with progress tracking
        with multiprocessing.Pool(processes=8) as pool:
            args = [(N, seed) for seed in seeds]
            results = list(tqdm(pool.imap(run_simulation, args),
                          total=100, desc=f"Processing N={N}"))
        
        # Process results
        failure_probs = [res[0] for res in results]
        costs = [res[1] for res in results]

        # Calculate performance metrics
        rel_error = rRMSE(failure_probs)
        avg_cost = np.mean(costs)
        
        print(f"Relative RMSE: {rel_error:.2e}")
        print(f"Average Computational Cost: {avg_cost:.2e}\n")
        
        # Store results for analysis
        error_list.append(rel_error)
        cost_list.append(avg_cost)
        np.save("SubS_sr_error_list5.npy", error_list)
        np.save("SubS_sr_cost_list5.npy", cost_list)