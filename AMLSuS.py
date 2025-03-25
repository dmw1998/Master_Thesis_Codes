# Import numerical computation and system libraries
import numpy as np
import time

def rRMSE(p_hat):
    """
    Calculate relative Root Mean Square Error (rRMSE) for probability estimates.
    
    Args:
        p_hat (array-like): Estimated probability values
        
    Returns:
        float: Relative RMSE compared to reference value 7.23e-5
    """
    p_hat = np.array(p_hat)
    reference = 7.23e-05
    return np.sqrt(np.mean((p_hat - reference)**2)) / reference

def delta(p_hat, I):
    """
    Calculate standard error of failure probability estimate with autocorrelation correction.
    
    Implements batch means method to account for sample correlation in MCMC chains.
    
    Args:
        p_hat (float): Estimated failure probability
        I (array-like): Binary indicator array of failure events
        
    Returns:
        float: Corrected standard error estimate
    """
    # Handle edge cases for probability estimates
    if p_hat == 0 or p_hat == 1:
        return 10.0 if p_hat == 0 else 0.0
    
    N = len(I)
    
    # Calculate sample variance with regularization
    var_I = np.var(I)
    var_I = max(var_I, 1e-12)  # Prevent division by zero
    
    # Compute autocorrelation function with limited lags
    max_lag = min(N//2, 100)  # Balance accuracy and computation time
    centered = I - np.mean(I)
    autocorr = np.correlate(centered, centered, mode='full')[N-1-max_lag:N+max_lag]
    autocorr = autocorr / (var_I * N)  # Normalize
    
    # Calculate effective sample size (ESS) with positive correlation truncation
    denominator = 1.0 + 2.0 * np.sum(np.clip(autocorr[1:max_lag+1], 0.0, None))
    denominator = max(denominator, 1e-6)  # Ensure numerical stability
    ess = N / denominator
    ess = max(ess, 1.0)  # Maintain minimum sample size
    
    # Compute final standard error
    variance = p_hat * (1 - p_hat)
    std_error = np.sqrt(variance / ess)
    
    return std_error

def compute_y(y_L, L, gamma):
    """
    Generate threshold sequence for multilevel subset simulation.
    
    Args:
        y_L (float): Final threshold value
        L (int): Number of levels
        gamma (float): Level scaling factor
        
    Returns:
        array: Array of threshold values for each level
    """
    y = np.zeros(L)
    y[-1] = y_L
    # Backward calculation of threshold sequence
    for l in range(L-2, -1, -1):
        y[l] = y[l+1] + gamma ** (l+1) + gamma ** (l+2)
    return y - 3.8  # Empirical adjustment

def adaptive_multilevel_subset_simulation(tolerance, L=5, gamma=0.5, y_L=-3.8):
    """
    Implement adaptive multilevel subset simulation for rare event estimation.
    
    Args:
        tolerance (float): Target error tolerance
        L (int): Number of simulation levels
        gamma (float): Level scaling factor
        y_L (float): Initial threshold value
        
    Returns:
        tuple: (estimated failure probability, array of computational costs per level)
    """
    y = compute_y(y_L, L, gamma)
    cost = np.zeros(L)
    G_l = np.array([])  # Noisy limit state evaluations
    G = np.array([])    # True limit state values
    N_l = 0             # Current sample count
    err = 10            # Initial error estimate
    p_hat = 0           # Current probability estimate
    tol = tolerance / np.sqrt(L)  # Level-specific tolerance
    
    # Initial sample size based on tolerance
    N = int(100 / tolerance)

    # Level 1 initialization
    G = np.random.normal(0, 1, N)
    G_l = G + np.random.uniform(-1, 1, N) * gamma
    mask = G_l <= y[0]
    p_hat = mask.mean()
    err = p_hat * (1 - p_hat) / N if p_hat != 0 else 10
    N_l = N
    cost[0] += N * gamma ** (-2)  # Cost accumulation

    # Level 1 adaptive sampling
    while err > tol:
        # Generate new sample with noise
        G_new = np.random.normal(0, 1)
        kappa = np.random.uniform(-1, 1)
        
        # Update sample arrays
        G_l = np.append(G_l, G_new + kappa * gamma)
        G = np.append(G, G_new)
        
        # Update probability estimate
        mask = G_l <= y[0]
        p_hat = mask.mean()
        N_l += 1
        cost[0] += gamma ** (-2)
        err = p_hat * (1 - p_hat) / N_l if p_hat != 0 else 10

    p_f = p_hat  # Initialize cumulative probability
    
    # Subsequent levels processing
    for l in range(1, L):
        tol /= np.sqrt(L)  # Tighten tolerance
        # Resample from conditional distribution
        G_l = G_l[mask][:1]
        G = G[mask][:1]
        N = int(N*0.8)  # Adaptive sample reduction

        # MCMC sampling phase
        for _ in range(N):
            # Generate correlated sample
            G_new = 0.8 * G[-1] + np.sqrt(1 - 0.8**2) * np.random.normal(0, 1)
            kappa = np.random.uniform(-1, 1)
            G_l_new = G_new + kappa * gamma ** (l + 1)
            
            # Acceptance/rejection
            if G_l_new <= y[l-1]:
                G_l = np.append(G_l, G_l_new)
                G = np.append(G, G_new)
            else:
                G_l = np.append(G_l, G_l[-1])  # Reuse previous
                G = np.append(G, G[-1])
            
            # Update estimates
            mask = G_l <= y[l]
            p_hat = mask.mean()
            err = delta(p_hat, mask)
            cost[l] += gamma ** (-2 * (l+1))

        # Adaptive refinement phase
        while err > tol:
            G_new = 0.8 * G[-1] + np.sqrt(1 - 0.8**2) * np.random.normal(0, 1)
            kappa_new = np.random.uniform(-1, 1)
            G_l_new = G_new + kappa_new * gamma ** (l + 1)
            
            if G_l_new <= y[l-1]:
                G_l = np.append(G_l, G_l_new)
                G = np.append(G, G_new)
            else:
                G_l = np.append(G_l, G_l[-1])
                G = np.append(G, G[-1])
            
            mask = G_l <= y[l]
            p_hat = mask.mean()
            err = delta(p_hat, mask)
            cost[l] += gamma ** (-2 * (l+1))
        
        p_f *= p_hat  # Update cumulative probability

    return p_f, cost

def run_simulation(args):
    """
    Wrapper function for parallel execution.
    
    Args:
        args (tuple): (tolerance value, random seed)
        
    Returns:
        tuple: (failure probability estimate, total computational cost)
    """
    tol, seed = args
    np.random.seed(seed)
    p_f, cost = adaptive_multilevel_subset_simulation(tol)
    return p_f, np.sum(cost)

if __name__ == "__main__":
    # Parallel execution setup
    import multiprocessing
    from tqdm import tqdm

    np.random.seed(99)  # Master seed for reproducibility
    error_list = []
    cost_list = []

    # Tolerance levels to analyze
    tolerance_levels = [0.3, 0.2, 0.1, 0.08, 0.05, 0.03]
    
    for tol in tolerance_levels:
        failure_probs = []
        total_costs = []
        seeds = np.random.randint(10, 10000, 100)  # Worker seeds

        # Parallel execution with progress tracking
        with multiprocessing.Pool(8) as pool:
            results = pool.imap(run_simulation, [(tol, seed) for seed in seeds])
            for p_f, cost in tqdm(results, total=100, desc=f"Processing tol={tol:.2f}"):
                failure_probs.append(p_f)
                total_costs.append(cost)
        
        # Calculate and store performance metrics
        err = rRMSE(failure_probs)
        error_list.append(err)
        cost_list.append(np.mean(total_costs))
        
        print(f"Tolerance: {tol:.2f} | Error: {err:.2e} | Cost: {cost_list[-1]:.2e}\n")
        
        # Save intermediate results
        np.save("AMLSS_error_list1", error_list)
        np.save("AMLSS_cost_list1", cost_list)

    # Result visualization
    import matplotlib.pyplot as plt
    from scipy.stats import linregress

    plt.figure(figsize=(12, 8))
    
    # Load results
    error_list_amlss = np.load("AMLSS_error_list1.npy")
    cost_list_amlss = np.load("AMLSS_cost_list1.npy")

    # Empirical data plot
    plt.loglog(error_list_amlss, cost_list_amlss, "tab:orange", 
              label="AMLSS", marker='o')

    # Theoretical scaling line
    x = np.linspace(1.1*error_list_amlss[0], 0.9*error_list_amlss[-1], 100)
    y = 0.1 * cost_list_amlss[0] * x ** (-2)
    plt.loglog(x, y, color="tab:orange", linestyle='-.', label=r"Theoretical $\epsilon^{-2}$")

    # Empirical scaling fit
    slope, intercept, *_ = linregress(np.log(error_list_amlss), np.log(cost_list_amlss))
    plt.loglog(error_list_amlss, np.exp(intercept) * error_list_amlss ** slope,
              "tab:orange", linestyle='--', 
              label=f"Empirical Fit: $\\epsilon^{{{slope:.2f}}}$")

    # Plot configuration
    plt.xlabel("Relative Error (rRMSE)")
    plt.ylabel("Computational Cost")
    plt.title("AMLSS Performance Profile")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()