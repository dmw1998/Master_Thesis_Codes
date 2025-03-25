# Selective Refinement Multilevel Estimator Implementation

import numpy as np

def generate_samples(w, G, N, gamma, c, l, burn_in=0):
    """
    Generate correlated samples with adaptive precision refinement.
    
    Implements Markov Chain Monte Carlo with selective refinement for
    efficient failure probability estimation.
    
    Args:
        w (array): Current latent variable samples
        G (array): Current limit state evaluations
        N (int): Target sample count after burn-in
        gamma (float): Precision refinement factor
        c (float): Current failure threshold
        l (int): Current simulation level
        burn_in (int): Burn-in periods per chain
        
    Returns:
        tuple: (filtered samples, updated evaluations, computational cost)
    """
    N0 = len(w)
    L_b = burn_in * N0  # Total burn-in samples
    cost = 0
    
    # MCMC sampling with adaptive refinement
    for i in range(N + (burn_in - 1) * N0):
        # Correlated proposal generation
        w_new = 0.6 * w[i] + np.sqrt(1 - 0.6**2) * np.random.normal(0, 1)
        
        # Initial noise injection
        kappa = np.random.uniform(-1, 1)
        add_term = kappa * gamma
        tol = gamma
        G_new = w_new + add_term
        cost += gamma ** (-2)  # Base evaluation cost

        # Selective refinement loop
        for j in range(2, l):
            if tol >= np.abs(G_new - c):
                # Increase precision level
                tol *= gamma
                add_term *= gamma
                G_new = w_new + add_term
                cost += gamma ** (-2 * j)  # Cumulative refinement cost
            else:
                break  # Stop refinement when sufficient
        
        # Acceptance/rejection with recycling
        if G_new <= c:
            w = np.append(w, w_new)
            G = np.append(G, G_new)
        else:
            # Reuse previous sample
            w = np.append(w, w[i])
            G = np.append(G, G[i])

    # Discard burn-in samples        
    return w[L_b:], G[L_b:], cost

def mle_sr(N, gamma=0.5, y=-3.8, p_0=0.25, L=7, burn_in=3):
    """
    Multilevel Estimation with Selective Refinement (MLE-SR) core algorithm.
    
    Args:
        N (int): Base samples per level
        gamma (float): Precision decay factor
        y (float): Final failure threshold
        p_0 (float): Conditional probability target per level
        L (int): Maximum simulation levels
        burn_in (int): Burn-in periods for chain stabilization
        
    Returns:
        tuple: (failure probability estimate, total computational cost)
    """
    N0 = int(p_0 * N)  # Surviving samples per level
    cost = 0

    # Level 1 initialization
    l = 1
    w = np.random.normal(0, 1, N)
    kappa = np.random.uniform(-1, 1, N)
    G = w + kappa * gamma
    cost += N * gamma ** (-2)
    
    # First threshold determination
    c_1 = np.sort(G)[N0-1]  # (1-p_0)-quantile

    # Level 2 processing (no burn-in)
    mask = G <= c_1
    w, G = w[mask][:N0], G[mask][:N0]
    
    w, G, add_cost = generate_samples(w, G, N, gamma, c_1, 2)
    cost += add_cost
    
    # Second threshold and denominator calculation
    c_2 = np.sort(G)[N0-1]
    _, G_2, add_cost = generate_samples(w, G, N, gamma, c_2, 2)
    cost += add_cost
    denominator = np.mean(G_2 <= c_1)  # Conditional probability

    # Multi-level processing with burn-in
    c_l = c_2
    for l in range(3, L):
        c_l_1 = c_l  # Previous threshold
        
        # Generate samples with burn-in
        w, G, add_cost = generate_samples(w, G, N, gamma, c_l, l, burn_in)
        cost += add_cost
        c_l = np.sort(G)[N0-1]  # New threshold
        
        # Early termination check
        if c_l <= y:
            mask = G <= y
            return p_0 ** (l-1) * np.mean(mask) / denominator, cost

        # Prepare for next level
        mask = G <= c_l
        w, G = w[mask][:N0], G[mask][:N0]
        
        # Update denominator with chain correlation
        _, G_l, add_cost = generate_samples(w, G, N, gamma, c_l, l, burn_in)
        cost += add_cost
        denominator *= np.mean(G_l <= c_l_1)

    # Final level processing
    c_L_1 = c_l
    w, G, add_cost = generate_samples(w, G, N, gamma, c_l, L, burn_in)
    cost += add_cost
    
    # Final probability calculation
    mask = G <= y
    p_L = np.mean(mask)
    
    # Compute final correction factor
    _, G_L, add_cost = generate_samples(w, G, N, gamma, y, L, burn_in)
    cost += add_cost
    denominator *= np.mean(G_L < c_L_1)
    
    return p_0 ** (L-1) * p_L / denominator, cost

def run_simulation(args):
    """
    Parallel execution wrapper for MLE-SR algorithm.
    
    Args:
        args (tuple): (sample size N, random seed)
        
    Returns:
        tuple: (failure probability estimate, total cost)
    """
    N, seed = args
    np.random.seed(seed)
    return mle_sr(N)

if __name__ == "__main__":
    # Convergence study execution and analysis
    import multiprocessing
    from tqdm import tqdm

    np.random.seed(180)  # Reproducibility seed
    cost_list = []
    err_list = []

    # Sample size progression for convergence analysis
    sample_sizes = [1000, 3000, 7000, 10000, 20000, 50000, 70000, 120000]
    
    for N in sample_sizes:
        failure_probs = []
        costs = []
        seeds = np.random.randint(10, 10000, 100)  # Worker seeds
        
        # Parallel execution with progress tracking
        with multiprocessing.Pool(8) as pool:
            args = [(N, seed) for seed in seeds]
            results = list(tqdm(pool.imap(run_simulation, args),
                              total=100, desc=f"Processing N={N}"))
            
            # Unpack results
            failure_probs = [res[0] for res in results]
            costs = [res[1] for res in results]
        
        # Calculate performance metrics
        avg_prob = np.mean(failure_probs)
        total_cost = np.mean(costs)
        rel_error = np.sqrt(np.mean((np.array(failure_probs) - 7.23e-05)**2)) / 7.23e-05
        
        # Store and display results
        cost_list.append(total_cost)
        err_list.append(rel_error)
        print(f"N={N}: p={avg_prob:.2e}, cost={total_cost:.2e}, error={rel_error:.2e}\n")
        
        # Persistent saving of results
        np.save("MLE_sr_error_list_2.npy", err_list)
        np.save("MLE_sr_cost_list_2.npy", cost_list)