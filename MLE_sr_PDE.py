from fenics import *
import numpy as np
import multiprocessing
from tqdm import tqdm
import os
import logging

# -------------------------
# Configure logging, environment variables, and constants
# -------------------------

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Set PETSc options to suppress solver output and write to a file
os.environ["PETSC_OPTIONS"] = "-log_view ascii:out.log"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Global constants
TARGET_RISK = 1.6e-04      # Used to calculate relative root mean square error
RELAXATION_FACTOR = 0.31   # Used to control refinement conditions
q = 0.6622
U_MAX = 0.535              # Critical value u_max
BETA = 1 / 0.01            # Beta used in KL expansion
MU = -0.5 * np.log(1.01)     # Log-normal mean parameter
SIGMA = np.sqrt(np.log(1.01))  # Log-normal standard deviation parameter

# -------------------------
# Helper functions
# -------------------------
def rRMSE(failure_probability):
    """
    Calculate the relative root mean square error (rRMSE) between failure_probability and TARGET_RISK
    """
    difference = np.array(failure_probability) - TARGET_RISK
    return np.sqrt(np.mean(difference ** 2)) / TARGET_RISK

def kl_expan(theta):
    """
    Vectorized Karhunen-Loève expansion to generate the random field a(x)
    
    Parameters:
      theta: 1D array of length M
      
    Returns:
      a_x: Values of the random field at 1000 points in [0,1]
    """
    M = len(theta)
    x = np.linspace(0, 1, 1000)
    
    # Construct mode indices and calculate corresponding angular frequencies w
    m_vals = np.arange(1, M+1).reshape(-1, 1)  # shape: (M,1)
    w = m_vals * np.pi  # shape: (M,1)
    
    # Calculate eigenvalues lambda and their square roots
    lambda_vals = 2 * BETA / (w**2 + BETA**2)  # shape: (M,1)
    sqrt_lambda = np.sqrt(lambda_vals)
    
    # Calculate eigenfunctions
    A = np.sqrt(2 * w**2 / (2 * BETA + w**2 + BETA**2))
    B = np.sqrt(2 * BETA**2 / (2 * BETA + w**2 + BETA**2))
    phi = A * np.cos(w * x) + B * np.sin(w * x)  # shape: (M, len(x))
    
    # Calculate log_a(x) and take the exponential to get a(x)
    log_a_x = MU + SIGMA * np.sum(sqrt_lambda * phi * theta.reshape(-1, 1), axis=0)
    a_x = np.exp(log_a_x)
    return a_x

def IoQ(a_x, n_grid):
    """
    Solve the PDE and return the solution value at x=1
    
    Parameters:
      a_x: Numerical array of the random field a(x)
      n_grid: Number of grid divisions, used to construct UnitIntervalMesh(n_grid)
      
    Returns:
      u_h(1): Value of the numerical solution of the PDE at x=1
    """
    mesh = UnitIntervalMesh(n_grid)
    V = FunctionSpace(mesh, 'P', 1)
    
    # Get node coordinates using V.tabulate_dof_coordinates()
    coordinates = V.tabulate_dof_coordinates().reshape(-1)
    a_values = np.interp(coordinates, np.linspace(0, 1, len(a_x)), a_x)
    a = Function(V)
    a.vector()[:] = a_values
    
    # Define boundary condition at x=0
    u0 = Constant(0.0)
    def boundary(x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)
    bc = DirichletBC(V, u0, boundary)
    
    # Define weak form
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(1.0)
    a_form = inner(a * grad(u), grad(v)) * dx
    L_form = f * v * dx
    
    # Solve the PDE
    u_h = Function(V)
    set_log_level(LogLevel.ERROR)
    solve(a_form == L_form, u_h, bc)
    
    return u_h(1)

def mh_sampling(N, G, thetas, c_l, l, u_max, n_grid, M=150, gamma=0.8):
    """
    Selective refinement sampling based on the Metropolis-Hastings method

    Parameters:
      N: Total number of samples
      G: Current failure indicator array
      thetas: Current theta sample array, shape (n_samples, M)
      c_l: Threshold for the current level
      l: Current level (integer)
      u_max: Critical value
      n_grid: Initial number of grid divisions
      M: Dimension of theta (default 150)
      gamma: Scaling factor for constructing candidate samples
      
    Returns:
      Updated G, thetas, and sampling statistics for each level (samples_numbers array)
    """
    N0 = len(G)
    n_grid_0 = n_grid
    samples_numbers = np.zeros(7, dtype=int)
    
    for i in range(N - N0):
        tol = RELAXATION_FACTOR
        current_n_grid = n_grid_0
        # Use the current sample as the seed
        seed = thetas[i:i+1]
        candidate = gamma * seed + np.sqrt(1 - gamma**2) * np.random.normal(0, 1, (1, M))
        
        # Calculate the indicator g = u_max - IoQ(kl_expan(candidate), n_grid)
        g = u_max - IoQ(kl_expan(candidate[0]), current_n_grid)
        samples_numbers[0] += 1
        
        # Refinement iteration: refine the grid based on the distance between g and c_l
        for j in range(1, l):
            if tol >= np.abs(g - c_l):
                tol *= RELAXATION_FACTOR
                current_n_grid *= 2
                g = u_max - IoQ(kl_expan(candidate[0]), current_n_grid)
                samples_numbers[j] += 1
            else:
                break
        
        if g <= c_l:
            G = np.append(G, g)
            thetas = np.append(thetas, candidate, axis=0)
        else:
            G = np.append(G, G[i])
            thetas = np.append(thetas, seed, axis=0)
    
    return G, thetas, samples_numbers

def mle(args):
    """
    Multi-level estimator (MLE) with selective refinement sampling for failure probability estimation

    Parameters:
      args: Tuple (N, seed)
          N: Total number of samples
          seed: Random seed for each process
          
    Returns:
      (Estimated failure probability, sampling statistics for each level)
    """
    N, seed = args
    np.random.seed(seed)
    
    # Parameter settings
    p0 = 0.25
    M = 150
    L_b = 10
    u_max = U_MAX
    n_grid = 64
    L = 7
    
    N0 = int(N * p0)
    P = 100 * p0  # Percentile
    
    # Generate initial theta samples
    theta_ls = np.random.normal(0, 1, (N, M))
    G = np.zeros(N)
    
    sample_numbers = np.zeros(L, dtype=int)
    for i in range(N):
        g = u_max - IoQ(kl_expan(theta_ls[i, :]), n_grid)
        G[i] = g
    sample_numbers[0] += N
    
    # Calculate the first level threshold c_l (percentile)
    c_l = np.percentile(G, P)
    
    # If the threshold is negative, return directly
    if c_l < 0:
        return np.mean(G <= 0), sample_numbers
    else:
        denominator = 1.0
    
    mask = G <= c_l
    G = G[mask][:N0]
    theta_ls = theta_ls[mask][:N0, :]
    
    # Second-level sampling (no burn-in)
    c_l_prev = c_l
    G, theta_ls, add_num = mh_sampling(N, G, theta_ls, c_l, 2, u_max, n_grid)
    c_l = np.percentile(G, P)
    mask = G <= c_l
    G = G[mask][:N0]
    theta_ls = theta_ls[mask][:N0, :]
    sample_numbers += add_num

    G_temp, _, add_num = mh_sampling(N, G, theta_ls, c_l, 2, u_max, n_grid)
    sample_numbers += add_num
    denominator *= np.mean(G_temp <= c_l_prev)
    
    if c_l <= 0:
        return p0 * np.mean(G <= 0) / denominator, sample_numbers

    # Third-level and subsequent sampling (with burn-in)
    N += L_b * N0
    for l in range(3, L):
        c_l_prev = c_l
        G, theta_ls, add_num = mh_sampling(N, G, theta_ls, c_l, l, u_max, n_grid)
        # Discard burn-in samples
        G = G[L_b * N0:]
        theta_ls = theta_ls[L_b * N0:, :]
        c_l = np.percentile(G, P)
        mask = G <= c_l
        G = G[mask][:N0]
        theta_ls = theta_ls[mask][:N0, :]
        sample_numbers += add_num

        if c_l <= 0:
            G_temp, _, add_num = mh_sampling(N, G, theta_ls, 0, l, u_max, n_grid)
            sample_numbers += add_num
            G_temp = G_temp[L_b * N0:]
            denominator *= np.mean(G_temp <= c_l_prev)
            
            return p0 ** (l-1) * np.mean(G <= 0) / denominator, sample_numbers

        G_temp, _, add_num = mh_sampling(N, G, theta_ls, c_l, l, u_max, n_grid)
        G_temp = G_temp[L_b * N0:]
        sample_numbers += add_num
        denominator *= np.mean(G_temp <= c_l_prev)
    
    # Final-level sampling
    G, theta_ls, add_num = mh_sampling(N, G, theta_ls, c_l, L, u_max, n_grid)
    sample_numbers += add_num
    G = G[L_b * N0:]
    theta_ls = theta_ls[L_b * N0:, :]
    
    mask = G <= 0
    G = G[mask][:N0]
    theta_ls = theta_ls[mask][:N0, :]
    
    G_temp, _, add_num = mh_sampling(N, G, theta_ls, 0, L, u_max, n_grid)
    G_temp = G_temp[L_b * N0:]
    sample_numbers += add_num
    denominator *= np.mean(G_temp <= c_l)
    
    return p0 ** (L-1) * np.mean(mask) / denominator, sample_numbers

# -------------------------
# Main program entry
# -------------------------
if __name__ == "__main__":
    error_list = []
    cost_list = []
    sample_number_list = []
    np.random.seed(19)
    # Experiment with different sample sizes N
    for N in [100, 200, 400, 600, 800, 1200]:
        with multiprocessing.Pool(processes=12) as pool:
            seeds = np.random.randint(10, 10000, 100)
            args = [(N, int(seed)) for seed in seeds]  # Assign different seeds to each subprocess
            results = list(tqdm(pool.imap(mle, args), total=100, desc=f"N = {N}"))
        
        # Collect results from subprocesses
        failure_probabilities = [res[0] for res in results]
        sample_numbers_arr = [res[1] for res in results]
        sample_numbers_mean = np.mean(sample_numbers_arr, axis=0)
        logging.info(f"Sample numbers for N={N}: {sample_numbers_mean}")
        sample_number_list.append(sample_numbers_mean)
        
        p_f = np.mean(failure_probabilities)
        err = rRMSE(failure_probabilities)
        error_list.append(err)
        
        # Estimate cost (use refinement ratio to calculate cost weights)
        exp = RELAXATION_FACTOR ** (-q * np.linspace(6, 6+6, num=7)) # set q = 0.6622 with l = 6, ..., 13
        cost = np.sum(sample_numbers_mean * exp)
        cost_list.append(cost)
        
        print(f"Failure probability: {p_f:.2e}")
        print(f"Error: {err:.2e}")
        print(f"Cost: {cost:.2e}\n")
        
        # Save results (update after each iteration)
        np.save("mle_sr_sample_numbers64_2.npy", sample_number_list)
        np.save("mle_sr_error_list64_2.npy", error_list)
        np.save("mle_sr_cost_list64_2.npy", cost_list)
