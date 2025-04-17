from fenics import *
import numpy as np
import multiprocessing
from tqdm import tqdm
import logging
import os

# Configure logging and computing environment
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
os.environ["PETSC_OPTIONS"] = "-log_view ascii:out.log"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Global constants
TARGET_RISK = 1.6e-04
RELAXATION_FACTOR = 0.31
q = 0.6622
U_MAX = 0.535
BETA = 1 / 0.01
MU = -0.5 * np.log(1.01)
SIGMA = np.sqrt(np.log(1.01))

def rRMSE(failure_probability):
    difference = np.array(failure_probability) - TARGET_RISK
    return np.sqrt(np.mean(difference ** 2)) / TARGET_RISK

def delta(p_hat, I):
    if p_hat == 0:
        return 10
    elif p_hat == 1:
        return 0
    
    N = len(I)
    
    max_lag = N//2
    autocorr = np.correlate(I - np.mean(I), I - np.mean(I), mode='full')[N-1-max_lag:N+max_lag]
    autocorr = autocorr / (np.var(I) * N)
    
    # ess = N / (1 + 2 * np.sum(autocorr[1:max_lag+1]))
    denominator = 1 + 2 * np.sum(np.clip(autocorr[1:max_lag+1], 0, None))
    ess = N / denominator
    return np.sqrt(p_hat * (1 - p_hat) / ess)
    
def compute_y(y_L, L, gamma):
    y = np.zeros(L)
    y[-1] = y_L
    
    for l in range(L-2, -1, -1):
        y[l] = y[l+1] + gamma**(l+4) + gamma**(l+3)
        
    return y

#  compute y as a global variable
y = compute_y(0, 4, 0.31)
print(y)

def kl_expan(theta):
    M = np.size(theta)
    
    x = np.linspace(0, 1, 1000)
    
    beta = 1 / 0.01
    
    def eigenvalue(m):
        w = m * np.pi
        return 2 * beta / (w**2 + beta**2)
    
    def eigenfunction(m, x):
        w = m * np.pi
        A = np.sqrt(2 * w**2 / (2*beta + w**2 + beta**2))
        B = np.sqrt(2 * beta**2 / (2*beta + w**2 + beta**2))
        return A*np.cos(w*x) + B*np.sin(w*x)
    
    mu = -0.5 * np.log(1.01)
    sigma = np.sqrt(np.log(1.01))
    
    log_a_x = mu + sigma * sum(np.sqrt(eigenvalue(m+1)) * eigenfunction(m+1, x) * theta[m] for m in range(M))
    
    a_x = np.exp(log_a_x)
    
    return a_x

def IoQ(a_x, n_grid):
    # Create the mesh and define function space
    mesh = UnitIntervalMesh(n_grid)
    V = FunctionSpace(mesh, 'P', 1)
    
    # Define the random field a(x) on the FEniCS mesh
    a = Function(V)
    a_values = np.interp(mesh.coordinates().flatten(), np.linspace(0, 1, len(a_x)), a_x)
    a.vector()[:] = a_values
    
    # Define boundary condition
    u0 = Constant(0.0)
    def boundary(x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)
    bc = DirichletBC(V, u0, boundary)
    
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(1.0)
    
    a_form = inner(a * grad(u), grad(v)) * dx
    L = f * v * dx
    
    # Compute solution
    u_h = Function(V)
    set_log_level(LogLevel.ERROR)  # Suppress the FEniCS log messages
    solve(a_form == L, u_h, bc)
    
    return u_h(1)

def AMLSuS(args, M = 150, u_max = 0.535, n_grid = 64, gamma = 0.8, L = 4):
    tol, seed = args
    np.random.seed(seed)
    
    G = np.array([])
    # thetas = np.array([])
    sample_numbers = np.zeros(L)
    err = 10
    
    N = int(200 / tol)
    
    thetas = np.random.normal(0, 1, (N,M))
    for i in range(N):
        theta = thetas[i:i+1]
        u_1 = IoQ(kl_expan(theta[0]), n_grid)
        g = u_max - u_1
        G = np.append(G, g)
        
    sample_numbers[0] += N
    
    mask = G <= y[0]
    p_hat = np.mean(mask)
    # print(f"Level 1: p_hat = {p_hat:.2e}")
    
    if p_hat == 0 or p_hat == 1:
        err = 10
    else:
        err = p_hat * (1 - p_hat) / sample_numbers[0]
    
    # Level 1
    tol /= np.sqrt(L)
    while err > tol:
        theta = np.random.normal(0, 1, (1,M))
        G_new = u_max - IoQ(kl_expan(theta[0]), n_grid)
        sample_numbers[0] += 1
        
        G = np.append(G, G_new)
        thetas = np.append(thetas, theta[:], axis=0)
        
        mask = G <= y[0]
        p_hat = np.mean(mask)
        
        if p_hat == 0:
            err = 10
        else:
            err = p_hat * (1 - p_hat) / sample_numbers[0]
            
        # if sample_numbers[0] % 1000 == 0:
        #     print(f"Level 1: iteration = {sample_numbers[0]:.0f}, err = {err:.2e}, p_hat = {p_hat:.2e}")
        #     if p_hat == 1:
        #         err = 0
            
    p_f = p_hat
    # print(f"Level 1: p_f = {p_f:.2e}, err = {err:.2e}, sample number = {sample_numbers[0]:.0f}")
    
    # Level 2 to L
    tol /= np.sqrt(L)
    for l in range(1, L):
        n_grid *= 2
        G = G[mask][:1]
        thetas = thetas[mask][:1, :]
        acc = 0
        
        if L > 1:
            N = int(N * 0.5)
        
        for i in range(N):
            seed = thetas[i:i+1]
            theta = gamma * seed + np.sqrt(1 - gamma**2) * np.random.normal(0, 1, (1,M))
            G_new = u_max - IoQ(kl_expan(theta[0]), n_grid)
            
            if G_new <= y[l-1]:
                G = np.append(G, G_new)
                thetas = np.append(thetas, theta, axis=0)
                acc += 1
            else:
                G = np.append(G, G[-1])
                thetas = np.append(thetas, seed, axis=0)
            
        sample_numbers[l] += N
        mask = G <= y[l]
        p_hat = np.mean(mask)
        
        err = delta(p_hat, mask)
        
        while err > tol:
            # print(thetas[-1])
            theta = gamma * thetas[-1] + np.sqrt(1 - gamma**2) * np.random.normal(0, 1, (1,M))
            G_new = u_max - IoQ(kl_expan(theta[0]), n_grid)
            
            if G_new <= y[l-1]:
                G = np.append(G, G_new)
                thetas = np.append(thetas, theta, axis=0)
                acc += 1
            else:
                G = np.append(G, G[-1])
                thetas = np.append(thetas, thetas[-1:], axis=0)
                
            sample_numbers[l] += 1
            
            mask = G <= y[l]
            p_hat = np.mean(mask)
            
            err = delta(p_hat, mask)
            # if sample_numbers[l] % 1000 == 0:
                # print(f"Level {l+1}: iteration = {sample_numbers[l]:.0f}, err = {err:.2e}, p_hat = {p_hat:.2e}, acc_rate = {acc/sample_numbers[l]:.2f}")
                # if p_hat == 1:
                #     err = 0
            
        p_f *= p_hat
        # print(f"Level {l+1}: p_f = {p_f:.2e}, p_hat = {p_hat:.2e}, err = {err:.2e}, sample number = {sample_numbers[l]:.0f}, acc_rate = {acc/sample_numbers[l]:.2f}")
        
    return p_f, sample_numbers

if __name__ == "__main__":
    np.random.seed(367)
    # np.random.seed(663)
    
    error_list = []
    cost_list = []
    
    for tol in [9e-1, 8e-1, 7e-1, 6e-1, 5e-1, 4e-1, 3e-1]:
        seeds = np.random.randint(10, 10000, 100)
        args = [(tol, seed) for seed in seeds]
        with multiprocessing.Pool(processes=12) as pool:
            results = list(tqdm(pool.imap(AMLSuS, args), total=100, desc=f"tol = {tol:.2e}"))
            
        failure_probabilities = [res[0] for res in results]
        sample_numbers_arr = [res[1] for res in results]
        sample_numbers_mean = np.mean(sample_numbers_arr, axis=0)
            
        error = rRMSE(failure_probabilities)
        error_list.append(error)
        
        # exp = RELAXATION_FACTOR ** (-q * np.linspace(7, 7+3, num=4))
        exp = RELAXATION_FACTOR ** (-q * np.linspace(6, 9, num=4))
        cost = np.sum(sample_numbers_mean * exp)
        cost_list.append(cost)
            
        print(f"Failure probability: {np.mean(failure_probabilities):.2e}")            
        print(f"Average error: {error:.2e}")
        print(f"Average cost: {cost:.2e}")
        print(f"Average sample numbers: {sample_numbers_mean}")
        
        np.save(f"amlsus_error_list64.npy", error_list)
        np.save(f"amlsus_cost_list64.npy", cost_list)
    