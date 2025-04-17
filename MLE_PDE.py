from fenics import *
import numpy as np
import multiprocessing
from tqdm import tqdm
import os

# Set PETSc options to suppress solver output (and write to file)
os.environ["PETSC_OPTIONS"] = "-log_view ascii:out.log"

def rRMSE(p_hat):
    p_hat = np.array(p_hat)
    difference = p_hat - 1.6e-04
    return np.sqrt(np.mean(difference ** 2)) / 1.6e-04


def kl_expan(theta):
    BETA = 1 / 0.01
    MU = -0.5 * np.log(1.01)
    SIGMA = np.sqrt(np.log(1.01))
    
    M = len(theta)
    x = np.linspace(0, 1, 1000)
    
    m_vals = np.arange(1, M+1).reshape(-1, 1)  # shape: (M,1)
    w = m_vals * np.pi  # shape: (M,1)
    
    lambda_vals = 2 * BETA / (w**2 + BETA**2)  # shape: (M,1)
    sqrt_lambda = np.sqrt(lambda_vals)
    
    A = np.sqrt(2 * w**2 / (2 * BETA + w**2 + BETA**2))
    B = np.sqrt(2 * BETA**2 / (2 * BETA + w**2 + BETA**2))
    phi = A * np.cos(w * x) + B * np.sin(w * x)  # shape: (M, len(x))
    
    log_a_x = MU + SIGMA * np.sum(sqrt_lambda * phi * theta.reshape(-1, 1), axis=0)
    a_x = np.exp(log_a_x)
    
    return a_x


def IoQ(a_x, n_grid):
    mesh = UnitIntervalMesh(n_grid)
    V = FunctionSpace(mesh, 'P', 1)

    coordinates = V.tabulate_dof_coordinates().reshape(-1)
    a_values = np.interp(coordinates, np.linspace(0, 1, len(a_x)), a_x)
    a = Function(V)
    a.vector()[:] = a_values

    u0 = Constant(0.0)

    def boundary(x, on_boundary):
        return on_boundary and near(x[0], 0, DOLFIN_EPS)

    bc = DirichletBC(V, u0, boundary)

    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(1.0)

    a_form = inner(a * grad(u), grad(v)) * dx
    L = f * v * dx

    u_h = Function(V)
    set_log_level(LogLevel.ERROR)
    solve(a_form == L, u_h, bc)
    
    return u_h(1)


def mh_sampling(N, G, thetas, c_l, u_max, n_grid, M, gamma=0.8):
    N0 = len(G)
    for i in range(N - N0):
        seed = thetas[i : i+1]
        candidate = gamma * seed + np.sqrt(1 - gamma**2) * np.random.normal(0, 1, (1, M))
        u_h = IoQ(kl_expan(candidate[0]), n_grid)
        if u_max - u_h <= c_l:
            G = np.append(G, u_max - u_h)
            thetas = np.append(thetas, candidate, axis=0)
        else:
            G = np.append(G, G[i])
            thetas = np.append(thetas, seed, axis=0)
    return G, thetas


def mle(args):
    N, seed = args
    np.random.seed(seed)  # Unique seed for each worker
    
    p0 = 0.25
    M = 150
    L_b = 10
    u_max = 0.535
    n_grid = 64
    L = 7
    
    N0 = int(N * p0)
    P = 100 * p0
    
    # l = 1
    G = np.zeros(N)
    sample_numbers = np.zeros(L)
    
    theta_ls = np.random.normal(0, 1, (N, M))
    for i in range(N):
        u_1 = IoQ(kl_expan(theta_ls[i, :]), n_grid)
        G[i] = u_max - u_1
        
    sample_numbers[0] = N
    
    c_l = np.percentile(G, P)
    
    if c_l < 0:
        return np.mean(G <= 0), sample_numbers
    
    denominator = 1
    mask = G <= c_l
    G = G[mask][:N0]
    theta_ls = theta_ls[mask][:N0, :]

    # l = 2
    c_l_1 = c_l
    n_grid *= 2
    sample_numbers[1] += N - N0
    G, theta_ls = mh_sampling(N, G, theta_ls, c_l, u_max, n_grid, M, gamma=0.8)
    
    c_l = np.percentile(G, P)
    
    mask = G <= c_l
    G = G[mask][:N0]
    theta_ls = theta_ls[mask][:N0, :]
    
    sample_numbers[1] += N - N0
    G_, _ = mh_sampling(N, G, theta_ls, c_l, u_max, n_grid, M, gamma=0.8)
    
    denominator *= np.mean(G_ <= c_l_1)
    
    if c_l < 0:
        return p0 * np.mean(G <= 0) / denominator, sample_numbers
    
    # l > 2, l = 3, ..., L-1
    N += L_b * N0
    for l in range(3, L):
        c_l_1 = c_l
        n_grid *= 2
        
        sample_numbers[l-1] += N - N0
        G, theta_ls = mh_sampling(N, G, theta_ls, c_l, u_max, n_grid, M, gamma=0.8)
        
        G = G[L_b * N0:]
        theta_ls = theta_ls[L_b * N0:, :]
        
        c_l = np.percentile(G, P)
        # print("c_", l, ": ", c_l)
        
        if c_l <= 0:
            mask = G <= 0
            return p0 ** (l-1) * mask.mean() / denominator, sample_numbers
        
        mask = G <= c_l
        G = G[mask][:N0]
        theta_ls = theta_ls[mask][:N0, :]

        # Resample
        sample_numbers[l-1] += N - N0
        G_, _ = mh_sampling(N, G, theta_ls, c_l, u_max, n_grid, M, gamma=0.8)
        G_ = G_[L_b * N0:]
        
        denominator *= np.mean(G_ <= c_l_1)
    
    # l = L
    n_grid *= 2
    sample_numbers[L-1] += N - N0
    G, theta_ls = mh_sampling(N, G, theta_ls, c_l, u_max, n_grid, M, gamma=0.8)
    G = G[L_b * N0:]
    theta_ls = theta_ls[L_b * N0:, :]
    
    mask = G <= 0
    G = G[mask][:N0]
    theta_ls = theta_ls[mask][:N0, :]
    
    # Resample
    sample_numbers[L-1] += N - N0
    G_, _ = mh_sampling(N, G, theta_ls, 0, u_max, n_grid, M, gamma=0.8)
    G_ = G_[L_b * N0:]
    
    denominator *= np.mean(G_ <= c_l)
    
    return p0 ** (L-1) * np.mean(mask) / denominator, sample_numbers

if __name__ == "__main__":
    np.random.seed(199)
    error_list = []
    cost_list = []

    # for N in [100, 200, 400, 800]:
    for N in [100, 200, 400, 600, 800, 1200]:
    # for N in [1000]:
        seeds = np.random.randint(0, 10000, 100)
        with multiprocessing.Pool(processes=12) as pool:
            args = [(N, seed) for seed in seeds]  # Unique seed offset for each worker
            results = list(tqdm(pool.imap(mle, args), total=100, desc=f"N = {N}"))
            
            failure_probabilities = [res[0] for res in results]
            sample_numbers = [res[1] for res in results]
            
        p_f = np.mean(failure_probabilities)

        s = np.mean(sample_numbers, axis=0)
        exp = 0.31 ** (-0.6622 * np.linspace(6, 12, num=7))    # set q = 2 with l
        cost = np.sum(s * exp)
        cost_list.append(cost)
        error = rRMSE(failure_probabilities)
        error_list.append(error)

        print(f"Failure probability: {p_f:.2e}")
        print(f"Error: {error:.2e}")
        print(f"Cost: {cost:.2e}\n")
    
        np.save("mle_error_list64.npy", error_list)
        np.save("mle_cost_list64.npy", cost_list)