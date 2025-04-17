from fenics import *
import numpy as np

TARGET_RISK = 1.6e-04      # Used to calculate relative root mean square error
RELAXATION_FACTOR = 0.31   # Used to control refinement conditions
q = 0.6622
U_MAX = 0.535              # Critical value u_max
BETA = 1 / 0.01            # Beta used in KL expansion
MU = -0.5 * np.log(1.01)     # Log-normal mean parameter
SIGMA = np.sqrt(np.log(1.01))  # Log-normal standard deviation parameter

def rRMSE(p_hat):
    difference = np.array(p_hat) - TARGET_RISK
    return np.sqrt(np.mean(difference ** 2)) / TARGET_RISK

# def kl_expan(theta):
#     M = len(theta)
#     x = np.linspace(0, 1, 1000)
    
#     m_vals = np.arange(1, M+1).reshape(-1, 1)  # shape: (M,1)
#     w = m_vals * np.pi  # shape: (M,1)
    
#     lambda_vals = 2 * BETA / (w**2 + BETA**2)  # shape: (M,1)
#     sqrt_lambda = np.sqrt(lambda_vals)
    
#     A = np.sqrt(2 * w**2 / (2 * BETA + w**2 + BETA**2))
#     B = np.sqrt(2 * BETA**2 / (2 * BETA + w**2 + BETA**2))
#     phi = A * np.cos(w * x) + B * np.sin(w * x)  # shape: (M, len(x))
    
#     log_a_x = MU + SIGMA * np.sum(sqrt_lambda * phi * theta.reshape(-1, 1), axis=0)
#     a_x = np.exp(log_a_x)
#     return a_x

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
    # Set PETSc options to suppress solver output
    import os

    os.environ["PETSC_OPTIONS"] = "-log_view ascii:out.log"

    # Create the mesh and define function space
    mesh = UnitIntervalMesh(n_grid)
    V = FunctionSpace(mesh, "P", 1)

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


def mh_sampling(N, G, thetas, c_l, u_max, n_grid, M, gamma=0.8):
    N0 = len(G)

    for i in range(N - N0):
        seed = thetas[i : i + 1]

        candidate = gamma * seed + np.sqrt(1 - gamma**2) * np.random.normal(0, 1, (1, M))

        u_h = IoQ(kl_expan(candidate[0]), n_grid)

        if u_max - u_h <= c_l:
            G = np.append(G, u_max - u_h)
            thetas = np.append(thetas, candidate, axis=0)
        else:
            G = np.append(G, G[i])
            thetas = np.append(thetas, seed, axis=0)

    return G, thetas


def subset_simulation(args, M=150, u_max=0.535, n_grid=512, gamma=0.8, L=4):
    N, seed = args
    np.random.seed(seed)  # Unique seed for each worker
    
    y = [0.015, 0.009, 0.005, 0]
    # y = [0.014, 0.007, 0.002, 0]

    l = 1
    mask = 0
    while np.sum(mask) == 0:    # In case that the thetas are all rejected
        G = np.zeros(N)
        thetas = np.zeros((N, M))
        sample_number = np.zeros(L)

        for i in range(N):
            theta = np.random.normal(0, 1, M)
            thetas[i] = theta
            u_1 = IoQ(kl_expan(theta), n_grid)
            G[i] = u_max - u_1

        sample_number[0] += N

        mask = G <= y[0]
    
    # Only keep the first failure sample
    G = G[mask][:1]
    thetas = thetas[mask][:1][:]
    p_f = np.mean(mask)
    # print("p_f at level 1: ", p_f)

    for l in range(2, L):
        mask = 0
        while np.sum(mask) == 0:
            G_, thetas_ = mh_sampling(N, G, thetas, y[l-2], u_max, n_grid, M, gamma)
            sample_number[l-1] += N - 1

            mask = G_ <= y[l-1]
            
        # Only keep the first failure sample
        G = G_[mask][:1]
        thetas = thetas_[mask][:1][:]
        p_f *= np.mean(mask)
        # print(f"p_f at level {l}: ", np.mean(mask))

    G_, _ = mh_sampling(N, G, thetas, y[L-2], u_max, n_grid, M, gamma)
    sample_number[L-1] += N - 1
    mask = G_ <= 0
        
    p_f *= mask.mean()
    # print("p_f at level 4: ", np.mean(mask))
    # print("p_f: {:.2e}".format(p_f))

    return p_f, sample_number


if __name__ == "__main__":
    from tqdm import tqdm
    import multiprocessing as mp

    np.random.seed(68)
    
    error_list = []
    cost_list = []
    sample_numbers = []
        
    for N in [100, 200, 500, 800, 1000, 1200, 1500, 2000]:
    # for N in [200, 400, 600, 800]:
        seeds = np.random.randint(100, 10000, 100)
        with mp.Pool(processes=12) as pool:
            args = [(N, seed) for seed in seeds]
            results = list(tqdm(pool.imap(subset_simulation, args), total=100, desc=f"N = {N}"))
            
            failure_probabilities = [res[0] for res in results]
            # print(failure_probabilities)
            sample_number = [res[1] for res in results]
            
        error = rRMSE(failure_probabilities)
        error_list.append(error)
        cost = np.sum(sample_number) * RELAXATION_FACTOR**(-6.622*0.9) # set q = 0.6622 with l = 7
        cost_list.append(cost)
        print(f"rRMSE: {error:.2e}, Cost: {cost:.2e}\n")
            
        np.save("SubS_sr_error_list512.npy", error_list)
        np.save("SubS_sr_cost_list512.npy", cost_list)

