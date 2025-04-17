from fenics import *
import numpy as np

# Set PETSc options to suppress solver output
import os
os.environ["PETSC_OPTIONS"] = "-log_view ascii:out.log"
    

def rRMSE(p_hat):
    p_hat = np.array(p_hat)
    
    difference = p_hat - 1.6e-04
    
    expaction = np.mean(difference ** 2)
    
    return np.sqrt(expaction) / 1.6e-04

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

def mh_sampling(N, G, thetas, c_l, u_max, n_grid, M, gamma = 0.8):
    N0 = len(G)
    
    for i in range(N - N0):
        seed = thetas[i:i+1]
        
        candidate = gamma * seed + np.sqrt(1 - gamma**2) * np.random.normal(0, 1, (1,M))
        
        u_h = IoQ(kl_expan(candidate[0]), n_grid)
        
        if u_max - u_h <= c_l:
            G = np.append(G, u_max - u_h)
            thetas = np.append(thetas, candidate, axis=0)
        else:
            G = np.append(G, G[i])
            thetas = np.append(thetas, seed, axis=0)
            
    return G, thetas

def subset_simulation(N, M = 150, p0 = 0.1, u_max = 0.535, n_grid = 512, gamma = 0.8, L = 4):
    N0 = int(N * p0)
    
    l = 1
    G = np.array([])
    sample_number = np.zeros(L)
    
    thetas = np.random.normal(0, 1, (N, M))
    for i in range(N):
        theta = thetas[i]
        u_1 = IoQ(kl_expan(theta), n_grid)
        g = u_max - u_1
        G = np.append(G, g)
        
    sample_number[0] = N
    
    # Compute the threshold value
    c_l = np.percentile(G, 100 * p0)
    # print(f"c_{l}: ", c_l)
    
    if c_l < 0:
        return len(G <= 0) / N, sample_number
    
    mask = G <= c_l
    G = G[mask][:N0]
    thetas = thetas[mask][:N0, :]
    p_f = p0
    
    for l in range(2, L):
        G, thetas = mh_sampling(N, G, thetas, c_l, u_max, n_grid, M, gamma)
        sample_number[l-1] = N - N0
        
        c_l = np.percentile(G, 100 * p0)
        # print(f"c_{l}: ", c_l)
        
        if c_l <= 0:
            mask = G <= 0
            p_f *= mask.mean()
            return p_f, sample_number
        
        mask = G <= c_l
        G = G[mask][:N0]
        thetas = thetas[mask][:N0, :]
        p_f *= p0
        
    G, _ = mh_sampling(N, G, thetas, c_l, u_max, n_grid, M, gamma)
    sample_number[L-1] = N - N0
    mask = G <= 0
    p_f *= mask.mean()
    
    return p_f, sample_number

def run(args):
    N, seed = args
    np.random.seed(seed)
    
    p_f, sample_number = subset_simulation(N)
    
    return p_f, sample_number

if __name__ == "__main__":
    from tqdm import tqdm
    import multiprocessing as mp
    
    np.random.seed(88)
    error_list = []
    cost_list = []
for N in [100, 200, 400, 800, 1200, 1500, 20009]:
    # for N in [200, 400, 800, 1000, 1500]:
        failure_probabilities = []
        cost = []
        seeds = np.random.randint(100, 10000, 100)
        with mp.Pool(processes=12) as pool:
            # p_f, sample_number = subset_simulation(N)
            results = list(tqdm(pool.imap(run, [(N, seed) for seed in seeds]), total=100, desc=f"N = {N}"))
            
            failure_probabilities = [res[0] for res in results]
            sample_numbers = [res[1] for res in results]
            nums_total = np.sum(sample_numbers)
            
        err = rRMSE(failure_probabilities)
        error_list.append(err)
        print("Relative root mean square error: {:.2e}".format(err))
        cost = nums_total * 0.31**(-6.622*0.9)    # set q = 0.6622 with l = 7
        cost_list.append(cost)
        print("Average cost: {:.2e}\n".format(cost))
        
        np.save("SubS_error_list512.npy", error_list)
        np.save("SubS_cost_list512.npy", cost_list)