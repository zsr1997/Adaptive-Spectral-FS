# core/ga_utils.py
import numpy as np

def init_position(lb, ub, N, dim):
    return lb + (ub - lb) * np.random.rand(N, dim)

def binary_conversion(X, thres):
    return (X > thres).astype(int)

def roulette_wheel(prob):
    C = np.cumsum(prob); C[-1] = 1.0
    P = np.random.rand()
    idx = np.searchsorted(C, P)
    return min(idx, len(prob) - 1)

def jfs(opts, fitness_function):
    ub, lb, thres = 1, 0, 0.5
    CR, MR, N, max_iter = 0.8, 0.01, opts['N'], opts['T']
    dim = opts['xt_in'].shape[1]
    X = binary_conversion(init_position(lb, ub, N, dim), thres)
    
    fit = np.array([fitness_function(X[i, :], opts) for i in range(N)]).reshape(-1, 1)
    best_idx = int(np.argmin(fit)); fitG = fit[best_idx].item()
    
    curve = [fitG] 
    
    for t in range(1, max_iter):
        fit_shifted = fit - np.min(fit)
        inv_fit = 1.0 / (1.0 + fit_shifted + 1e-8)
        prob = inv_fit / inv_fit.sum()
        
        Nc = max(1, int(np.sum(np.random.rand(N) < CR)))
        x1, x2 = np.zeros((Nc, dim), dtype=int), np.zeros((Nc, dim), dtype=int)
        
        for i in range(Nc):
            P1 = X[roulette_wheel(prob)]; P2 = X[roulette_wheel(prob)]
            idx = np.random.randint(1, dim - 1) if dim > 2 else 1
            
            child1 = np.concatenate((P1[:idx], P2[idx:])); child2 = np.concatenate((P2[:idx], P1[idx:]))
            child1 ^= (np.random.rand(dim) < MR).astype(int)
            child2 ^= (np.random.rand(dim) < MR).astype(int)
            x1[i], x2[i] = child1, child2
            
        Xnew = np.vstack((x1, x2))
        Fnew = np.array([fitness_function(Xnew[i, :], opts) for i in range(Xnew.shape[0])]).reshape(-1, 1)
        if np.min(Fnew) < fitG: fitG = np.min(Fnew)
        
        curve.append(fitG) 
        
        XX = np.vstack((X, Xnew)); FF = np.vstack((fit, Fnew))
        keep = np.argsort(FF.ravel())[:N]; X, fit = XX[keep], FF[keep]
        
    return {'X': X, 'fit': fit, 'curve': curve} 

def keep_elite(X, fit, curve=None, elite_ratio=0.2): 
    n = X.shape[0]; ne = max(5, int(np.ceil(n * elite_ratio)))
    idx = np.argsort(fit.ravel())[:ne]
    return X[idx]