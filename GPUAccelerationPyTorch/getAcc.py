import torch
import numpy as np
from scipy.sparse.linalg import spsolve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getAcc_gpu(pos, Nx, boxsize, n0, Gmtx, Lmtx):

    N = pos.shape[0]
    dx = boxsize / Nx

    # ----- particle → grid -----
    frac = pos[:,0] / dx
    j = torch.floor(frac).long() % Nx

    wjp1 = frac - j
    wj = 1.0 - wjp1

    

    jp1 = (j + 1) % Nx

    n = torch.zeros(Nx, device=device)

    j = torch.clamp(j, 0, Nx-1)

    n.index_add_(0, j, wj)
    n.index_add_(0, jp1, wjp1)

    n = n * (n0 * boxsize / N / dx)

    # ----- solve Poisson (CPU) -----
    n_cpu = n.cpu().numpy()

    rho = n_cpu - n0
    rho -= rho.mean()

    # Fix gauge to avoid singular matrix
    L = Lmtx.copy().tolil()
    L[0,:] = 0
    L[0,0] = 1
    rho[0] = 0

    phi_grid = spsolve(L.tocsr(), rho)

    # ----- electric field -----
    E_grid = -(Gmtx @ phi_grid)

    E_grid = torch.tensor(E_grid, device=device)

    # ----- grid → particle -----
    E = wj * E_grid[j] + wjp1 * E_grid[jp1]

    a = -E.unsqueeze(1)

    return a