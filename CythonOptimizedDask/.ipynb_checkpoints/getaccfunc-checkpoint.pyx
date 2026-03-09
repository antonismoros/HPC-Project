import numpy as np
cimport numpy as cnp
from scipy.sparse.linalg import spsolve

# cython optimizations
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

def deposit_particles(
        cnp.ndarray[double, ndim=2] pos,
        int Nx,
        double boxsize,
        double n0):

    cdef:
        int i, cell
        int N = pos.shape[0]
        double dx = boxsize / Nx
        double frac
        double wj, wjp1

    cdef cnp.ndarray[double] n = np.zeros(Nx)
    cdef cnp.ndarray[int] j = np.zeros(N, dtype=np.int32)
    cdef cnp.ndarray[int] jp1 = np.zeros(N, dtype=np.int32)
    cdef cnp.ndarray[double] weight_j = np.zeros(N)
    cdef cnp.ndarray[double] weight_jp1 = np.zeros(N)

    cdef double[:] n_view = n
    cdef int[:] j_view = j
    cdef int[:] jp1_view = jp1
    cdef double[:] wj_view = weight_j
    cdef double[:] wjp1_view = weight_jp1

    for i in range(N):

        frac = pos[i,0] / dx
        cell = <int>frac

        wjp1 = frac - cell
        wj = 1.0 - wjp1

        j_view[i] = cell
        jp1_view[i] = (cell + 1) % Nx
        wj_view[i] = wj
        wjp1_view[i] = wjp1

        n_view[cell] += wj
        n_view[(cell + 1) % Nx] += wjp1

    n *= n0 * boxsize / N / dx

    return n, j, jp1, weight_j, weight_jp1

def getAcc(cnp.ndarray[double, ndim=2] pos,
           int Nx,
           double boxsize,
           double n0,
           Gmtx,
           Lmtx):

    cdef:
        int i
        int N = pos.shape[0]

    n, j, jp1, weight_j, weight_jp1 = deposit_particles(pos, Nx, boxsize, n0)

    phi_grid = spsolve(Lmtx, n - n0, permc_spec="MMD_AT_PLUS_A")

    E_grid = -Gmtx @ phi_grid

    cdef cnp.ndarray[double, ndim=2] E = np.zeros((N,1))
    cdef double[:, :] E_view = E

    cdef double[:] Egrid_view = E_grid
    cdef int[:] j_view = j
    cdef int[:] jp1_view = jp1
    cdef double[:] wj_view = weight_j
    cdef double[:] wjp1_view = weight_jp1

    for i in range(N):
        E_view[i,0] = wj_view[i]*Egrid_view[j_view[i]] + \
                      wjp1_view[i]*Egrid_view[jp1_view[i]]

    a = -E

    return a