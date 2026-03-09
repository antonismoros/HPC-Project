import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from getaccfunc import getAcc
import time
import numpy as np
import dask.array as da
from dask.distributed import Client

NList = [10000000, 100000, 1000000, 10000000]
cores=4
timings = []
def main():


    boxsize = 50.0
    Nx = 400
    n0 = 1.0
    dt = 1.0
    
    for N in NList:
        
        starttime = time.time()
        dx = boxsize / Nx
        e = np.ones(Nx)
        diags = np.array([-1, 1])
        vals = np.vstack((-e, e))
        Gmtx = sp.spdiags(vals, diags, Nx, Nx)
        Gmtx = sp.lil_matrix(Gmtx)
        Gmtx[0, Nx - 1] = -1
        Gmtx[Nx - 1, 0] = 1
        Gmtx /= 2 * dx
        Gmtx = sp.csr_matrix(Gmtx)
     
        pos_np = np.random.rand(N, 1) * boxsize

        pos = da.from_array(pos_np, chunks=(N // cores, 1))

        diags = np.array([-1, 0, 1])
        vals = np.vstack((e, -2 * e, e))
        Lmtx = sp.spdiags(vals, diags, Nx, Nx)
        Lmtx = sp.lil_matrix(Lmtx)
        Lmtx[0, Nx - 1] = 1
        Lmtx[Nx - 1, 0] = 1
        Lmtx /= dx**2
        Lmtx = sp.csr_matrix(Lmtx)
        
        vel_np = np.random.randn(N, 1)
        vel = da.from_array(vel_np, chunks=(N // cores, 1))

        # 3. Parallel Execution of Cython Kernel
 
        acc = pos.map_blocks(
            getAcc, 
            Nx=Nx, boxsize=boxsize, n0=n0, Gmtx=Gmtx, Lmtx=Lmtx,
            dtype=float
        )

        # Compute the results .compute()
        final_acc = acc.compute()
        
        endtime = time.time()
        print(f"N={N} took {endtime-starttime:.4f} seconds with Dask + Cython with 4 cores") ; timings.append(endtime-starttime)
    '''plt.xscale('log')
    plt.xticks(NList)       
    plt.title( f"Runtime with {cores} workers")
    plt.xlabel('Number of Particles')
    plt.ylabel('Time (s)')
    plt.plot(NList, timings)
    plt.savefig("dask_timings4.png", dpi=300 ) '''

if __name__ == "__main__":
    client = Client(n_workers=cores, threads_per_worker=1)
    print(f"Dashboard is at: {client.dashboard_link}")
    
    main() 

    input("Simulation complete. Press Enter to close the Dashboard...")
    


