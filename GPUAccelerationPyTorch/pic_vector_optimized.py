import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from getAcc import getAcc_gpu
import time
import torch

def main():
    """Plasma PIC simulation"""
    

    # Simulation parameters
    NList = [10000, 100000,1000000,10000000]  # Number of particles
    Nx = 400  # Number of mesh cells
    t = 0  # current time of the simulation
    tEnd = 50  # time at which simulation ends
    dt = 1  # timestep
    boxsize = 50  # periodic domain [0,boxsize]
    n0 = 1  # electron number density
    vb = 3  # beam velocity
    vth = 1  # beam width
    A = 0.1  # perturbation
    plotRealTime = True  # switch on for plotting as the simulation goes along
    TimeTaken = []
    # Generate Initial Conditions
    np.random.seed(42)  # set the random number generator seed
    # construct 2 opposite-moving Gaussian beams

    for N in NList:

        starttime = time.time()

        pos = torch.rand(N,1, device='cuda') * boxsize
        vel = vth * torch.randn(N,1, device='cuda') + vb
        Nh = int(N / 2)
        vel[Nh:] *= -1
        # add perturbation
        vel *= 1 + A * torch.sin(2 * torch.pi * pos / boxsize)

        # Construct matrix G to computer Gradient  (1st derivative)
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

        # Construct matrix L to computer Laplacian (2nd derivative)
        diags = np.array([-1, 0, 1])
        vals = np.vstack((e, -2 * e, e))
        Lmtx = sp.spdiags(vals, diags, Nx, Nx)
        Lmtx = sp.lil_matrix(Lmtx)
        Lmtx[0, Nx - 1] = 1
        Lmtx[Nx - 1, 0] = 1
        Lmtx /= dx**2
        Lmtx = sp.csr_matrix(Lmtx)

        # calculate initial gravitational accelerations
        acc = getAcc_gpu(pos, Nx, boxsize, n0, Gmtx, Lmtx)

        # number of timesteps
        Nt = int(np.ceil(tEnd / dt))

        # prep figure
        
        fig, ax = plt.subplots(figsize=(6,4))
        scatter = ax.scatter(
            pos[:5000].flatten().cpu().numpy(),
            vel[:5000].flatten().cpu().numpy(),
            s=1
        )

        ax.set_xlim(0, boxsize)
        ax.set_ylim(-6,6)
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.set_title("Two Stream Instability Phase Space")

        plt.ion()
        plt.show()
        
        # Simulation Main Loop
        for i in range(Nt):
            # (1/2) kick
            vel += acc * dt / 2.0

            # drift (and apply periodic boundary conditions)
            pos += vel * dt
            pos = torch.remainder(pos, boxsize)

            # update accelerations
            acc = getAcc_gpu(pos, Nx, boxsize, n0, Gmtx, Lmtx)

            # (1/2) kick
            vel += acc * dt / 2.0
            
            
            # ---- LIVE PLOT ----
            
            if plotRealTime and i % 10 == 0:

                pts = torch.stack(
                    (pos[:5000].flatten(), vel[:5000].flatten()),
                    dim=1
                ).detach().cpu().numpy()

                scatter.set_offsets(pts)

                ax.set_title(f"Two Stream Instability (t = {i*dt:.2f})")

                plt.pause(5)
            

        endtime = time.time()
        TimeTaken = np.append(TimeTaken, endtime-starttime)

    print(TimeTaken)
    return 0
    


if __name__ == "__main__":
    main()
