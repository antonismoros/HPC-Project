import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import time
"""
Create Your Own Plasma PIC Simulation (With Python)
Philip Mocz (2020) Princeton University, @PMocz

Simulate the 1D Two-Stream Instability
Code calculates the motions of electron under the Poisson-Maxwell equation
using the Particle-In-Cell (PIC) method

"""

#@profile
def getAcc(pos, Nx, boxsize, n0, Gmtx, Lmtx):
    """
    Calculate the acceleration on each particle due to electric field
        pos      is an Nx1 matrix of particle positions
        Nx       is the number of mesh cells
        boxsize  is the domain [0,boxsize]
        n0       is the electron number density
        Gmtx     is an Nx x Nx matrix for calculating the gradient on the grid
        Lmtx     is an Nx x Nx matrix for calculating the laplacian on the grid
        a        is an Nx1 matrix of accelerations
    """
    # Calculate Electron Number Density on the Mesh by
    # placing particles into the 2 nearest bins (j & j+1, with proper weights)
    # and normalizing
    N = pos.shape[0]
    dx = boxsize / Nx
    frac = (pos / dx)
    j = frac.astype(int)
    weight_jp1 = frac - j
    weight_j = 1.0 - weight_jp1
    jp1 = np.mod(1+j, Nx)  # periodic BC
    n = np.bincount(j[:, 0], weights=weight_j[:, 0], minlength=Nx)
    n += np.bincount(jp1[:, 0], weights=weight_jp1[:, 0], minlength=Nx)
    n *= n0 * boxsize / N / dx

    # Solve Poisson's Equation: laplacian(phi) = n-n0
    phi_grid = spsolve(Lmtx, n - n0, permc_spec="MMD_AT_PLUS_A")

    # Apply Derivative to get the Electric field
    E_grid = -Gmtx @ phi_grid

    # Interpolate grid value onto particle locations
    E = weight_j * E_grid[j]        # Creates first array
    E += weight_jp1 * E_grid[jp1]

    a = -E

    return a


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

        pos = np.random.rand(N, 1) * boxsize
        vel = vth * np.random.randn(N, 1) + vb
        Nh = int(N / 2)
        vel[Nh:] *= -1
        # add perturbation
        vel *= 1 + A * np.sin(2 * np.pi * pos / boxsize)

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
        acc = getAcc(pos, Nx, boxsize, n0, Gmtx, Lmtx)

        # number of timesteps
        Nt = int(np.ceil(tEnd / dt))

        # prep figure
        """
        fig, ax = plt.subplots(figsize=(6,4))
        scatter = ax.scatter(pos[:5000], vel[:5000], s=1)

        ax.set_xlim(0, boxsize)
        ax.set_ylim(-6,6)
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.set_title("Two Stream Instability Phase Space")

        plt.ion()
        plt.show()
        """
        # Simulation Main Loop
        for i in range(Nt):
            # (1/2) kick
            vel += acc * dt / 2.0

            # drift (and apply periodic boundary conditions)
            pos += vel * dt
            pos = np.mod(pos, boxsize)

            # update accelerations
            acc = getAcc(pos, Nx, boxsize, n0, Gmtx, Lmtx)

            # (1/2) kick
            vel += acc * dt / 2.0

            # ---- LIVE PLOT ----
            """if plotRealTime:

                scatter.set_offsets(
                    np.column_stack((pos[:5000], vel[:5000]))
                )

                ax.set_title(f"Two Stream Instability (t = {i*dt:.2f})")

                plt.pause(0.001)"""
        endtime = time.time()
        TimeTaken = np.append(TimeTaken, endtime-starttime)

    print(TimeTaken)
    return 0
    


if __name__ == "__main__":
    main()
