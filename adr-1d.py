import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# Parameters
L = 1.0      # Domain length
N = 10       # Number of nodes
nu = 0.1    # Diffusion coefficient
c = 0.5      # Advection speed
sigma = 1.0  # Reaction rate
f = 1.0      # Source term

def solve_adr_1d(L, N, nu, c, sigma, f_val):
    """
    Solves -nu*u'' + c*u' + sigma*u = f on (0, L) with u(0)=u(L)=0
    using P1 Finite Elements on a uniform mesh.
    """
    h = L / (N - 1) # N nodes, N-1 elements
    x = np.linspace(0, L, N)
    
    # Main diagonal and off-diagonals for a tridiagonal system
    main_diag = np.ones(N)
    off_diag  = np.ones(N - 1)

    # Diffusion Matrix: (nu/h) * [-1, 2, -1]
    K_diff = (nu / h) * diags([-off_diag, 2 * main_diag, -off_diag], [-1, 0, 1]).toarray()
    # Advection Matrix: (c/2) * [-1, 0, 1]
    K_adv = (c / 2.0) * diags([-off_diag, 0 * main_diag, off_diag], [-1, 0, 1]).toarray()
    # Reaction Matrix: (sigma*h/6) * [1, 4, 1]
    K_react = (sigma * h / 6.0) * diags([off_diag, 4 * main_diag, off_diag], [-1, 0, 1]).toarray()
    A = K_diff + K_adv + K_react

    # 2. Right-Hand Side (Load Vector): f_i = integral(f * phi_i)
    # For f=1, F_i = h (sum of two triangles of area h/2)
    F = np.full(N, f_val * h)
    F[0] /= 2.0   # Boundary nodes only have half the support
    F[-1] /= 2.0

    # 3. Apply Dirichlet Boundary Conditions (u(0) = u(L) = 0)
    # We replace the first and last equations with identity constraints
    A[0, :] = 0; A[0, 0] = 1; F[0] = 0
    A[-1, :] = 0; A[-1, -1] = 1; F[-1] = 0

    # 4. Solve
    u = spsolve(A, F)
    return x, u

x, u = solve_adr_1d(L, N, nu, c, sigma, f)

# Visualization
plt.figure(figsize=(8, 5))
plt.plot(x, u, 'b-', label='FE Solution (P1)')
plt.title(f"1D ADR: $\\nu={nu}, c={c}, \\sigma={sigma}$")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid(True)
plt.legend()
plt.show()
