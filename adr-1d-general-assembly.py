import numpy as np
from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt

# 1. Problem Parameters
L, n_el = 1.0, 20
n_nodes = n_el + 1
x_nodes = np.linspace(0, L, n_nodes)
#x_nodes = x_nodes**(4.5)  # non-uniform meshes!
h = L / n_el
# Now we can use functions!
nu_func = lambda x: 0.01 + 0.1 * x**2
b_func  = lambda x: 1.0 + np.sin(np.pi * x)
c_func  = lambda x: 0.5 * np.ones_like(x)
f_func  = lambda x: 10 * np.exp(-(x-0.5)**2 / 0.02)

# Local functions and their derivatives
N = lambda xi: np.array([0.5 * (1 - xi), 0.5 * (1 + xi)])
dN_dxi = np.array([-0.5, 0.5])

# Gauss Quadrature (2-point rule, exact for polynomials up to degree 3)
q_pts = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
q_wts = np.array([1.0, 1.0])


# 2. Mesh and Connectivity (IEN)
# IEN[e, local_node] -> global_node. Starting from left.
IEN = np.array([[e, e + 1] for e in range(n_el)])

# 3. Equation Numbering (ID)
# u(0) = u(L) = 0. Dirichlet nodes are marked as -1
N_bc = 2
ID = np.arange(n_nodes) - (N_bc-1)
ID[0] = ID[-1] = -1  # Boundary nodes
n_eq = np.max(ID) + (N_bc-1)   # Total unknowns

K = np.zeros((n_eq, n_eq))
F = np.zeros(n_eq)

for e in range(n_el):
    coords = x_nodes[IEN[e]]

    # Local element contribution
    k_e = np.zeros((2, 2)) # N_local x N_local
    f_e = np.zeros(2)

    # Jacobian of the mapping (constant for P1 elements: dx/dxi = h/2)
    dx_dxi = (coords[1] - coords[0]) / 2.0
    detJ = dx_dxi
    invJ = 1.0 / dx_dxi

    # Sum each quadrature contribution independently
    for q in range(len(q_pts)):
        xi = q_pts[q]
        w = q_wts[q]

        # Map quadrature point to physical space
        xq = np.dot(N(xi), coords)

        # Evaluate coefficients at quadrature point
        nu_q = nu_func(xq)
        b_q  = b_func(xq)
        c_q  = c_func(xq)
        f_q  = f_func(xq)

        # Shape functions and gradients in physical space
        phi = N(xi)
        grad_phi = invJ * dN_dxi

        # Integrate local stiffness and force
        for a in range(2):
            # Force vector
            f_e[a] += f_q * phi[a] * detJ * w

            for b in range(2):
                # Diffusion + Advection + Reaction
                k_e[a, b] += (nu_q * grad_phi[a] * grad_phi[b] +
                              b_q * grad_phi[b] * phi[a] +
                              c_q * phi[a] * phi[b]) * detJ * w

    # Global Scatter
    for a in range(2):
        i = ID[IEN[e,a]]
        print(i)
        if i != -1:
            F[i] += f_e[a]
            for b in range(2):
                j = ID[IEN[e,b]]
                if j != -1:
                    K[i, j] += k_e[a, b]

# 6. Solve and Reconstruct
u_free = np.linalg.solve(K, F)
u = np.zeros(n_nodes)
u[ID != -1] = u_free

# 7. Plot
plt.plot(x_nodes, u, 'r', marker='o', label="Assembled Solution")
plt.xlabel("x"); plt.ylabel("u(x)"); plt.grid(True); plt.legend()
plt.show()
