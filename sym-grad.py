from firedrake import *
import matplotlib.pyplot as plt

def main():
    # 1. Mesh definition (2D Channel)
    # Using 40x8 resolution for clear vector spacing in quiver plots
    L, H = 2.0, 1.0
    mesh = RectangleMesh(8, 4, L, H)

    # 2. Finite Element Space (Vectors, Continuous Galerkin, degree 2)
    V = VectorFunctionSpace(mesh, "CG", 2)

    u = TrialFunction(V)
    v = TestFunction(V)

    # 3. Boundary Conditions
    x, y = SpatialCoordinate(mesh)
    
    # Parabolic profile for the inlet
    inflow_profile = as_vector([4.0 * y * (H - y) / (H**2), 0.0])
    
    bcs = [
        DirichletBC(V, inflow_profile, 1),           # Left boundary (inflow)
        DirichletBC(V, as_vector([0.0, 0.0]), (3, 4)) # Top and bottom solid walls
    ]

    f = Constant((0.0, 0.0))

    # =====================================================================
    # CASE 1: Standard Gradient (Uncoupled Vector Laplacian)
    # =====================================================================
    a_std = inner(grad(u), grad(v)) * dx
    L_std = inner(f, v) * dx

    u_std = Function(V, name="Standard_Gradient")
    print("Solving standard gradient problem...")
    solve(a_std == L_std, u_std, bcs=bcs)

    # =====================================================================
    # CASE 2: Symmetric Gradient (Coupled, Stokes-like)
    # =====================================================================
    def epsilon(w):
        """Returns the symmetric part of the gradient (strain-rate tensor)"""
        return sym(grad(w))

    a_sym = 2.0 * inner(epsilon(u), epsilon(v)) * dx
    L_sym = inner(f, v) * dx

    u_sym = Function(V, name="Symmetric_Gradient")
    print("Solving symmetric gradient problem...")
    solve(a_sym == L_sym, u_sym, bcs=bcs)

    # =====================================================================
    # 4. Field Normalization for Directional Visualization
    # =====================================================================
    print("Projecting normalized vector fields...")
    
    # We add a small tolerance (1e-12) to avoid division by zero at the no-slip walls
    mag_std = sqrt(inner(u_std, u_std) + 1e-12)
    u_std_norm = project(u_std / mag_std, V)

    mag_sym = sqrt(inner(u_sym, u_sym) + 1e-12)
    u_sym_norm = project(u_sym / mag_sym, V)

    # =====================================================================
    # 5. Direct Visualization with Matplotlib (2x2 Grid)
    # =====================================================================
    print("Generating matplotlib figures...")
    
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))

    # --- Top Left: Standard Gradient (Velocity Magnitude) ---
    axes[0, 0].set_title("Standard Gradient: Velocity Field")
    q1 = quiver(u_std, axes=axes[0, 0], cmap='viridis')
    axes[0, 0].set_aspect('equal')
    fig.colorbar(q1, ax=axes[0, 0], label="Velocity Magnitude")

    # --- Top Right: Standard Gradient (Normalized Direction) ---
    axes[0, 1].set_title("Standard Gradient: Flow Direction (Normalized)")
    # Using black arrows to focus purely on the geometry/direction of flow
    q2 = quiver(u_std_norm, axes=axes[0, 1], color='black')
    axes[0, 1].set_aspect('equal')

    # --- Bottom Left: Symmetric Gradient (Velocity Magnitude) ---
    axes[1, 0].set_title("Symmetric Gradient: Velocity Field")
    q3 = quiver(u_sym, axes=axes[1, 0], cmap='viridis')
    axes[1, 0].set_aspect('equal')
    fig.colorbar(q3, ax=axes[1, 0], label="Velocity Magnitude")

    # --- Bottom Right: Symmetric Gradient (Normalized Direction) ---
    axes[1, 1].set_title("Symmetric Gradient: Flow Direction (Normalized)")
    q4 = quiver(u_sym_norm, axes=axes[1, 1], color='black')
    axes[1, 1].set_aspect('equal')

    # Ensure all plots have the correct spatial limits
    for ax in axes.flat:
        ax.set_xlim(0, L)
        ax.set_ylim(0, H)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
