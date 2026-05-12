from firedrake import *
from tqdm import tqdm
import numpy as np

# ==========================================================
# 1. Mesh and Function Spaces
# ==========================================================
# A 2D rubber band of length 5.0 and height 1.0
length, height = 5.0, 1.0
mesh = RectangleMesh(50, 10, length, height)
saveEvery = 5

# Mixed Space: P2 (quadratic) for displacement, P1 (linear) for temperature
# This Taylor-Hood type pairing avoids volumetric locking in near-incompressible media
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V * Q

# ==========================================================
# 2. Material Parameters (Rubber-like)
# ==========================================================
rho0 = Constant(1000.0)      # Density (kg/m^3)
mu = Constant(1.0e6)         # Shear modulus (Pa)
lmbda = Constant(5.0e7)      # Lamé parameter (Pa) - high for near-incompressibility
K = lmbda + (2.0/3.0)*mu     # Bulk modulus (Pa)
alpha = Constant(1.0e-4)     # Thermal expansion coefficient (1/K)
cv = Constant(1500.0)        # Specific heat capacity (J/(kg K))
kappa = Constant(0.2)        # Thermal conductivity (W/(m K))
theta0 = Constant(293.15)    # Reference temperature (20 C)

# ==========================================================
# 3. Functions and Kinematics
# ==========================================================
w_np1 = Function(W, name="State_np1")  # State at t_{n+1}
w_n = Function(W, name="State_n")      # State at t_n
w_nm1 = Function(W, name="State_nm1")  # State at t_{n-1}

u_np1, theta_np1 = split(w_np1)
u_n, theta_n = split(w_n)
u_nm1, theta_nm1 = split(w_nm1)

v, q = TestFunctions(W)

# Initialize temperatures to room temperature
w_np1.sub(1).interpolate(theta0)
w_n.sub(1).interpolate(theta0)
w_nm1.sub(1).interpolate(theta0)

# Kinematic quantities
d = mesh.geometric_dimension
I = Identity(d)

def get_F(u):
    return I + grad(u)

F_np1 = get_F(u_np1)
J_np1 = det(F_np1)
FinvT = inv(F_np1).T

# ==========================================================
# 4. Constitutive Models (From Slides)
# ==========================================================
# First Piola-Kirchhoff Stress Tensor
P = mu * F_np1 + (lmbda * ln(J_np1) - mu - 3 * alpha * K * (theta_np1 - theta0)) * FinvT

# Structural Heating (Gough-Joule effect coupling)
dt = Constant(2e-3)
v_vel = (u_np1 - u_n) / dt  # Velocity approximation
F_dot = grad(v_vel)         # Rate of deformation gradient
H_mech = 3 * alpha * K * theta_np1 * tr(F_dot * inv(F_np1))

# ==========================================================
# 5. Weak Formulation (Semi-Discrete)
# ==========================================================
# 5a. Momentum Equation
a_np1 = (u_np1 - 2*u_n + u_nm1) / dt**2
Res_Mech = (rho0 * inner(a_np1, v) + inner(P, grad(v))) * dx

# 5b. Energy Equation
theta_dot = (theta_np1 - theta_n) / dt
Res_Therm = (rho0 * cv * theta_dot * q + kappa * inner(grad(theta_np1), grad(q)) - H_mech * q) * dx

# Total Residual
F_total = Res_Mech + Res_Therm

# ==========================================================
# 6. Boundary Conditions
# ==========================================================
# Left boundary (x=0, tag 1): Fixed
bc_left = DirichletBC(W.sub(0), Constant((0.0, 0.0)), 1)

# Right boundary (x=L, tag 2): Applied stretching
# We will dynamically update the 'pull' value in the time loop
pull = Constant(0.0)
bc_right = DirichletBC(W.sub(0).sub(0), pull, 2)
bc_right_y = DirichletBC(W.sub(0).sub(1), Constant(0.0), 2)

bcs = [bc_left, bc_right, bc_right_y]
# Note: No thermal BCs implies Neumann boundary q.n = 0 (Adiabatic)

# ==========================================================
# 7. Time Stepping and ParaView Export
# ==========================================================
outfile = VTKFile("output/rubber_stretch.pvd")

# Extract sub-functions for saving to ParaView
u_out, theta_out = w_n.subfunctions
u_out.rename("Displacement")
theta_out.rename("Temperature")
outfile.write(u_out, theta_out, time=0.0)

t = 0.0
T_end = 1.0
stretch_rate = 2.0  # m/s

i = 0
print("Starting time loop...")

N = int(T_end/dt(0))
for i in tqdm(range(N)):
#while t <= T_end:
    t += float(dt)
    
    # Update boundary condition (stretch the rubber)
    pull.assign(stretch_rate * t)
    
    # Solve the monolithic nonlinear system
    # Firedrake uses Newton-Raphson by default for nonlinear problems
    solve(F_total == 0, w_np1, bcs=bcs, solver_parameters={
        "snes_type": "newtonls",
        #"snes_monitor": None,
        "ksp_type": "preonly",
        "pc_type": "lu"
    })
    
    # Advance time variables
    w_nm1.assign(w_n)
    w_n.assign(w_np1)
    
    # Export to ParaView
    u_out, theta_out = w_n.subfunctions
    u_out.rename("Displacement")
    theta_out.rename("Temperature")
    if i % saveEvery == 0: outfile.write(u_out, theta_out, time=t)
    
    #print(f"Time: {t:.2f} s | Max Temp: {theta_out.dat.data.max():.2f} K")
    i+= 1

print("Simulation complete. Open 'rubber_stretch.pvd' in ParaView.")
