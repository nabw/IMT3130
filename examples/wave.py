"""
Simulates the 2D acoustic wave equation comparing explicit, implicit, and symplectic time-stepping schemes in Firedrake.
"""
from firedrake import *

EXPLICIT=0
IMPLICIT=1
SYMPLECTICU=2
SYMPLECTICV=3


N = 50
P = 2
mesh = UnitSquareMesh(N, N)
time_scheme = SYMPLECTICV

mu = Constant(1e-1, domain=mesh)
dt = 1e-2
tf = 20
save_every = 10

V = FunctionSpace(mesh, "CG", P)
W = V * V
h = Function(V, name="h")
du = TrialFunction(V)
dv = TestFunction(V)
v, u = TrialFunctions(W)
vt, ut = TestFunctions(W)
sol = Function(W)
un = Function(V, name="u")
vn = Function(V)

idt = Constant(1/dt)
H = 0.5 * mu * (grad(sol.sub(1)))**2 + 0.5 * sol.sub(0)**2
if time_scheme == EXPLICIT:
    us = un
    vs = vn
elif time_scheme == IMPLICIT:
    us = u
    vs = v
elif time_scheme == SYMPLECTICU:
    us = u
    vs = vn
else: # SYMPLECTICV
    us = un
    vs = v

a = idt * (v - vn) * vt * dx + idt * (u - un) * ut * dx - \
    vs * ut * dx + mu * inner(grad(us), grad(vt)) * dx

X = mesh.coordinates
C = Constant((0.5, 0.5))
XC = X-C
r = sqrt(dot(XC, XC))
u0 = conditional(lt(r, 0.3), Constant(1), 0)
solve(Constant(1) * du*dv*dx + Constant(0.5) * CellDiameter(mesh)**2 * dot(grad(du), grad(dv))*dx==u0*dv*dx, un, bcs=DirichletBC(V, u0, "on_boundary"))
#un.interpolate(u0)
sol.sub(1).assign(un)
sol.sub(0).interpolate(Constant(0))

problem = LinearVariationalProblem(lhs(a), rhs(a), sol, constant_jacobian=True)

params = {"ksp_type": "gmres",
          "mat_type": "nest",
          "ksp_norm_type": "unpreconditioned",
          "ksp_atol": 0.0,
          "ksp_rtol": 1e-6, 
          "pc_type": "fieldsplit",
          "pc_fieldsplit_type": "multiplicative",
          "fieldsplit_0_ksp_type": "preonly",
          "fieldsplit_0_pc_type": "jacobi",
          "fieldsplit_1_ksp_type": "preonly",
          "fieldsplit_1_pc_type": "hypre"
          }

solver = LinearVariationalSolver(problem, solver_parameters=params)

outfile = File("output/wave.pvd")
t = 0
h.interpolate(H)
outfile.write(un, h, t=t)
i = 0
energies = []
while t < tf:
    if i % 10 == 0: print("Solving t={:.2f}".format(t))
    solver.solve()
    vn.assign(sol.sub(0))
    un.assign(sol.sub(1))
    t += dt
    if i % save_every == 0:
        h.interpolate(H)
        outfile.write(un, h, t=t)
        energies.append(assemble(H * dx))
    i += 1
import matplotlib.pyplot as plt
plt.plot(energies)
plt.show()
