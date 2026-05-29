"""
Solves a non-linear Poisson equation using Firedrake.
Compares Picard fixed-point iterations with Newton-Raphson methods.
"""
from firedrake import *
import matplotlib.pyplot as plt

N = 20
p = 10 # Exponent, 5 nice
F = 1 # 20 nice
u0 = 1 # 100.0 nice

f = Constant(F)
mesh = UnitSquareMesh(N, N)
V = FunctionSpace(mesh, 'CG', 1)
bcs = DirichletBC(V, Constant(0), "on_boundary")

u = Function(V)
v = TestFunction(V)


params = {"snes_converged_reason": None, "ksp_type": "preonly", "pc_type": "lu"}
def solver(formulation):

    if formulation == "newton": 
        u.interpolate(Constant(u0)) # Restart from the previously computed solution...
        F = dot(grad(u), grad(v)) * dx + (u**p - f) * v * dx
        solve(F == 0, u, bcs=bcs, solver_parameters=params)

    else: # picard

        u.interpolate(Constant(u0))
        w = Function(V)
        w.interpolate(Constant(u0))
        F = dot(grad(u), grad(v)) * dx + (pow(u,p) - f) * v * dx
        Fw = dot(grad(u), grad(v)) * dx + (pow(w,p-1)*u - f) * v * dx

        res_vec = assemble(F, bcs=bcs)
        err = sqrt(res_vec.vector().inner(res_vec.vector()))
        it = 0 
        err0 = err
        tol = 1e-6
        maxit = 100
        print(f"It {it}, err={err} ") 

        while err / err0 > tol and it < maxit:
            solve( Fw == 0, u, bcs=bcs, solver_parameters=params )
            assemble(F, bcs=bcs, tensor=res_vec)
            err = sqrt(res_vec.vector().inner(res_vec.vector()))
            w.assign(u)
            print(f"It {it}, err={err}") 
            it += 1

print("==================== Newton")
#solver("newton")
print("==================== Fixed-point")
solver("picard")
File("output/nl.pvd").write(u)




