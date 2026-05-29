"""
Demonstrates PDE-constrained optimization and mesh-independence
using various Riesz representations of gradients (l2, L2, H1, H01).
"""
from firedrake import *
import matplotlib.pyplot as plt
import scipy.sparse as sp

alpha = 1e-1
rtol = 1e-8
maxit = 1000
verbose = False

# Poisson 
def solvePoisson(N, grad_type, alpha=0.1, verbose=False):
    mesh = UnitSquareMesh(N, N)
    V = FunctionSpace(mesh, 'CG', 1)

    f = Constant(1.0)
    u = TrialFunction(V)
    v = TestFunction(V)

    bc = DirichletBC(V, Constant(0), "on_boundary")

    sol = Function(V)
    grad_sol = Function(V)

    err=1.0
    err_vec = assemble( dot(grad(sol), grad(v)) * dx - f * v * dx, bcs=bc)
    with err_vec.dat.vec_ro as vv:
        err0 = vv.norm()

    it = 0
    if verbose: 
        print(f"It {it:4}, error={err:2.2e}")
    while err > rtol and it < maxit:

        if grad_type == "L2":
            inner = lambda _u: _u*v*dx
        elif grad_type == "H01":
            inner = lambda _u: dot(grad(_u), grad(v)) * dx
        elif grad_type == "H1":
            inner = lambda _u: (_u*v + dot(grad(_u), grad(v)) ) * dx
        else: # l2
            L = dot(grad(sol), grad(v)) * dx - f * v * dx

            # Extract a sparse python matrix
            res = assemble(L, bcs=bc)
            sol.vector().axpy(-alpha, res)
           
        if grad_type != "l2": 
            a = inner(u)
            L = dot(grad(sol), grad(v)) * dx - f * v * dx

            solve(a==L, grad_sol, bcs=bc)
            sol.vector().axpy(-alpha, grad_sol)

        err_vec = assemble( dot(grad(sol), grad(v)) * dx - f * v * dx, bcs=bc)
        with err_vec.dat.vec_ro as vv:
            err = vv.norm()
        it += 1
        if verbose: 
            print(f"It {it:4}, error={err:2.4e}")
        if err > 1e20: return 0.0
    return it
    
its_l2 = []
its_L2 = []
its_H1 = []
its_H01 = []
Ns = [2,5,10,20,40,80]
for N in Ns: 
    print("==========================")
    print("Solving for N =", N)
    print("==========================")
    it_l2 = solvePoisson(N, "l2", alpha=alpha, verbose=verbose)
    it_L2 = solvePoisson(N, "L2", alpha=alpha, verbose=verbose)
    it_H1 = solvePoisson(N, "H1", alpha=alpha, verbose=verbose)
    it_H01 = solvePoisson(N, "H01", alpha=alpha, verbose=verbose)
    its_l2.append(it_l2)
    its_L2.append(it_L2)
    its_H1.append(it_H1)
    its_H01.append(it_H01)
   
print("l2:", its_l2)
print("L2:", its_L2)
print("H1:", its_H1)
print("H01:", its_H01)
plt.semilogx(Ns, its_l2, label="l2")
plt.semilogx(Ns, its_L2, label="L2")
plt.semilogx(Ns, its_H1, label="H1")
plt.semilogx(Ns, its_H01, label="H01")
plt.legend()
plt.show()
