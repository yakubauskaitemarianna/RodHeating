from sympy import *
from mpmath import plot

# [x0, x1] - function implements on the interval
# q - Polinomials maximum degree for choosing Galerkin method basis
# 0th degree basis function is fixed to the initial value
# ode - ordinary differential equation
# x - function argument
# u0 - initial value

# Assume that the solution can be approximated as a linear combination of the basis functions

def galerkin(ode, x, x0, x1, u0, q):
    basis = [x**k for k in range(q+1)]
    # Coefficients for the basis monomials
    xi = [Symbol("xi_%i" % k) for k in range(q+1)]
    # Solution function ansatz
    u = u0 + sum(xi[k]*basis[k] for k in range(1,q+1))
    # Form system of linear equations
    equations = [integrate(ode(u)*basis[k], (x, x0, x1)) \
        for k in range(1,q+1)]
    coeffs = solve(equations, xi[1:])
    return u.subs(coeffs)

if __name__ == "__main__":
    x = Symbol('x')
    ode = lambda u: u.diff(x) - u
    for q in range(1,6):
        pprint(galerkin(ode, x, 0, 1, 1, q))
    u1 = Lambda(x, galerkin(ode, x, 0, 1, 1, 1))
    u2 = Lambda(x, galerkin(ode, x, 0, 1, 1, 2))
    u3 = Lambda(x, galerkin(ode, x, 0, 1, 1, 3))
    plot([exp, u1, u2, u3], [0, 2])
