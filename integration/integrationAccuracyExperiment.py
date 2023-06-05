#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thoughts about integration (for error evalutaion of numerical solutions).

Created on Mon Jun 5 13:21:36 2023

@author: maxvondanwitz
"""

import random

import numpy as np
from numpy import linalg as la

from scipy.integrate import nquad

from sympy import symbols, integrate, sqrt
from sympy.plotting import plot3d
from sympy import lambdify


def monteCarloIntegration(nn, runs):
    # Performs number of `runs` Monte Carlo integrations of u_lam and
    # reports the outcomes, mean and standard deviation.
    res = np.zeros(runs)
    for run in range(runs):
        # Evaluating the function u at nn random points per dimension.
        vals = [
            u_lam(random.random(), random.random())
            for i in range(nn)
            for j in range(nn)
        ]

        res[run] = la.norm(np.array(vals, dtype=float)) / nn

    return res, np.mean(res), np.std(res)


def leftPointIntegration(nn):
    # Evaluating the function u at nn equidistant points per dimension results
    # in a left point integration.
    vals = [
        # u.subs([(x, i / nn), (y, j / nn)])
        u_lam(i / nn, j / nn)
        for i in range(nn)
        for j in range(nn)
    ]
    return la.norm(np.array(vals, dtype=float)) / nn


# As computational domain, we consider a unit square and need two coordinates
# to parametrize it.
x, y = symbols("x y")

# This is the function we want to investigate. It is in fact a solution to a
# Poisson problem taken from ngsolve
# https://docu.ngsolve.org/latest/whetting_the_appetite/poisson.html
# Still, let's pretend it is the error of our numerical scheme.
u = 16 * x * (1 - x) * y * (1 - y)

# u is actually pretty well-behaved..., it is just an example.
plot3d(u, (x, 0, 1), (y, 0, 1))


# We lambdify it to allow for fast evaluation.
# https://docs.sympy.org/latest/modules/utilities/lambdify.html
# Lambdify converts the sympy expression u, which is suitable for sybolic
# manipulation, e.g., symbolic integration, to an object of a numerical library,
# e.g., numpy function, which is suitable for fast numerical evaluation.
# For the Monte Carlo integration below, it does make a difference.
u_lam = lambdify([x, y], u)


# A good way to quantify the error over the computational is the L2 norm, based
# on an domains integral. (H1 might be more appropriate in some cases,
# but let's keep it simple.)
### Option 1
L2 = sqrt(integrate(u * u, (x, 0, 1), (y, 0, 1)))

# For our closed-form expression u, the analytical integral returns a rational
# number and all is good.
print("Analytical value of L2-norm:", L2)


# The analytical value L2 allows us to evaluate the accuracy of a numerical
# approximation l2.
def relativeError(L2, l2):
    return abs(L2 - l2) / L2


# In most cases, we don't have a closed-form expression for the error, so how
# about sampling.

# Why not random sampling? Monte Carlo
print()
print("Monte Carlo")


runs = 10
for nn in [10, 100, 1000]:
    ### Option 2a
    l2 = monteCarloIntegration(nn, runs)
    print(
        "l2-sum approximation based on",
        nn * nn,
        "random sampling points and",
        runs,
        "runs.",
    )
    print(l2)
    print("Relative error:", relativeError(L2, l2[1]))
# We don't see this in every run, but it looks like O(N^(-1/2)),
# which is also what theory tells us.

# Ok, that is why we do not use random sampling.
print()
print("Equidistant Sampling")
for nn in [10, 100, 1000]:
    ### Option 2b
    l2 = leftPointIntegration(nn)

    print(
        "l2-sum approximation",
        l2,
        " based on",
        nn * nn,
        "equidistant sampling points.",
    )

    print("Relative error:", relativeError(L2, l2))
# Here, we see O(N^-2). Much better. How to obain (approximately) equidistant
# samples for more complex geometries? Meshing.

### Option 3: Mesh-based integration
# works quite well, e.g., using ngsolve.
# Installing netgen from https://ngsolve.org and running `netgen meshBasedInteg.py`
# produces:
l2 = 0.5333333327319278
# `netgen meshBasedInteg.py` also visualizes the coarse mesh.
print()
print("Mash Based Integration")
print("L2 norm with mesh based integral:", l2)
print("Relative error:", relativeError(L2, l2))

### Option 4: Adaptive numerical integration
print()
print("Adaptive Numerical Integration")


# The integrand can be basically anything that takes coordinates as an input
# and returns a real number, e.g., a trained PINN can be evaluated with the
# following wrapper.
# def integrand2(x, t):
#     Xint = np.array([[x, t]])
#     return model.predict(Xint)[0][0]
def integrand(x, y):
    return u_lam(x, y) * u_lam(x, y)


# If our FEM solver does not allow us to interpolate, fit griddata
# https://docs.scipy.org/doc/scipy/tutorial/interpolate/ND_unstructured.html


# def bounds_t(x): # Allows us to integrate over a quarter disk.
#     return [0.0, np.sqrt(1 - x**2)]

value, errorbound = nquad(integrand, [[0.0, 1.0], [0.0, 1.0]])
l2 = sqrt(value)
print("L2-norm:", l2)
print("Relative error:", relativeError(L2, l2))
print("Upper bound for integration error reported by nquad()", errorbound, ".")
