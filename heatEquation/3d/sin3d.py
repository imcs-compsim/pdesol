#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 13:21:36 2022

Three-dimensional version of 2d/sinsin.py (with further comments). 

@author: maxvondanwitz
"""

from sympy import sin, pi, exp
from sympy import symbols, diff, solveset
from sympy import S
from sympy.plotting import plot3d

x, y, z, t, k = symbols('x y z t k')

u = sin(pi*x) * sin(pi*y) * sin(pi*z) * exp(-3*k*pi**2*t)

d2udx2 = diff(u, x, 2)

d2udy2 = diff(u, y, 2)

d2udz2 = diff(u, z, 2)

dudt = diff(u, t)

f = dudt - k * (d2udx2 + d2udy2 + d2udz2)

# u fulfills the homogeneous heat equation
assert f == 0

# Fixing the temperature at 7 of the 8 3d-faces of our hypercube.
ic = u.subs(t, 0)

bc1 = u.subs(x, -1)

bc2 = u.subs(y, -1)

bc3 = u.subs(z, -1)

bc4 = u.subs(x, 1)

bc5 = u.subs(y, 1)

bc6 = u.subs(z, 1)

# Our cube of side length 2 m is assumed to be of copper.
# Hence, we get a k in mm^2/s.
eqt = u.subs([(k, 0.00011647545279832275), (x, 0.5), (y, 0.5), (z, 0.5)])

# How long do we have to wait for the temperature peaks to half their amplitude?
tdet = solveset(eqt - 0.5, t, domain=S.Reals)

# 200 seconds it seems.
print(tdet.evalf().args[0])

# Checking our numerical approximations.
utkxy = u.subs([(k, 0.0001165), (t, 200), (x, 0.5), (y, 0.5), (z, 0.5)])

print(utkxy.evalf())

# Plotting the solution in two planes orthogonal to the z-axis.
plot3d(u.subs([(k, 0.0001165), (t, 200), (z, 0.5)]),
       (x, -1, 1), (y, -1, 1), title='z = 0.5')
plot3d(u.subs([(k, 0.0001165), (t, 200), (z, -0.5)]),
       (x, -1, 1), (y, -1, 1), title='z = -0.5')
