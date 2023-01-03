#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 13:21:36 2022

@author: maxvondanwitz
"""

from sympy import sin, pi, exp
from sympy import symbols, diff, solveset
from sympy import S
from sympy.plotting import plot3d

# The analyitcal solution depends on spatial and temporal coordinates and the
# diffusion coefficient k.
x, y, t, k = symbols('x y t k')

u = sin(pi*x) * sin(pi*y) * exp(-2*k*pi**2*t)

# Next, we verify that u fulfills the homogeneous heat equation 
d2udx2 = diff(u, x, 2)

d2udy2 = diff(u, y, 2)

dudt = diff(u, t)

f = dudt - k * (d2udx2 + d2udy2)

assert f == 0

# To construct an IBVP we check for initial and boundary conditions.
ic = u.subs(t, 0)

bc1 = u.subs(x, -1)

bc2 = u.subs(y, -1)

bc3 = u.subs(x, 1)

bc4 = u.subs(y, 1) 


# To visualize u, fix k and t.
utk = u.subs([(k, 0.1), (t, 2)])

plot3d(utk, (x, -1, 1), (y, -1, 1))

# What k do we need for the sine wave to decay to half of its initial amplitude?
eqk = u.subs([(t, 2), (x, 0.5), (y, 0.5)])

kdet = solveset(eqk - 0.5, k, domain=S.Reals)

print (kdet.evalf().args[0])

# Check an approximation of k...
utkxy = u.subs([(k, 0.01755), (t, 2), (x, 0.5), (y, 0.5)])

print (utkxy.evalf())
