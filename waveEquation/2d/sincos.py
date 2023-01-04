#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 13:21:36 2022

@author: maxvondanwitz
"""
import numpy as np
from sympy import sin, cos, pi
from sympy import symbols, diff
from sympy.plotting import plot3d

# The analyitcal solution depends on spatial and temporal coordinates and the
# speed of sound c.
x, y, t, c = symbols('x y t c')

# Speed of sound
c = 0.5

# Displacement solution
u = sin(pi*x) * sin(pi*y) * (sin(pi*t)+cos(pi*t))

# Momentum solution
p = diff(u, t)/(c*c)

# Next, we verify that u fulfills the homogeneous wave equation
d2udx2 = diff(u, x, 2)

d2udy2 = diff(u, y, 2)

d2udt2 = diff(u, t, 2)

f = d2udt2 - c * (d2udx2 + d2udy2)

assert f == 0

# Initial conditions
icu = u.subs(t, 0.0)
icp = p.subs(t, 0.0)

# Check homogeneous boundary conditions for both variables
for var in [u, p]:

    bc1 = var.subs(x, -1)
    assert bc1 == 0

    bc2 = var.subs(y, -1)
    assert bc2 == 0

    bc3 = var.subs(x, 1)
    assert bc3 == 0

    bc4 = var.subs(y, 1)
    assert bc4 == 0

doPlot = False
if doPlot:  # Plotting 8 snapshots of the periodic solution (T=2).
    for ti in np.arange(0.0, 2.25, 0.25):
        plot3d(u.subs(t, ti), (x, -1, 1), (y, -1, 1), title='t = ' + str(ti))
