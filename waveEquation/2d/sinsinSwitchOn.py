#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:13:38 2023

@author: maxvondanwitz
"""

# Product of two Gaussians
from sympy import Symbol, diff
from sympy import pi, sin, Piecewise
from sympy.plotting import plot, plot3d

def quinticHermiteInterpolation(x, x_0, dx, fx0, fx1, f1x0, f1x1, f2x0, f2x1):
    # Quintic Hermite Interpolation (https://en.wikipedia.org/wiki/Hermite_interpolation)
    # Terms of 0th to 5th order...
    t0 = fx0 
    t1 = f1x0 * (x - x_0) 
    t2 = 1/2 * f2x0 * (x - x_0)**2
    t3 = ((fx1 - fx0 - f1x0 * (dx) - 1/2 * f2x0 * (dx)**2) 
       / (dx)**3) * (x - x_0)**3
    t4 = ((3 * fx0 - 3 * fx1 + 2 * (f1x0 + f1x1) * (dx) + 1/2 * f2x0 * (dx)**2) 
       / (dx)**4) * (x - x_0)**3 * (x - x_1)
    t5 = ((6 * fx1 - 6 * fx0 - 3 * (f1x0 + f1x1) * (dx) + 1/2 * (f2x1 - f2x0) * (dx)**2) 
       / (dx)**5) * (x - x_0)**3 * (x - x_1)**2

    # ... added up to the qunitic Hermite interpolation
    return t0 + t1 + t2 + t3 + t4 + t5


x = Symbol('x', real=True)
y = Symbol('y', real=True)
t = Symbol('t', real=True)

# Hesch et al. say in the text that they used an interval of 10-3 for the 
# switch-on process. However, Fig. 1 in their paper suggests that 10-2 is used.
# dx = 1/1000.0

dx = 1/100.0
# Two positions where we have data
x_0 = 1/2.0
x_1 = x_0 + dx

# Function values at x0, x1
fx0 = 0
fx1 = 1

# First derivatives at x0, x1
f1x0 = 0
f1x1 = 0

# Second derivatives at x0, x1
f2x0 = 0
f2x1 = 0

qhi = quinticHermiteInterpolation(t, x_0, dx, fx0, fx1, f1x0, f1x1, f2x0, f2x1)

# Construct the switch-on process s.
s = Piecewise((0, t <= x_0), (1, t >= x_1), (qhi, True))

# Displacement solution
u = sin(2*pi*x) * sin(2*pi*y) * s

# Speed of sound
c = 1.0

# Momentum solution
p = diff(u, t)/(c*c)

# Inserting into the wave equation, we obtain the piecewise defined source term:
f = diff(u, t, 2) - c * (diff(u, x, 2) + diff(u, y, 2))
print(f.expand())

# At least, with this complicated source term, we have nice and simple 
# homogeneous initial and boundary conditions for both variables
for var in [u, p]:
    
    ic = var.subs(t, 0.0)
    assert ic == 0

    bc1 = var.subs(x, 0)
    assert bc1 == 0

    bc2 = var.subs(y, 0)
    assert bc2 == 0

    bc3 = var.subs(x, 1)
    assert bc3 == 0

    bc4 = var.subs(y, 1)
    assert bc4 == 0

# Plotting, so we can visually check that the returned functions fulfill our
# requirements. 
doPlot = True
if doPlot: 

    # Quintic Hermite Interpolation
    plot(qhi, (t, x_0, x_1))
    dqhi = diff(qhi,t)
    plot(dqhi, (t, x_0, x_1))
    ddqhi = diff(dqhi,t)
    plot(ddqhi, (t, x_0, x_1))
    
    # Switch-on Process
    plot(s, (t, 0, 1))
    plot(s, (t, 0.49, 0.52))
    
    # Plotting 2 snapshots of the displacement solution.
    # plot3d(u.subs(t, 0.0), (x, 0, 1), (y, 0, 1), title='t = 0.0')
    plot3d(u.subs(t, 0.505), (x, 0, 1), (y, 0, 1), title='Displacement solution at t = 0.505')
    plot3d(u.subs(t, 1.0), (x, 0, 1), (y, 0, 1), title='Displacement solution at t = 1.0')
    
    # And details of the momentum solution
    plot3d(p.subs(y, 0.25),(x, 0, 1), (t, 0, 1), title='Momentum solution at y = 0.25' , nb_of_points_y = 201)
    ptime = p.subs([(x, 0.25), (y, 0.25)])
    plot(ptime, (t, 0, 1))
    