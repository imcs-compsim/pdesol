#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:13:38 2023

@author: maxvondanwitz
"""

# Required Sympy functionality
from sympy import Symbol, diff
from sympy import pi, sin, Piecewise
from sympy.plotting import plot, plot3d
# Local dependency
from hermite import quinticHermiteInterpolation
### from plotOwn3d import plotOwn3d # see below

x = Symbol('x', real=True)
t = Symbol('t', real=True)

# Switch from off to on takes places over an interval dx
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
u = sin(2*pi*x) * s

# Speed of sound
c = 1.0

# Momentum solution
p = diff(u, t)/(c*c)

# Inserting into the wave equation, we obtain the piecewise defined source term:
f = diff(u, t, 2) - c * diff(u, x, 2) 
print(f.expand())

# At least, with this complicated source term, we have nice and simple 
# homogeneous initial and boundary conditions for both variables
for var in [u, p]:
    
    ic = var.subs(t, 0.0)
    assert ic == 0

    bc1 = var.subs(x, 0)
    assert bc1 == 0

    bc3 = var.subs(x, 1)
    assert bc3 == 0

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
    
    # Solution field
    ### For these plots to work, I had to introduce a hotfix 
    ### in sympy/plotting/experimental_lambdify.py to prevent conversion to real.
    ### If plot3d of piecewise function still does not work, we can use
    ### plotOwn3d(u, x, 0, 1, t, 0, 1, 201)
    plot3d(u, (x, 0, 1), (t, 0, 1), title='displacement solution')
    plot3d(p, (x, 0, 1), (t, 0, 1),  title='momentum solution', nb_of_points_y = 201)
     
    plot(p.subs(x, 0.25), (t,0,1))
    



