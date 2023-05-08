#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:21:36 2023

@author: maxvondanwitz

Advection-diffusion model problem used in https://doi.org/10.1002/nme.7241

Time-continuous and time-discontinuous space-time finite elements for
advection-diffusion problems
"""

from sympy import sin, pi, exp, sqrt
from sympy import symbols, diff, integrate
from sympy.plotting import plot3d

# The analyitcal solution depends on spatial and temporal coordinate as well as
# the advection velocity a and the diffusion coefficient k.
x, t = symbols("x t")
a, k = symbols("a k")

u = -sin(pi * (x - a * t)) * exp(-k * pi**2 * t)

# Next, we verify that u fulfills the homogeneous acvection-diffusion equation.
dudx = diff(u, x)

d2udx2 = diff(u, x, 2)

dudt = diff(u, t)

f = dudt + a * dudx - k * d2udx2

assert f.simplify() == 0

# To construct an IBVP we check for initial and boundary conditions.
ic = u.subs(t, 0)

bc1 = u.subs(x, -1)

bc3 = u.subs(x, 1)

# To visualize u, fix a and k.
uak = u.subs([(k, 0.1), (a, 1)])

plot3d(uak, (x, -1, 1), (t, 0, 2), title="u")

# We want to compare the relative influnce of advection and diffusion (Peclet number).
# In fact, this is used to calibrate the Peclet number for this problem as published
# in https://doi.org/10.1002/nme.7241

# Therefore, we construct the pure advection solution ua (k=0) and evaluate the
# L2-difference to the advection-diffusion solution.
ua = u.subs([(k, 0), (a, 1)])
plot3d(ua, (x, -1, 1), (t, 0, 2), title="ua")

dk = uak - ua
plot3d(dk, (x, -1, 1), (t, 0, 2), title="u-ua")

dk2 = (dk * dk).simplify()
plot3d(dk2, (x, -1, 1), (t, 0, 2), title="dk2")

dkIntt = integrate(dk2, (t, 0, 2))
dkIntxt = integrate(dkIntt, (x, -1, 1))
print(sqrt(dkIntxt.evalf()))

# And we construct the pure diffusion solution ud (a=0) and evaluate the
# L2-difference to the advection-diffusion solution.
ud = u.subs([(k, 0.1), (a, 0)])
plot3d(ud, (x, -1, 1), (t, 0, 2), title="ud")

da = uak - ud
plot3d(da, (x, -1, 1), (t, 0, 2), title="u - ud")

da2 = (da * da).simplify()
plot3d(da2, (x, -1, 1), (t, 0, 2), title="da2")

daIntx = integrate(da2, (x, -1, 1))
daIntxt = integrate(daIntx, (t, 0, 2))
print(sqrt(daIntxt.evalf()))

# Turns out that advective and diffusive effects are of similar importance for
# this choice of parameter values. Hence, we choue the charateristic lenth such
# that these parameter values result in a Peclet number of 1.
