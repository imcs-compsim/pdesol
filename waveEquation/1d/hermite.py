#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:13:38 2023

@author: maxvondanwitz
"""

def quinticHermiteInterpolation(x, x_0, dx, fx0, fx1, f1x0, f1x1, f2x0, f2x1):
    # Quintic Hermite Interpolation (https://en.wikipedia.org/wiki/Hermite_interpolation)
    
    # Second support point
    x_1 = x_0 + dx
    
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
