#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 14:10:55 2023

@author: maxvondanwitz
"""

import numpy as np
import matplotlib.pyplot as plt

def plotOwn3d(expr, var1, var1min, var1max, var2, var2min, var2max, N):

    xloc = np.linspace(var1min, var1max, N)
    yloc = np.linspace(var2min, var2max, N)
    X, Y = np.meshgrid(xloc, yloc)
    Z = np.zeros([N, N])

    for i, xi in enumerate(xloc):
        for j, yi in enumerate(yloc):
            Z[i,j] = expr.subs([(var1, xi), (var2, yi)]) 

    # Plot the surface
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z)

    plt.show()
    
    return