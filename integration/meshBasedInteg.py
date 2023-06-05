# Taken from https://docu.ngsolve.org/v6.2.1808/whetting_the_appetite/poisson.html

from ngsolve import *
from netgen.geom2d import unit_square

# Generate a triangular mesh of mesh-size 0.2
mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))

# To see the really coarse mesh, I ran `netgen meshBasedInteg.py` in my shell.
Draw(mesh)

exact = 16 * x * (1 - x) * y * (1 - y)
l2 = sqrt(Integrate(exact * exact, mesh))
print("L2 norm with mesh based integral:", l2)

# The analytical result is:
L2 = 8 / 15
print("Relative error:", (L2 - l2) / L2)
