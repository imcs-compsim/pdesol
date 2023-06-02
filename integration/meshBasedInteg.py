# Taken from https://docu.ngsolve.org/v6.2.1808/whetting_the_appetite/poisson.html

from ngsolve import *
from netgen.geom2d import unit_square

# generate a triangular mesh of mesh-size 0.2
mesh = Mesh(unit_square.GenerateMesh(maxh=0.2))

# to see the really coars mesh, I ran `netgen meshBasedInteg.py` in my shell
Draw(mesh)

exact = 16 * x * (1 - x) * y * (1 - y)
L2 = sqrt(Integrate(exact * exact, mesh))
print("L2 norm with mesh based integral:", L2)
print("Relative error:", (8 / 15 - L2) / (8 / 15))
