import taichi as ti
import numpy as np
import utils
from engine import mpm_solver

# Try to run on GPU
ti.init(arch=ti.cuda)

mpm = mpm_solver.MPMSolver(res=(24, 24, 24), size=1)
mpm.set_gravity((0, -20, 0))
mpm.add_surface_collider((0.5, 0.5, 0.5), (1.0, 1.0, 0.0))

for frame in range(5):
    mpm.add_cube((0.2, 0.8, 0.45), (0.1, 0.03, 0.1),
                 mpm.material_water,
                 color=0x8888FF)
    mpm.step(4e-3)
    particles = mpm.particle_info()
