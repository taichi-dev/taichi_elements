import taichi as ti
import numpy as np
import utils
from engine.mpm_solver import MPMSolver


# Try to run on GPU
ti.init(arch=ti.cuda)


mpm = MPMSolver(res=(24, 24, 24), size=10)

mpm.add_ellipsoid(center=[2, 4, 3],
                  radius=1,
                  material=MPMSolver.material_snow,
                  velocity=[0, -10, 0])
mpm.add_cube(lower_corner=[2, 6, 3],
             cube_size=[1, 1, 3],
             material=MPMSolver.material_elastic)

mpm.set_gravity((0, -50, 0))

for frame in range(5):
    mpm.step(4e-3)
    colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00], dtype=np.uint32)
    particles = mpm.particle_info()
    np_x = particles['position'] / 10.0
