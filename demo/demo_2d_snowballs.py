import taichi as ti
import numpy as np
import utils
from engine.mpm_solver import MPMSolver

write_to_disk = False

ti.init(arch=ti.cuda)  # Try to run on GPU

gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)

mpm = MPMSolver(res=(256, 256))
mpm.set_gravity([0, 0])

mpm.add_ellipsoid(center=[0.25, 0.45],
                  radius=0.07,
                  velocity=[1, 0],
                  color=0xAAAAFF,
                  material=MPMSolver.material_snow)

mpm.add_ellipsoid(center=[0.75, 0.52],
                  radius=0.07,
                  velocity=[-1, 0],
                  color=0xFFAAAA,
                  material=MPMSolver.material_snow)

for frame in range(500):
    mpm.step(8e-3)
    particles = mpm.particle_info()
    gui.circles(particles['position'], radius=1.5, color=particles['color'])
    gui.show(f'{frame:06d}.png' if write_to_disk else None)
