import taichi as ti
import numpy as np
from mpm import MPMSolver

# Try to run on GPU
# ti.cfg.arch = ti.cuda

gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)

mpm = MPMSolver(res=(128, 128))

for i in range(3):
  mpm.add_cube(lower_corner=[0.2 + i * 0.1, 0.3 + i * 0.1], cube_size=[0.1, 0.1], material=MPMSolver.material_snow)

for frame in range(20000):
  mpm.step(2e-3)
  colors = np.array([0x068587, 0xED553B, 0xEEEEF0], dtype=np.uint32)
  np_x, np_v, np_material = mpm.particle_info()
  gui.circles(np_x, radius=1.5, color=colors[np_material])
  gui.show()  # Change to gui.show(f'{frame:06d}.png') to write images to disk
