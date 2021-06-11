import taichi as ti
import numpy as np
import utils
from engine.mpm_solver import MPMSolver

write_to_disk = False

# Try to run on GPU
ti.init(arch=ti.cuda, device_memory_GB=3.0)

gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41)

mpm = MPMSolver(res=(64, 64, 64), size=1)

triangles = np.fromfile('suzanne.npy', dtype=np.float32)
triangles = np.reshape(triangles, (len(triangles) // 9, 9)) * 0.306 + 0.501

mpm.add_mesh(triangles=triangles,
             material=MPMSolver.material_elastic,
             color=0xFFFF00)

mpm.set_gravity((0, -20, 0))

for frame in range(1500):
    mpm.step(4e-3)
    particles = mpm.particle_info()
    np_x = particles['position'] / 1.0

    # simple camera transform
    screen_x = ((np_x[:, 0] + np_x[:, 2]) / 2**0.5) - 0.2
    screen_y = (np_x[:, 1])

    screen_pos = np.stack([screen_x, screen_y], axis=-1)

    gui.circles(screen_pos, radius=1.1, color=particles['color'])
    gui.show(f'{frame:06d}.png' if write_to_disk else None)
