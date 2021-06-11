import taichi as ti
import numpy as np
import os
import utils
from engine.mpm_solver import MPMSolver

write_to_disk = False

if write_to_disk:
    os.makedirs('outputs', exist_ok=True)

# Try to run on GPU
ti.init(arch=ti.cuda, device_memory_GB=2.0, use_unified_memory=False)

gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41)

mpm = MPMSolver(res=(64, 64, 64), size=1)

mpm.set_gravity((0, -20, 0))

mpm.add_sphere_collider(center=(0.25, 0.5, 0.5),
                        radius=0.1,
                        surface=mpm.surface_slip)
mpm.add_sphere_collider(center=(0.5, 0.5, 0.5),
                        radius=0.1,
                        surface=mpm.surface_sticky)
mpm.add_sphere_collider(center=(0.75, 0.5, 0.5),
                        radius=0.1,
                        surface=mpm.surface_separate)

for frame in range(1500):
    mpm.add_cube((0.2, 0.8, 0.45), (0.1, 0.03, 0.1),
                 mpm.material_water,
                 color=0x8888FF)
    mpm.add_cube((0.45, 0.8, 0.45), (0.1, 0.03, 0.1),
                 mpm.material_water,
                 color=0xFF8888)
    mpm.add_cube((0.7, 0.8, 0.45), (0.1, 0.03, 0.1),
                 mpm.material_water,
                 color=0xFFFFFF)
    mpm.step(4e-3)
    particles = mpm.particle_info()
    np_x = particles['position'] / 1.0

    # simple camera transform
    screen_x = ((np_x[:, 0] + np_x[:, 2]) / 2**0.5) - 0.2
    screen_y = (np_x[:, 1])

    screen_pos = np.stack([screen_x, screen_y], axis=-1)

    gui.circles(screen_pos, radius=1.1, color=particles['color'])
    gui.show(f'outputs/{frame:06d}.png' if write_to_disk else None)
