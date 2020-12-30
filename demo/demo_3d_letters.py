import taichi as ti
import math
import time
import numpy as np
from plyfile import PlyData, PlyElement
import os
import utils
from utils import create_output_folder
from engine.mpm_solver import MPMSolver

with_gui = True
write_to_disk = True

# Try to run on GPU
ti.init(arch=ti.cuda,
        kernel_profiler=True,
        use_unified_memory=False,
        device_memory_fraction=0.7)

max_num_particles = 10000000

if with_gui:
    gui = ti.GUI("MLS-MPM", res=512, background_color=0x112F41)

if write_to_disk:
    output_dir = create_output_folder('./sim')


def load_mesh(fn, scale, offset):
    print(f'loading {fn}')
    plydata = PlyData.read(fn)
    x = plydata['vertex']['x']
    y = plydata['vertex']['y']
    z = plydata['vertex']['z']
    elements = plydata['face']
    num_tris = len(elements['vertex_indices'])
    triangles = np.zeros((num_tris, 9), dtype=np.float32)

    for i, face in enumerate(elements['vertex_indices']):
        assert len(face) == 3
        for d in range(3):
            triangles[i, d * 3 + 0] = x[face[d]] * scale + offset[0]
            triangles[i, d * 3 + 1] = y[face[d]] * scale + offset[1]
            triangles[i, d * 3 + 2] = z[face[d]] * scale + offset[2]

    print('loaded')

    return triangles


# Use 512 for final simulation/render
R = 256

mpm = MPMSolver(res=(R, R, R), size=1, unbounded=True, dt_scale=1)

mpm.add_surface_collider(point=(0, 0, 0),
                         normal=(0, 1, 0),
                         surface=mpm.surface_slip,
                         friction=0.5)

triangles = load_mesh('taichi.ply', scale=0.02, offset=(0.5, 0.6, 0.5))

mpm.set_gravity((0, -25, 0))


def visualize(particles):
    np_x = particles['position'] / 1.0

    # simple camera transform
    screen_x = ((np_x[:, 0] + np_x[:, 2]) / 2**0.5) - 0.2
    screen_y = (np_x[:, 1])

    screen_pos = np.stack([screen_x, screen_y], axis=-1)

    gui.circles(screen_pos, radius=0.8, color=particles['color'])
    gui.show()


counter = 0

start_t = time.time()

for frame in range(15000):
    print(f'frame {frame}')
    t = time.time()
    if mpm.n_particles[None] < max_num_particles:
        i = frame % 4 - 2
        j = frame / 4 % 4 - 1

        r = 255 if frame % 3 == 0 else 128
        g = 255 if frame % 3 == 1 else 128
        b = 255 if frame % 3 == 2 else 128
        color = r * 65536 + g * 256 + b
        mpm.add_mesh(triangles=triangles,
                     material=MPMSolver.material_elastic,
                     color=color,
                     velocity=(0, -2, 0),
                     translation=((i + 0.5) * 0.25, 0, (2 - j) * 0.1))

    mpm.step(2e-3, print_stat=True)
    if with_gui and frame % 3 == 0:
        particles = mpm.particle_info()
        visualize(particles)

    if write_to_disk:
        mpm.write_particles(f'{output_dir}/{frame:05d}.npz')
    print(f'Frame total time {time.time() - t:.3f}')
    print(f'Total running time {time.time() - start_t:.3f}')
