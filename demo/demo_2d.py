import os
import taichi as ti
import numpy as np
import utils
import math
from engine.mpm_solver import MPMSolver
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out-dir', type=str, help='Output folder')
    args = parser.parse_args()
    print(args)
    return args

args = parse_args()

write_to_disk = args.out_dir is not None
if write_to_disk:
    os.mkdir(f'{args.out_dir}')

ti.init(arch=ti.cuda)  # Try to run on GPU

gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41)

mpm = MPMSolver(res=(128, 128))

for i in range(3):
    mpm.add_cube(lower_corner=[0.2 + i * 0.1, 0.3 + i * 0.1],
                 cube_size=[0.1, 0.1],
                 material=MPMSolver.material_elastic)

for frame in range(500):
    mpm.step(8e-3)
    if frame < 500:
        mpm.add_cube(lower_corner=[0.1, 0.8],
                     cube_size=[0.01, 0.05],
                     velocity=[1, 0],
                     material=MPMSolver.material_sand)
    if 10 < frame < 100:
        mpm.add_cube(lower_corner=[0.6, 0.7],
                     cube_size=[0.2, 0.01],
                     material=MPMSolver.material_water,
                     velocity=[math.sin(frame * 0.1), 0])
    if 120 < frame < 200 and frame % 10 == 0:
        mpm.add_cube(
            lower_corner=[0.4 + frame * 0.001, 0.6 + frame // 40 * 0.02],
            cube_size=[0.2, 0.1],
            velocity=[-3, -1],
            material=MPMSolver.material_snow)
    colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00],
                      dtype=np.uint32)
    particles = mpm.particle_info()
    gui.circles(particles['position'],
                radius=1.5,
                color=colors[particles['material']])
    gui.show(f'{args.out_dir}/{frame:06d}.png' if write_to_disk else None)
