import os
import taichi as ti
import math
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

gui = ti.GUI("Taichi Elements", res=512, background_color=0xaabbcc)

mpm = MPMSolver(res=(128, 128))

for frame in range(500):
    mpm.step(8e-3)


    colors = [0x000000, 0xFFFFFF, 0x5588ff]
    for i in range(3):
        f = math.sin(frame * 0.05 + 2 * math.pi / 3 * i)
        g = math.sin((5 + frame) * 0.02 + 2 * math.pi / 3 * i) + 1
        mpm.add_cube(lower_corner=[0.5 + 0.3 * f, 0.5 + 0.1 * i * i],
                     cube_size=[0.02, 0.02 * g],
                     velocity=[0, 0],
                     material=MPMSolver.material_sand,
                     color=colors[i])
    particles = mpm.particle_info()
    gui.circles(particles['position'],
                radius=1.5,
                color=particles['color'])
    gui.show(f'{args.out_dir}/{frame:06d}.png' if write_to_disk else None)
