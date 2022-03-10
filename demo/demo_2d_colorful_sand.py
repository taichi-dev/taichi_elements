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

for frame in range(500):
    mpm.step(8e-3)
    mpm.add_cube(lower_corner=[0.1, 0.8],
                 cube_size=[0.01, 0.05],
                 velocity=[1, 0],
                 material=MPMSolver.material_sand)
    particles = mpm.particle_info()
    gui.circles(particles['position'],
                radius=1.5,
                color=particles['color'])
    gui.show(f'{args.out_dir}/{frame:06d}.png' if write_to_disk else None)
