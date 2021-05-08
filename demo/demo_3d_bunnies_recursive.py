import taichi as ti
import math
import time
import numpy as np
import os
import utils
from utils import create_output_folder
from engine.mpm_solver import MPMSolver
import argparse
from engine.mesh_io import load_mesh


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                        '--show',
                        action='store_true',
                        help='Run with gui')
    parser.add_argument('-f',
                        '--frames',
                        type=int,
                        default=10000,
                        help='Number of frames')
    parser.add_argument('-r', '--res', type=int, default=256, help='1 / dx')
    parser.add_argument('-o', '--out-dir', type=str, help='Output folder')
    parser.add_argument('-l', '--lod', type=int, help='LOG')
    parser.add_argument('-p',
                        '--output-ply',
                        type=str,
                        help='Output PLY files too')
    parser.add_argument('-m',
                        '--max-num-particles',
                        type=int,
                        help='Max num particles (millions)')
    parser.add_argument('--gpu-memory',
                        type=float,
                        default=9,
                        help='GPU memory')
    args = parser.parse_args()
    print(args)
    return args


args = parse_args()

with_gui = args.show
write_to_disk = args.out_dir is not None

# Try to run on GPU
ti.init(arch=ti.cuda,
        kernel_profiler=True,
        use_unified_memory=False,
        device_memory_GB=args.gpu_memory)

if with_gui:
    gui = ti.GUI("MLS-MPM",
                 res=1024,
                 background_color=0x112F41,
                 show_gui=args.show)

if write_to_disk:
    # output_dir = create_output_folder(args.out_dir)
    output_dir = args.out_dir
    os.makedirs(f'{output_dir}/particles')
    os.makedirs(f'{output_dir}/previews')
    print("Writing 2D vis and binary particle data to folder", output_dir)
else:
    output_dir = None

# Use 512 for final simulation/render
R = args.res

mpm = MPMSolver(res=(R, R, R),
                size=1,
                unbounded=True,
                dt_scale=1,
                quant=True,
                use_g2p2g=True,
                support_plasticity=False)

mpm.add_surface_collider(point=(0, 0, 0),
                         normal=(0, 1, 0),
                         surface=mpm.surface_slip,
                         friction=0.5)

mpm.add_surface_collider(point=(0, 1.9, 0),
                         normal=(0, -1, 0),
                         surface=mpm.surface_slip,
                         friction=0.5)

bound = 1.9

for d in [0, 2]:
    point = [0, 0, 0]
    normal = [0, 0, 0]
    b = bound
    point[d] = b
    normal[d] = -1
    mpm.add_surface_collider(point=point,
                             normal=normal,
                             surface=mpm.surface_separate,
                             friction=0.5)

    point[d] = -b
    normal[d] = 1
    mpm.add_surface_collider(point=point,
                             normal=normal,
                             surface=mpm.surface_separate,
                             friction=0.5)

bunnies = []
LOD = args.lod
h_start = 0.05
total_bunnies = 0

max_num_particles = args.max_num_particles * 1000000

for l in range(LOD):
    print(f"Generating LOD {l}")
    scale = 1 / 2**l * 0.625
    bunnies.append(
        load_mesh('bunny_low.ply', scale=scale * 0.6, offset=(0.5, 0.5, 0.5)))
    bb_size = scale
    bb_count = 2**(l + 1)
    layers = max(l - 1, 1)

    r = 255 if l % 3 == 0 else 128
    g = 255 if l % 3 == 1 else 128
    b = 255 if l % 3 == 2 else 128
    color = r * 65536 + g * 256 + b

    for k in range(layers):
        print(f"  Generating layer {k}, h_start {h_start}")
        for i in range(bb_count):
            for j in range(bb_count):
                x, y, z = -1.1 + (
                    i + 0.5) * bb_size, h_start + bb_size * k * 1.1, -1.1 + (
                        j + 0.5) * bb_size
                if mpm.n_particles[None] < max_num_particles:
                    mpm.add_mesh(triangles=bunnies[l],
                                 material=MPMSolver.material_elastic,
                                 color=color,
                                 velocity=(0, -5, 0),
                                 translation=(x, y, z))
                    print(
                        f'Total particles: {mpm.n_particles[None] / 1e6:.4f} M'
                    )
                    total_bunnies += 1
    h_start += bb_size * layers
    h_start -= 0.05 * max(0, 2 - l)  # adjustments

mpm.set_gravity((0, -25, 0))

print(f'Per particle space: {mpm.particle.cell_size_bytes} B')
print(f'Total bunnies: {total_bunnies}')
print(f'Total particles: {mpm.n_particles[None] / 1e6:.4f} M')


def visualize(particles, frame, output_dir=None):
    np_x = particles['position'] / 1.0

    screen_x = np_x[:, 0] * 0.5 + 0.5
    screen_y = np_x[:, 1] * 0.5

    screen_pos = np.stack([screen_x, screen_y], axis=-1)

    gui.circles(screen_pos, radius=1.0, color=particles['color'])
    if output_dir is None:
        gui.show()
    else:
        gui.show(f'{output_dir}/previews/{frame:05d}.png')


counter = 0

start_t = time.time()

for frame in range(args.frames):
    print(f'frame {frame}')
    t = time.time()
    mpm.step(1e-2, print_stat=True)
    if with_gui:
        particles = mpm.particle_info()
        visualize(particles, frame, output_dir)

    if write_to_disk:
        mpm.write_particles(f'{output_dir}/particles/{frame:05d}.npz')
        if args.output_ply:
            mpm.write_particles_ply(f'{output_dir}/particles/{frame:05d}.ply')
    print(f'Frame total time {time.time() - t:.3f}')
    print(f'Total running time {time.time() - start_t:.3f}')
