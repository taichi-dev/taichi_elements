import taichi as ti
import math
import time
import numpy as np
from plyfile import PlyData, PlyElement
import os
from engine.mpm_solver import MPMSolver
import argparse


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
    parser.add_argument('-stateFre',
                        '--state-fre',
                        type=int,
                        default=128,
                        help='frequency to store the middle state for debug')
    parser.add_argument('-t',
                        '--thin',
                        action='store_true',
                        help='Use thin letters')
    parser.add_argument('-o', '--out-dir', type=str, help='Output folder')
    parser.add_argument('-i', '--state-dir', type=str, help='State folder to store the middle state')
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
        device_memory_GB=9)

# max_num_particles = 235000000
max_num_particles = 2e8
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
    os.makedirs(f'{output_dir}/states')
    print("Writing 2D vis and binary particle data to folder", output_dir)
else:
    output_dir = None


def load_mesh(fn, scale, offset):
    if isinstance(scale, (int, float)):
        scale = (scale, scale, scale)
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
            triangles[i, d * 3 + 0] = x[face[d]] * scale[0] + offset[0]
            triangles[i, d * 3 + 1] = y[face[d]] * scale[1] + offset[1]
            triangles[i, d * 3 + 2] = z[face[d]] * scale[2] + offset[2]

    print('loaded')

    return triangles


# Use 512 for final simulation/render
# R = 256
R = 128
count = ti.field(ti.i32, shape=())
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

for d in [0, 2]:
    point = [0, 0, 0]
    normal = [0, 0, 0]
    point[d] = 1.9
    normal[d] = -1
    mpm.add_surface_collider(point=point,
                             normal=normal,
                             surface=mpm.surface_slip,
                             friction=0.5)

    point[d] = -1.9
    normal[d] = 1
    mpm.add_surface_collider(point=point,
                             normal=normal,
                             surface=mpm.surface_slip,
                             friction=0.5)

if args.thin:
    # scale = (0.02, 0.02, 0.02)
    scale = (0.015, 0.015, 0.015)
else:
    scale = (0.02, 0.02, 0.8)

quantized = load_mesh('quantized.ply', scale=scale, offset=(0.5, 0.6, 0.5))
simulation = load_mesh('simulation.ply', scale=scale, offset=(0.5, 0.6, 0.5))

mpm.set_gravity((0, -25, 0))

print(f'Per particle space: {mpm.particle.cell_size_bytes} B')


def visualize(particles, frame, output_dir=None):
    np_x = particles['position'] / 1.0

    screen_x = np_x[:, 0]
    screen_y = np_x[:, 1]

    screen_pos = np.stack([screen_x, screen_y], axis=-1)

    gui.circles(screen_pos, radius=0.8, color=particles['color'])
    if output_dir is None:
        gui.show()
    else:
        gui.show(f'{output_dir}/previews/{frame:05d}.png')


counter = 0

start_t = time.time()


def seed_letters(subframe):
    i = subframe % 2
    j = subframe / 4 % 4 - 1

    r = 255 if subframe // 2 % 3 == 0 else 128
    g = 255 if subframe // 2 % 3 == 1 else 128
    b = 255 if subframe // 2 % 3 == 2 else 128
    color = r * 65536 + g * 256 + b
    triangles = quantized if subframe % 2 == 0 else simulation

    for y in range(-4, 2):
        mpm.add_mesh(triangles=triangles,
                     material=MPMSolver.material_elastic,
                     color=color,
                     velocity=(0, -2, 0),
                     translation=((i - 0.5) * 0.6, 0.1 * y, (2 - j) * 0.1))


def seed_bars(subframe):
    r = 255 if subframe % 3 == 0 else 128
    g = 255 if subframe % 3 == 1 else 128
    b = 255 if subframe % 3 == 2 else 128
    color = r * 65536 + g * 256 + b
    for i, t in zip(range(2), [quantized, simulation]):
        mpm.add_mesh(triangles=t,
                     material=MPMSolver.material_elastic,
                     color=color,
                     velocity=(0, -10, 0),
                     translation=((i - 0.5) * 0.6 - 0.5, 1.1, 0.1))


@ti.kernel
def count_activate(grid: ti.template()):
    for I in ti.grouped(grid):
        ti.atomic_add(count[None], 1)


@ti.kernel
def copy_grid(np_idx: ti.ext_arr(), np_val: ti.ext_arr(), grid: ti.template(), solver: ti.template()):
    """
    save the sparse grid_v or grid_m
    :param np_idx:
    :param np_val:
    :param grid:
    :return:
    """
    i = 0
    for I in ti.grouped(grid):
        j = ti.atomic_add(i, 1)
        # print(j, I)
        for d in ti.static(range(solver.dim)):
            np_idx[j, d] = I[d]
            np_val[j, d] = grid[I][d]


@ti.kernel
def copy_matrix(np_matrix: ti.ext_arr(), mt: ti.template(), solver: ti.template()):
    for p in range(solver.n_particles[None]):
        for j in ti.static(range(solver.dim)):
            for k in ti.static(range(solver.dim)):
                np_matrix[p, j, k] = mt[p][j, k]


def save_mpm_state(solver: MPMSolver, frame: int, save_dir: str):
    """
    Save MPM middle state as a npz file
    :param solver:
    :param frame:
    :param save_dir:
    :return:
    """
    if not solver.use_g2p2g:
        raise NotImplementedError
    particles = solver.particle_info()  # particle information
    # other meta data
    phase = solver.input_grid
    # save grid_v
    count_activate(solver.grid_v[phase])
    print(f"we have {count[None]} nodes activated")
    np_grid_idx = np.ndarray((count[None], solver.dim), dtype=np.float32)
    np_grid_val = np.ndarray((count[None], solver.dim), dtype=np.float32)
    copy_grid(np_grid_idx, np_grid_val, solver.grid_v[phase], solver)

    # save deformation gradient
    np_F = np.ndarray((solver.n_particles[None], solver.dim, solver.dim), dtype=np.float32)
    copy_matrix(np_F, solver.F, solver)
    particles['F'] = np_F

    if mpm.support_plasticity:
        np_j = np.ndarray((solver.n_particles[None],), dtype=np.float32)
        solver.copy_dynamic(np_j, solver.Jp)
        particles['p_Jp'] = np_j

    np.savez(save_dir,
             frame=frame,
             input_grid=phase,
             grid_v_idx=np_grid_idx,
             grid_v_val=np_grid_val,
             **particles
             )
    print(f'save {frame}th frame to {save_dir}')
    return


@ti.kernel
def copyback_dynamic_nd(solver: ti.template(), np_x: ti.ext_arr(), input_x: ti.template()):
    for i in range(solver.n_particles[None]):
        for j in ti.static(range(solver.dim)):
            # print(i, j)
            input_x[i][j] = np_x[i, j]


@ti.kernel
def copyback_dynamic(solver: ti.template(), np_x: ti.ext_arr(), input_x: ti.template()):
    for i in range(solver.n_particles[None]):
        # print(i)
        input_x[i] = np_x[i]


@ti.kernel
def copyback_grid(np_idx: ti.ext_arr(), np_val: ti.ext_arr(), grid: ti.template(), solver: ti.template()):
    num_active_cell = np_idx.shape[0]
    for i in range(num_active_cell):
        idx = []
        val = []
        for j in ti.static(range(solver.dim)):
            idx.append(int(np_idx[i, j]))
            val.append(np_val[i, j])

        ti_idx = ti.Vector(idx)
        ti_val = ti.Vector(val)
        grid[ti_idx] = ti_val


@ti.kernel
def copyback_matrix(np_matrix: ti.ext_arr(), mt: ti.template(), solver: ti.template()):
    for p in range(solver.n_particles[None]):
        for j in ti.static(range(solver.dim)):
            for k in ti.static(range(solver.dim)):
                mt[p][j, k] = np_matrix[p, j, k]


def load_mpm_state(solver: MPMSolver, save_dir: str):
    if not solver.use_g2p2g:
        raise NotImplementedError
    # load particle information
    state = np.load(save_dir)
    resume_frame = state['frame']
    phase = state['input_grid']

    num_particle = state['position'].shape[0]
    # assert(num_particle == )
    print(f"we have {num_particle} particles !")
    solver.n_particles[None] = num_particle

    solver.input_grid = phase
    copyback_dynamic_nd(solver, state['position'], solver.x)
    copyback_dynamic_nd(solver, state['velocity'], solver.v)
    copyback_dynamic(solver, state['material'], solver.material)
    copyback_dynamic(solver, state['color'], solver.color)
    copyback_matrix(state['F'], solver.F, solver)
    if solver.support_plasticity:
        copyback_dynamic(solver, state['p_Jp'], solver.Jp)

    copyback_grid(state['grid_v_idx'], state['grid_v_val'], solver.grid_v[phase], solver)

    print(f'load {resume_frame}th frame from {save_dir}!')
    return resume_frame


start_frame = 0
# load the state dict for debug
if args.state_dir is not None:
    start_frame = load_mpm_state(mpm, args.state_dir) + 1

for frame in range(start_frame, args.frames):
    print(f'frame {frame}')
    t = time.time()
    if args.thin:
        frame_split = 5
    else:
        frame_split = 1

    for subframe in range(frame * frame_split, (frame + 1) * frame_split):
        if mpm.n_particles[None] < max_num_particles:
            if args.thin:
                seed_letters(subframe)
            else:
                seed_bars(subframe)

        mpm.step(1e-2 / frame_split, print_stat=True)

    if frame % args.state_fre == 0:
        save_mpm_state(mpm, frame, f'{output_dir}/states/{frame:05d}.npz')

    if with_gui and frame % 4 == 0:
        particles = mpm.particle_info()
        visualize(particles, frame, output_dir)

    if write_to_disk and frame % 4 == 0:
        mpm.write_particles(f'{output_dir}/particles/{frame:05d}.npz')
    print(f'Frame total time {time.time() - t:.3f}')
    print(f'Total running time {time.time() - start_t:.3f}')
