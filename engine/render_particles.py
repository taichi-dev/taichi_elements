import taichi as ti
import os
import time
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b',
                        '--begin',
                        type=int,
                        default=0,
                        help='Beginning frame')
    parser.add_argument('-e',
                        '--end',
                        type=int,
                        default=10000,
                        help='Ending frame')
    parser.add_argument('-s', '--step', type=int, default=1, help='Frame step')
    parser.add_argument('-r',
                        '--res',
                        type=int,
                        default=512,
                        help='Grid resolution')
    parser.add_argument('-g', '--gui', action='store_true', help='Show GUI')
    parser.add_argument('-o', '--out-dir', type=str, help='Output folder')
    parser.add_argument('-i', '--in-dir', type=str, help='Input folder')
    parser.add_argument('-t',
                        '--shutter-time',
                        type=float,
                        default=2e-3,
                        help='Shutter time')
    parser.add_argument('-f',
                        '--force',
                        action='store_true',
                        help='Overwrite existing outputs')
    parser.add_argument('--gpu-memory',
                        type=float,
                        default=9,
                        help='GPU memory')
    parser.add_argument('-M',
                        '--max-particles',
                        type=int,
                        default=128,
                        help='Max num particles (million)')
    args = parser.parse_args()
    print(args)
    return args


args = parse_args()

ti.init(arch=ti.cuda,
        use_unified_memory=False,
        device_memory_GB=args.gpu_memory)

output_folder = args.out_dir
os.makedirs(output_folder, exist_ok=True)

from engine.renderer import Renderer

res = args.res
renderer = Renderer(dx=1 / res,
                    sphere_radius=0.3 / res,
                    shutter_time=args.shutter_time,
                    max_num_particles_million=args.max_particles,
                    taichi_logo=False)

with_gui = args.gui
if with_gui:
    gui = ti.GUI('Particle Renderer', (1280, 720))

spp = 200

# 0.23, (0.0, 0.8, 5.5)


def main():
    for f in range(args.begin, args.end, args.step):
        print('frame', f, end=' ')
        output_fn = f'{output_folder}/{f:05d}.png'
        if os.path.exists(output_fn) and not args.force:
            print('skip.')
            continue
        else:
            print('rendering...')
        Path(output_fn).touch()
        t = time.time()

        renderer.set_camera_pos(3.24, 1.86, -4.57)
        renderer.floor_height[None] = -5e-3

        renderer.initialize_particles_from_taichi_elements(
            f'{args.in_dir}/{f:05d}.npz')

        total_voxels = renderer.total_non_empty_voxels()
        total_inserted_particles = renderer.total_inserted_particles()
        print('Total particles (with motion blur)', total_inserted_particles)
        print('Total nonempty voxels', total_voxels)
        print('Average particle_list_length',
              total_inserted_particles / total_voxels)
        img = renderer.render_frame(spp=spp)

        if with_gui:
            gui.set_image(img)
            gui.show(output_fn)
        else:
            ti.imwrite(img, output_fn)

        print(f'Frame rendered. {spp} take {time.time() - t} s.')


# if __name__ == '__main__':
main()
