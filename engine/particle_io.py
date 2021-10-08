from engine.mesh_io import write_point_cloud
import numpy as np
import taichi as ti
import time
import gc


class ParticleIO:
    v_bits = 8
    x_bits = 32 - v_bits

    @staticmethod
    def write_particles(solver, fn, slice_size=1000000):
        t = time.time()
        output_fn = fn

        n_particles = solver.n_particles[None]

        x_and_v = np.ndarray((n_particles, solver.dim), dtype=np.uint32)
        # Value ranges of x and v components, for quantization
        ranges = np.ndarray((2, solver.dim, 2), dtype=np.float32)

        # Fetch data slice after slice since we don't have the GPU memory to fetch them channel after channel...
        num_slices = (n_particles + slice_size - 1) // slice_size

        for d in range(solver.dim):
            np_x = np.ndarray((n_particles, ), dtype=np.float32)
            np_v = np.ndarray((n_particles, ), dtype=np.float32)

            np_x_slice = np.ndarray((slice_size, ), dtype=np.float32)
            np_v_slice = np.ndarray((slice_size, ), dtype=np.float32)

            for s in range(num_slices):
                begin = slice_size * s
                end = min(slice_size * (s + 1), n_particles)
                solver.copy_ranged(np_x_slice, solver.x.get_scalar_field(d),
                                   begin, end)
                solver.copy_ranged(np_v_slice, solver.v.get_scalar_field(d),
                                   begin, end)

                np_x[begin:end] = np_x_slice[:end - begin]
                np_v[begin:end] = np_v_slice[:end - begin]

            ranges[0, d] = [np.min(np_x), np.max(np_x)]
            ranges[1, d] = [np.min(np_v), np.max(np_v)]

            # Avoid too narrow ranges
            for c in range(2):
                ranges[c, d, 1] = max(ranges[c, d, 0] + 1e-5, ranges[c, d, 1])
            np_x = ((np_x - ranges[0, d, 0]) *
                    (1 / (ranges[0, d, 1] - ranges[0, d, 0])) *
                    (2**ParticleIO.x_bits - 1) + 0.499).astype(np.uint32)
            np_v = ((np_v - ranges[1, d, 0]) *
                    (1 / (ranges[1, d, 1] - ranges[1, d, 0])) *
                    (2**ParticleIO.v_bits - 1) + 0.499).astype(np.uint32)
            x_and_v[:, d] = (np_x << ParticleIO.v_bits) + np_v
            del np_x, np_v

        color = np.ndarray((n_particles, 3), dtype=np.uint8)
        np_color = np.ndarray((n_particles, ), dtype=np.uint32)

        np_color_slice = np.ndarray((slice_size, ), dtype=np.float32)

        for s in range(num_slices):
            begin = slice_size * s
            end = min(slice_size * (s + 1), n_particles)

            solver.copy_ranged(np_color_slice, solver.color, begin, end)
            np_color[begin:end] = np_color_slice[:end - begin]

        for c in range(3):
            color[:, c] = (np_color >> (8 * (2 - c))) & 255

        np.savez(output_fn, ranges=ranges, x_and_v=x_and_v, color=color)

        print(f'Writing to disk: {time.time() - t:.3f} s')

    @staticmethod
    def read_particles_3d(fn):
        return ParticleIO.read_particles(fn, 3)

    @staticmethod
    def read_particles_2d(fn):
        return ParticleIO.read_particles(fn, 2)

    @staticmethod
    def read_particles(fn, dim):
        data = np.load(fn)
        ranges = data['ranges']
        color = data['color']
        x_and_v = data['x_and_v']
        del data
        gc.collect()
        x = (x_and_v >> ParticleIO.v_bits).astype(np.float32) / (
            (2**ParticleIO.x_bits - 1))
        for c in range(dim):
            x[:,
              c] = x[:, c] * (ranges[0, c, 1] - ranges[0, c, 0]) + ranges[0, c,
                                                                          0]
        v = (x_and_v & (2**ParticleIO.v_bits - 1)).astype(
            np.float32) / (2**ParticleIO.v_bits - 1)
        for c in range(dim):
            v[:,
              c] = v[:, c] * (ranges[1, c, 1] - ranges[1, c, 0]) + ranges[1, c,
                                                                          0]
        return x, v, color

    @staticmethod
    def convert_particle_to_ply(fns):
        for fn in fns:
            print(f'Converting {fn}...')
            x, _, color = ParticleIO.read_particles_3d(fn)
            x = x.astype(np.float32)
            color = (color[:, 2].astype(np.uint32) << 16) + (
                color[:, 1].astype(np.uint32) << 8) + color[:, 0]
            color = color[:, None]
            pos_color = np.hstack([x, color.view(np.float32)])
            del x, color
            gc.collect()
            write_point_cloud(fn + ".ply", pos_color)


if __name__ == '__main__':
    import sys
    ParticleIO.convert_particle_to_ply(sys.argv[1:])
