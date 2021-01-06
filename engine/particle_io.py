import numpy as np
import taichi as ti
import time


class ParticleIO:
    v_bits = 8
    x_bits = 32 - v_bits

    @staticmethod
    def write_particles(solver, fn):
        t = time.time()
        output_fn = fn

        n_particles = solver.n_particles[None]

        x_and_v = np.ndarray((n_particles, solver.dim), dtype=np.uint32)
        # Value ranges of x and v components, for quantization
        ranges = np.ndarray((2, solver.dim, 2), dtype=np.float32)

        for d in range(solver.dim):
            np_x = np.ndarray((n_particles, ), dtype=np.float32)
            np_v = np.ndarray((n_particles, ), dtype=np.float32)
            solver.copy_dynamic(np_x, solver.x(d))
            solver.copy_dynamic(np_v, solver.v(d))
            ranges[0, d] = [np.min(np_x), np.max(np_x)]
            ranges[1, d] = [np.min(np_v), np.max(np_v)]

            # Avoid too narrow ranges
            for c in range(2):
                ranges[c, d, 1] = max(ranges[c, d, 0] + 1e-5, ranges[c, d, 1])
            np_x = ((np_x - ranges[0, d, 0]) *
                    (1 / (ranges[0, d, 1] - ranges[0, d, 0])) *
                    (2**ParticleIO.x_bits - 1) + 0.5).astype(np.uint32)
            np_v = ((np_v - ranges[1, d, 0]) *
                    (1 / (ranges[1, d, 1] - ranges[1, d, 0])) *
                    (2**ParticleIO.v_bits - 1) + 0.5).astype(np.uint32)
            x_and_v[:, d] = (np_x << ParticleIO.v_bits) + np_v
            del np_x, np_v

        color = np.ndarray((n_particles, 3), dtype=np.uint8)
        np_color = np.ndarray((n_particles, ), dtype=np.uint32)
        solver.copy_dynamic(np_color, solver.color)
        for c in range(3):
            color[:, c] = (np_color >> (8 * (2 - c))) & 255

        np.savez(output_fn, ranges=ranges, x_and_v=x_and_v, color=color)

        print(f'Writing to disk: {time.time() - t:.3f} s')

    @staticmethod
    def read_particles_3d(fn):
        data = np.load(fn)
        ranges = data['ranges']
        color = data['color']
        x_and_v = data['x_and_v']
        x = (x_and_v >> ParticleIO.v_bits).astype(np.float32) / (
            (2**ParticleIO.x_bits - 1))
        for c in range(3):
            x[:, c] = x[:, c] * (ranges[0, c, 1] - ranges[0, c, 0]) + ranges[0, c, 0]
        v = (x_and_v & (2**ParticleIO.v_bits - 1)).astype(
            np.float32) / (2**ParticleIO.v_bits - 1)
        for c in range(3):
            v[:, c] = v[:, c] * (ranges[1, c, 1] - ranges[0, c, 1]) + ranges[1, c, 0]
        return x, v, color
