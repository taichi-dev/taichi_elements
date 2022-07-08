import taichi as ti
import numpy as np


@ti.func
def cross2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


@ti.func
def inside_ccw(p, a, b, c):
    return cross2d(a - p, b - p) >= 0 and cross2d(
        b - p, c - p) >= 0 and cross2d(c - p, a - p) >= 0


@ti.data_oriented
class Voxelizer:
    def __init__(self, res, dx, super_sample=2, precision=ti.f64, padding=3):
        assert len(res) == 3
        res = list(res)
        for i in range(len(res)):
            r = 1
            while r < res[i]:
                r = r * 2
            res[i] = r
        print(f'Voxelizer resolution {res}')
        # Super sample by 2x
        self.res = (res[0] * super_sample, res[1] * super_sample,
                    res[2] * super_sample)
        self.dx = dx / super_sample
        self.inv_dx = 1 / self.dx
        self.voxels = ti.field(ti.i32)
        self.block = ti.root.pointer(
            ti.ijk, (self.res[0] // 8, self.res[1] // 8, self.res[2] // 8))
        self.block.dense(ti.ijk, 8).place(self.voxels)

        assert precision in [ti.f32, ti.f64]
        self.precision = precision
        self.padding = padding

    @ti.func
    def fill(self, p, q, height, inc):
        for i in range(self.padding, height):
            self.voxels[p, q, i] += inc

    @ti.kernel
    def voxelize_triangles(self, num_triangles: ti.i32,
                           triangles: ti.types.ndarray()):
        for i in range(num_triangles):
            jitter_scale = ti.cast(0, self.precision)
            if ti.static(self.precision == ti.f32):
                jitter_scale = 1e-4
            else:
                jitter_scale = 1e-8
            # We jitter the vertices to prevent voxel samples from lying precicely at triangle edges
            jitter = ti.Vector([
                -0.057616723909439505, -0.25608986292614977,
                0.06716309129743714
            ]) * jitter_scale
            a = ti.Vector([triangles[i, 0], triangles[i, 1], triangles[i, 2]
                           ]) + jitter
            b = ti.Vector([triangles[i, 3], triangles[i, 4], triangles[i, 5]
                           ]) + jitter
            c = ti.Vector([triangles[i, 6], triangles[i, 7], triangles[i, 8]
                           ]) + jitter

            bound_min = ti.Vector.zero(self.precision, 3)
            bound_max = ti.Vector.zero(self.precision, 3)
            for k in ti.static(range(3)):
                bound_min[k] = min(a[k], b[k], c[k])
                bound_max[k] = max(a[k], b[k], c[k])

            p_min = int(ti.floor(bound_min[0] * self.inv_dx))
            p_max = int(ti.floor(bound_max[0] * self.inv_dx)) + 1

            p_min = max(self.padding, p_min)
            p_max = min(self.res[0] - self.padding, p_max)

            q_min = int(ti.floor(bound_min[1] * self.inv_dx))
            q_max = int(ti.floor(bound_max[1] * self.inv_dx)) + 1

            q_min = max(self.padding, q_min)
            q_max = min(self.res[1] - self.padding, q_max)

            normal = ((b - a).cross(c - a)).normalized()

            if abs(normal[2]) < 1e-10:
                continue

            a_proj = ti.Vector([a[0], a[1]])
            b_proj = ti.Vector([b[0], b[1]])
            c_proj = ti.Vector([c[0], c[1]])

            for p in range(p_min, p_max):
                for q in range(q_min, q_max):
                    pos2d = ti.Vector([(p + 0.5) * self.dx,
                                       (q + 0.5) * self.dx])
                    if inside_ccw(pos2d, a_proj, b_proj, c_proj) or inside_ccw(
                            pos2d, a_proj, c_proj, b_proj):
                        base_voxel = ti.Vector([pos2d[0], pos2d[1], 0])
                        height = int(-normal.dot(base_voxel - a) / normal[2] *
                                     self.inv_dx + 0.5)
                        height = min(height, self.res[1] - self.padding)
                        inc = 0
                        if normal[2] > 0:
                            inc = 1
                        else:
                            inc = -1
                        self.fill(p, q, height, inc)

    def voxelize(self, triangles):
        assert isinstance(triangles, np.ndarray)
        triangles = triangles.astype(np.float64)
        assert triangles.dtype in [np.float32, np.float64]
        if self.precision is ti.f32:
            triangles = triangles.astype(np.float32)
        elif self.precision is ti.f64:
            triangles = triangles.astype(np.float64)
        else:
            assert False
        assert len(triangles.shape) == 2
        assert triangles.shape[1] == 9

        self.block.deactivate_all()
        num_triangles = len(triangles)
        self.voxelize_triangles(num_triangles, triangles)


if __name__ == '__main__':
    n = 256
    vox = Voxelizer((n, n, n), 1.0 / n)
    # triangle = np.array([[0.1, 0.1, 0.1, 0.6, 0.2, 0.1, 0.5, 0.7,
    #                       0.7]]).astype(np.float32)
    triangles = np.fromfile('triangles.npy', dtype=np.float32)
    triangles = np.reshape(triangles, (len(triangles) // 9, 9)) * 0.306 + 0.501
    offsets = [0.0, 0.0, 0.0]
    for i in range(9):
        triangles[:, i] += offsets[i % 3]
    print(triangles.shape)
    print(triangles.max())
    print(triangles.min())

    vox.voxelize(triangles)

    voxels = vox.voxels.to_numpy().astype(np.float32)

    import os
    os.makedirs('outputs', exist_ok=True)
    gui = ti.GUI('cross section', (n, n))
    for i in range(n):
        gui.set_image(voxels[:, :, i])
        gui.show(f'outputs/{i:04d}.png')
