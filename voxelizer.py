import taichi as ti
import numpy as np


@ti.func
def cross2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


@ti.func
def inside_ccw(p, a, b, c):
    eps = 1e-10
    return cross2d(a - p, b - p) >= -eps and cross2d(
        b - p, c - p) >= -eps and cross2d(c - p, a - p) >= -eps

@ti.data_oriented
class Voxelizer:
    def __init__(self, res, dx):
        assert len(res) == 3
        self.res = res
        self.dx = dx
        self.inv_dx = 1 / self.dx
        self.voxels = ti.var(ti.i32, shape=res)

    @ti.func
    def fill(self, p, q, height, inc):
        for i in range(height):
            self.voxels[p, q, i] += inc

    @ti.kernel
    def voxelize_triangles(self, num_triangles: ti.i32,
                           triangles: ti.ext_arr()):
        for i in range(num_triangles):
            a = ti.Vector([triangles[i, 0], triangles[i, 1], triangles[i, 2]])
            b = ti.Vector([triangles[i, 3], triangles[i, 4], triangles[i, 5]])
            c = ti.Vector([triangles[i, 6], triangles[i, 7], triangles[i, 8]])

            bound_min = ti.Vector.zero(ti.f32, 3)
            bound_max = ti.Vector.zero(ti.f32, 3)
            for k in ti.static(range(3)):
                bound_min[k] = min(a[k], b[k], c[k])
                bound_max[k] = max(a[k], b[k], c[k])

            p_min = int(ti.floor(bound_min[0] * self.inv_dx))
            p_max = int(ti.floor(bound_max[0] * self.inv_dx)) + 1

            q_min = int(ti.floor(bound_min[1] * self.inv_dx))
            q_max = int(ti.floor(bound_max[1] * self.inv_dx)) + 1

            normal = ti.normalized(ti.cross(b - a, c - a))
            
            if abs(normal[2]) > 1e-8:

                a_proj = ti.Vector([a[0], a[1]])
                b_proj = ti.Vector([b[0], b[1]])
                c_proj = ti.Vector([c[0], c[1]])

                for p in range(p_min, p_max):
                    for q in range(q_min, q_max):
                        pos2d = ti.Vector([p * self.dx, q * self.dx])
                        if inside_ccw(pos2d, a_proj, b_proj, c_proj) or inside_ccw(pos2d, a_proj, c_proj, b_proj):
                            base_voxel = ti.Vector([p * self.dx, q * self.dx, 0])
                            height = int(
                                -ti.dot(normal, base_voxel - a) /
                                normal[2] * self.inv_dx)
                            height = min(height, self.res[1] - 1)
                            inc = 0
                            if normal[2] > 0:
                                inc = 1
                            else:
                                inc = -1
                            self.fill(p, q, height, inc)

    def voxelize(self, triangles):
        assert isinstance(triangles, np.ndarray)
        assert triangles.dtype == np.float32
        assert len(triangles.shape) == 2
        assert triangles.shape[1] == 9

        self.voxels.fill(0)
        num_triangles = len(triangles)
        self.voxelize_triangles(num_triangles, triangles)


if __name__ == '__main__':
    n = 256
    vox = Voxelizer((n, n, n), 1.0 / n)
    # triangle = np.array([[0.1, 0.1, 0.1, 0.6, 0.2, 0.1, 0.5, 0.7,
    #                       0.7]]).astype(np.float32)
    triangles = np.fromfile('triangles.npy', dtype=np.float32)
    triangles = np.reshape(triangles, (len(triangles) // 9, 9)) * 0.306 + 0.501
    print(triangles.shape)
    print(triangles.max())
    print(triangles.min())

    vox.voxelize(triangles)

    voxels = vox.voxels.to_numpy().astype(np.float32)

    gui = ti.GUI('cross section', (n, n))
    for i in range(n):
        gui.set_image(voxels[:, :, i])
        gui.show(f'outputs/{i:04d}.png')
