import taichi as ti
import numpy as np
import time
import numbers
import math
import multiprocessing as mp

USE_IN_BLENDER = False


# TODO: water needs Jp - fix this.


@ti.data_oriented
class MPMSolver:
    material_water = 0
    material_elastic = 1
    material_snow = 2
    material_sand = 3
    material_stationary = 4
    materials = {
        'WATER': material_water,
        'ELASTIC': material_elastic,
        'SNOW': material_snow,
        'SAND': material_sand,
        'STATIONARY': material_stationary,
    }

    # Surface boundary conditions

    # Stick to the boundary
    surface_sticky = 0
    # Slippy boundary
    surface_slip = 1
    # Slippy and free to separate
    surface_separate = 2

    surfaces = {
        'STICKY': surface_sticky,
        'SLIP': surface_slip,
        'SEPARATE': surface_separate
    }

    def __init__(
            self,
            res,
            quant=False,
            use_voxelizer=True,
            size=1,
            max_num_particles=2**30,
            # Max 1 G particles
            padding=3,
            unbounded=False,
            dt_scale=1,
            E_scale=1,
            voxelizer_super_sample=2,
            use_g2p2g=False,  # Ref: A massively parallel and scalable multi-GPU material point method
            v_clamp_g2p2g=True,
            use_bls=True,
            g2p2g_allowed_cfl=0.9,  # 0.0 for no CFL limit
            water_density=1.0,
            support_plasticity=True,  # Support snow and sand materials
            use_adaptive_dt=False,
            use_ggui=False,
            use_emitter_id=False
    ):
        self.dim = len(res)
        self.quant = quant
        self.use_g2p2g = use_g2p2g
        self.v_clamp_g2p2g = v_clamp_g2p2g
        self.use_bls = use_bls
        self.g2p2g_allowed_cfl = g2p2g_allowed_cfl
        self.water_density = water_density
        self.grid_size = 4096

        assert self.dim in (
            2, 3), "MPM solver supports only 2D and 3D simulations."

        self.t = 0.0
        self.res = res
        self.n_particles = ti.field(ti.i32, shape=())
        self.dx = size / res[0]
        self.inv_dx = 1.0 / self.dx
        self.default_dt = 2e-2 * self.dx / size * dt_scale
        self.p_vol = self.dx**self.dim
        self.p_rho = 1000
        self.p_mass = self.p_vol * self.p_rho
        self.max_num_particles = max_num_particles
        self.gravity = ti.Vector.field(self.dim, dtype=ti.f32, shape=())
        self.source_bound = ti.Vector.field(self.dim, dtype=ti.f32, shape=2)
        self.source_velocity = ti.Vector.field(self.dim,
                                               dtype=ti.f32,
                                               shape=())
        self.input_grid = 0
        self.all_time_max_velocity = 0
        self.support_plasticity = support_plasticity
        self.use_adaptive_dt = use_adaptive_dt
        self.use_ggui = use_ggui
        self.F_bound = 4.0

        # Affine velocity field
        if not self.use_g2p2g:
            self.C = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32)
        # Deformation gradient

        if quant:
            ci21 = ti.type_factory.custom_int(21, True)
            cft = ti.type_factory.custom_float(significand_type=ci21,
                                               scale=1 / (2**19))
            self.x = ti.Vector.field(self.dim, dtype=cft)

            cu6 = ti.type_factory.custom_int(7, False)
            ci19 = ti.type_factory.custom_int(19, True)
            cft = ti.type_factory.custom_float(significand_type=ci19,
                                               exponent_type=cu6)
            self.v = ti.Vector.field(self.dim, dtype=cft)

            ci16 = ti.type_factory.custom_int(16, True)
            cft = ti.type_factory.custom_float(significand_type=ci16,
                                               scale=(self.F_bound + 0.1) /
                                               (2**15))
            self.F = ti.Matrix.field(self.dim, self.dim, dtype=cft)
        else:
            self.v = ti.Vector.field(self.dim, dtype=ti.f32)
            self.x = ti.Vector.field(self.dim, dtype=ti.f32)
            self.F = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32)

        self.use_emitter_id = use_emitter_id
        if self.use_emitter_id:
            self.emitter_ids = ti.field(dtype=ti.i32)

        self.last_time_final_particles = ti.field(dtype=ti.i32, shape=())
        # Material id
        if quant and self.dim == 3:
            self.material = ti.field(dtype=ti.quant.int(16, False))
        else:
            self.material = ti.field(dtype=ti.i32)
        # Particle color
        self.color = ti.field(dtype=ti.i32)
        if self.use_ggui:
            self.color_with_alpha = ti.Vector.field(4, dtype=ti.f32)
        # Plastic deformation volume ratio
        if self.support_plasticity:
            self.Jp = ti.field(dtype=ti.f32)

        if self.dim == 2:
            indices = ti.ij
        else:
            indices = ti.ijk

        if unbounded:
            # The maximum grid size must be larger than twice of
            # simulation resolution in an unbounded simulation,
            # Otherwise the top and right sides will be bounded by grid size
            while self.grid_size <= 2 * max(self.res):
                self.grid_size *= 2  # keep it power of two
        offset = tuple(-self.grid_size // 2 for _ in range(self.dim))
        self.offset = offset

        self.num_grids = 2 if self.use_g2p2g else 1

        grid_block_size = 128
        if self.dim == 2:
            self.leaf_block_size = 16
        else:
            # TODO: use 8?
            self.leaf_block_size = 4

        self.grid = []
        self.grid_v = []
        self.grid_m = []
        self.pid = []

        for g in range(self.num_grids):
            # Grid node momentum/velocity
            grid_v = ti.Vector.field(self.dim, dtype=ti.f32)
            grid_m = ti.field(dtype=ti.f32)
            pid = ti.field(ti.i32)
            self.grid_v.append(grid_v)
            # Grid node mass
            self.grid_m.append(grid_m)
            grid = ti.root.pointer(indices, self.grid_size // grid_block_size)
            block = grid.pointer(indices,
                                 grid_block_size // self.leaf_block_size)
            self.block = block
            self.grid.append(grid)

            def block_component(c):
                block.dense(indices, self.leaf_block_size).place(c,
                                                                 offset=offset)

            block_component(grid_m)
            for d in range(self.dim):
                block_component(grid_v.get_scalar_field(d))

            self.pid.append(pid)

            block_offset = tuple(o // self.leaf_block_size
                                 for o in self.offset)
            self.block_offset = block_offset
            block.dynamic(ti.axes(self.dim),
                          1024 * 1024,
                          chunk_size=self.leaf_block_size**self.dim * 8).place(
                              pid, offset=block_offset + (0, ))

        self.padding = padding

        # Young's modulus and Poisson's ratio
        self.E, self.nu = 1e6 * size * E_scale, 0.2
        # Lame parameters
        self.mu_0, self.lambda_0 = self.E / (
            2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) *
                                                    (1 - 2 * self.nu))

        # Sand parameters
        friction_angle = math.radians(45)
        sin_phi = math.sin(friction_angle)
        self.alpha = math.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)

        # An empirically optimal chunk size is 1/10 of the expected particle number
        chunk_size = 2**20 if self.dim == 2 else 2**23
        self.particle = ti.root.dynamic(ti.i, max_num_particles, chunk_size)

        if self.quant:
            if not self.use_g2p2g:
                self.particle.place(self.C)
            if self.support_plasticity:
                self.particle.place(self.Jp)
            self.particle.bit_struct(num_bits=64).place(self.x)
            self.particle.bit_struct(num_bits=64).place(self.v,
                                                        shared_exponent=True)

            if self.dim == 3:
                self.particle.bit_struct(num_bits=32).place(
                    self.F.get_scalar_field(0, 0),
                    self.F.get_scalar_field(0, 1))
                self.particle.bit_struct(num_bits=32).place(
                    self.F.get_scalar_field(0, 2),
                    self.F.get_scalar_field(1, 0))
                self.particle.bit_struct(num_bits=32).place(
                    self.F.get_scalar_field(1, 1),
                    self.F.get_scalar_field(1, 2))
                self.particle.bit_struct(num_bits=32).place(
                    self.F.get_scalar_field(2, 0),
                    self.F.get_scalar_field(2, 1))
                self.particle.bit_struct(num_bits=32).place(
                    self.F.get_scalar_field(2, 2), self.material)
            else:
                assert self.dim == 2
                self.particle.bit_struct(num_bits=32).place(
                    self.F.get_scalar_field(0, 0),
                    self.F.get_scalar_field(0, 1))
                self.particle.bit_struct(num_bits=32).place(
                    self.F.get_scalar_field(1, 0),
                    self.F.get_scalar_field(1, 1))
                # No quantization on particle material in 2D
                self.particle.place(self.material)
            self.particle.place(self.color)
            if self.use_emitter_id:
                self.particle.place(self.emitter_ids)
        else:
            if self.use_emitter_id:
                self.particle.place(self.x, self.v, self.F, self.material,
                                self.color, self.emitter_ids)
            else:
                self.particle.place(self.x, self.v, self.F, self.material,
                                self.color)
            if self.support_plasticity:
                self.particle.place(self.Jp)
            if not self.use_g2p2g:
                self.particle.place(self.C)

        if self.use_ggui:
            self.particle.place(self.color_with_alpha)

        self.total_substeps = 0
        self.unbounded = unbounded

        if self.dim == 2:
            self.voxelizer = None
            self.set_gravity((0, -9.8))
        else:
            if use_voxelizer:
                if USE_IN_BLENDER:
                    from .voxelizer import Voxelizer
                else:
                    from engine.voxelizer import Voxelizer
                self.voxelizer = Voxelizer(res=self.res,
                                           dx=self.dx,
                                           padding=self.padding,
                                           super_sample=voxelizer_super_sample)
            else:
                self.voxelizer = None
            self.set_gravity((0, -9.8, 0))

        self.voxelizer_super_sample = voxelizer_super_sample

        self.grid_postprocess = []

        self.add_bounding_box(self.unbounded)

        self.writers = []

        if not self.use_g2p2g:
            self.grid = self.grid[0]
            self.grid_v = self.grid_v[0]
            self.grid_m = self.grid_m[0]
            self.pid = self.pid[0]

    def stencil_range(self):
        return ti.ndrange(*((3, ) * self.dim))

    def set_gravity(self, g):
        assert isinstance(g, (tuple, list))
        assert len(g) == self.dim
        self.gravity[None] = g

    @ti.func
    def sand_projection(self, sigma, p):
        sigma_out = ti.Matrix.zero(ti.f32, self.dim, self.dim)
        epsilon = ti.Vector.zero(ti.f32, self.dim)
        for i in ti.static(range(self.dim)):
            epsilon[i] = ti.log(max(abs(sigma[i, i]), 1e-4))
            sigma_out[i, i] = 1
        tr = epsilon.sum() + self.Jp[p]
        epsilon_hat = epsilon - tr / self.dim
        epsilon_hat_norm = epsilon_hat.norm() + 1e-20
        if tr >= 0.0:
            self.Jp[p] = tr
        else:
            self.Jp[p] = 0.0
            delta_gamma = epsilon_hat_norm + (
                self.dim * self.lambda_0 +
                2 * self.mu_0) / (2 * self.mu_0) * tr * self.alpha
            for i in ti.static(range(self.dim)):
                sigma_out[i, i] = ti.exp(epsilon[i] - max(0, delta_gamma) /
                                         epsilon_hat_norm * epsilon_hat[i])

        return sigma_out

    @ti.kernel
    def build_pid(self, pid: ti.template(), grid_m: ti.template(),
                  offset: ti.template()):
        """
        grid has blocking (e.g. 4x4x4), we wish to put the particles from each block into a GPU block,
        then used shared memory (ti.block_local) to accelerate
        :param pid:
        :param grid_m:
        :param offset:
        :return:
        """
        ti.block_dim(64)
        for p in self.x:
            base = int(ti.floor(self.x[p] * self.inv_dx - 0.5)) \
                   - ti.Vector(list(self.offset))
            # Pid grandparent is `block`
            base_pid = ti.rescale_index(grid_m, pid.parent(2), base)
            ti.append(pid.parent(), base_pid, p)

    @ti.kernel
    def g2p2g(self, dt: ti.f32, pid: ti.template(), grid_v_in: ti.template(),
              grid_v_out: ti.template(), grid_m_out: ti.template()):
        ti.block_dim(256)
        ti.no_activate(self.particle)
        if ti.static(self.use_bls):
            ti.block_local(grid_m_out)
            for d in ti.static(range(self.dim)):
                ti.block_local(grid_v_in.get_scalar_field(d))
                ti.block_local(grid_v_out.get_scalar_field(d))
        for I in ti.grouped(pid):
            p = pid[I]
            # G2P
            base = ti.floor(self.x[p] * self.inv_dx - 0.5).cast(int)
            Im = ti.rescale_index(pid, grid_m_out, I)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], Im[D], 0, 1)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            w = [
                0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2
            ]
            new_v = ti.Vector.zero(ti.f32, self.dim)
            C = ti.Matrix.zero(ti.f32, self.dim, self.dim)
            # Loop over 3x3 grid node neighborhood
            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = offset.cast(float) - fx
                g_v = grid_v_in[base + offset]
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                new_v += weight * g_v
                C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)

            if p >= self.last_time_final_particles[None]:
                # New particles. No G2P.
                new_v = self.v[p]
                C = ti.Matrix.zero(ti.f32, self.dim, self.dim)

            if self.material[p] != self.material_stationary:
                self.v[p] = new_v
                self.x[p] += dt * self.v[p]  # advection

            # P2G
            base = ti.floor(self.x[p] * self.inv_dx - 0.5).cast(int)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], Im[D], -1, 2)

            fx = self.x[p] * self.inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w2 = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            # Deformation gradient update
            new_F = (ti.Matrix.identity(ti.f32, self.dim) + dt * C) @ self.F[p]
            if ti.static(self.quant):
                new_F = max(-self.F_bound, min(self.F_bound, new_F))
            self.F[p] = new_F
            # Hardening coefficient: snow gets harder when compressed
            h = 1.0
            if ti.static(self.support_plasticity):
                h = ti.exp(10 * (1.0 - self.Jp[p]))
            if self.material[
                    p] == self.material_elastic:  # Jelly, make it softer
                h = 0.3
            mu, la = self.mu_0 * h, self.lambda_0 * h
            if self.material[p] == self.material_water:  # Liquid
                mu = 0.0
            U, sig, V = ti.svd(self.F[p])
            J = 1.0
            if self.material[p] != self.material_sand:
                for d in ti.static(range(self.dim)):
                    new_sig = sig[d, d]
                    if self.material[p] == self.material_snow:  # Snow
                        new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                                      1 + 4.5e-3)  # Plasticity
                    if ti.static(self.support_plasticity):
                        self.Jp[p] *= sig[d, d] / new_sig
                    sig[d, d] = new_sig
                    J *= new_sig
            if self.material[p] == self.material_water:
                # Reset deformation gradient to avoid numerical instability
                new_F = ti.Matrix.identity(ti.f32, self.dim)
                new_F[0, 0] = J
                self.F[p] = new_F
            elif self.material[p] == self.material_snow:
                # Reconstruct elastic deformation gradient after plasticity
                self.F[p] = U @ sig @ V.transpose()

            stress = ti.Matrix.zero(ti.f32, self.dim, self.dim)

            if self.material[p] != self.material_sand:
                stress = 2 * mu * (
                    self.F[p] - U @ V.transpose()) @ self.F[p].transpose(
                    ) + ti.Matrix.identity(ti.f32, self.dim) * la * J * (J - 1)
            else:
                if ti.static(self.support_plasticity):
                    sig = self.sand_projection(sig, p)
                    self.F[p] = U @ sig @ V.transpose()
                    log_sig_sum = 0.0
                    center = ti.Matrix.zero(ti.f32, self.dim, self.dim)
                    for i in ti.static(range(self.dim)):
                        log_sig_sum += ti.log(sig[i, i])
                        center[i, i] = 2.0 * self.mu_0 * ti.log(
                            sig[i, i]) * (1 / sig[i, i])
                    for i in ti.static(range(self.dim)):
                        center[i,
                               i] += self.lambda_0 * log_sig_sum * (1 /
                                                                    sig[i, i])
                    stress = U @ center @ V.transpose() @ self.F[p].transpose()

            stress = (-dt * self.p_vol * 4 * self.inv_dx**2) * stress
            affine = stress + self.p_mass * C

            # Loop over 3x3 grid node neighborhood
            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = (offset.cast(float) - fx) * self.dx
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w2[offset[d]][d]
                grid_v_out[base +
                           offset] += weight * (self.p_mass * self.v[p] +
                                                affine @ dpos)
                grid_m_out[base + offset] += weight * self.p_mass

        self.last_time_final_particles[None] = self.n_particles[None]

    @ti.kernel
    def p2g(self, dt: ti.f32):
        ti.no_activate(self.particle)
        ti.block_dim(256)
        if ti.static(self.use_bls):
            for d in ti.static(range(self.dim)):
                ti.block_local(self.grid_v.get_scalar_field(d))
            ti.block_local(self.grid_m)
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            base = ti.floor(self.x[p] * self.inv_dx - 0.5).cast(int)
            Im = ti.rescale_index(self.pid, self.grid_m, I)
            for D in ti.static(range(self.dim)):
                # For block shared memory: hint compiler that there is a connection between `base` and loop index `I`
                base[D] = ti.assume_in_range(base[D], Im[D], 0, 1)

            fx = self.x[p] * self.inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            # Deformation gradient update
            F = self.F[p]
            if self.material[p] == self.material_water:  # liquid
                F = ti.Matrix.identity(ti.f32, self.dim)
                if ti.static(self.support_plasticity):
                    F[0, 0] = self.Jp[p]

            F = (ti.Matrix.identity(ti.f32, self.dim) + dt * self.C[p]) @ F
            # Hardening coefficient: snow gets harder when compressed
            h = 1.0
            if ti.static(self.support_plasticity):
                if self.material[p] != self.material_water:
                    h = ti.exp(10 * (1.0 - self.Jp[p]))
            if self.material[
                    p] == self.material_elastic:  # jelly, make it softer
                h = 0.3
            mu, la = self.mu_0 * h, self.lambda_0 * h
            if self.material[p] == self.material_water:  # liquid
                mu = 0.0
            U, sig, V = ti.svd(F)
            J = 1.0
            if self.material[p] != self.material_sand:
                for d in ti.static(range(self.dim)):
                    new_sig = sig[d, d]
                    if self.material[p] == self.material_snow:  # Snow
                        new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                                      1 + 4.5e-3)  # Plasticity
                    if ti.static(self.support_plasticity):
                        self.Jp[p] *= sig[d, d] / new_sig
                    sig[d, d] = new_sig
                    J *= new_sig
            if self.material[p] == self.material_water:
                # Reset deformation gradient to avoid numerical instability
                F = ti.Matrix.identity(ti.f32, self.dim)
                F[0, 0] = J
                if ti.static(self.support_plasticity):
                    self.Jp[p] = J
            elif self.material[p] == self.material_snow:
                # Reconstruct elastic deformation gradient after plasticity
                F = U @ sig @ V.transpose()

            stress = ti.Matrix.zero(ti.f32, self.dim, self.dim)

            if self.material[p] != self.material_sand:
                stress = 2 * mu * (F - U @ V.transpose()) @ F.transpose(
                ) + ti.Matrix.identity(ti.f32, self.dim) * la * J * (J - 1)
            else:
                if ti.static(self.support_plasticity):
                    sig = self.sand_projection(sig, p)
                    F = U @ sig @ V.transpose()
                    log_sig_sum = 0.0
                    center = ti.Matrix.zero(ti.f32, self.dim, self.dim)
                    for i in ti.static(range(self.dim)):
                        log_sig_sum += ti.log(sig[i, i])
                        center[i, i] = 2.0 * self.mu_0 * ti.log(
                            sig[i, i]) * (1 / sig[i, i])
                    for i in ti.static(range(self.dim)):
                        center[i,
                               i] += self.lambda_0 * log_sig_sum * (1 /
                                                                    sig[i, i])
                    stress = U @ center @ V.transpose() @ F.transpose()
            self.F[p] = F

            stress = (-dt * self.p_vol * 4 * self.inv_dx**2) * stress
            # TODO: implement g2p2g pmass
            mass = self.p_mass
            if self.material[p] == self.material_water:
                mass *= self.water_density
            affine = stress + mass * self.C[p]

            # Loop over 3x3 grid node neighborhood
            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = (offset.cast(float) - fx) * self.dx
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                self.grid_v[base + offset] += weight * (mass * self.v[p] +
                                                        affine @ dpos)
                self.grid_m[base + offset] += weight * mass

    @ti.kernel
    def grid_normalization_and_gravity(self, dt: ti.f32, grid_v: ti.template(),
                                       grid_m: ti.template()):
        v_allowed = self.dx * self.g2p2g_allowed_cfl / dt
        for I in ti.grouped(grid_m):
            if grid_m[I] > 0:  # No need for epsilon here
                grid_v[I] = (1 / grid_m[I]) * grid_v[I]  # Momentum to velocity
                grid_v[I] += dt * self.gravity[None]

            # Grid velocity clamping
            if ti.static(self.g2p2g_allowed_cfl > 0 and self.use_g2p2g
                         and self.v_clamp_g2p2g):
                grid_v[I] = min(max(grid_v[I], -v_allowed), v_allowed)

    @ti.kernel
    def grid_bounding_box(self, t: ti.f32, dt: ti.f32,
                          unbounded: ti.template(), grid_v: ti.template()):
        for I in ti.grouped(grid_v):
            for d in ti.static(range(self.dim)):
                if ti.static(unbounded):
                    if I[d] < -self.grid_size // 2 + self.padding and grid_v[
                            I][d] < 0:
                        grid_v[I][d] = 0  # Boundary conditions
                    if I[d] >= self.grid_size // 2 - self.padding and grid_v[
                            I][d] > 0:
                        grid_v[I][d] = 0
                else:
                    if I[d] < self.padding and grid_v[I][d] < 0:
                        grid_v[I][d] = 0  # Boundary conditions
                    if I[d] >= self.res[d] - self.padding and grid_v[I][d] > 0:
                        grid_v[I][d] = 0

    def add_sphere_collider(self, center, radius, surface=surface_sticky):
        center = list(center)

        @ti.kernel
        def collide(t: ti.f32, dt: ti.f32, grid_v: ti.template()):
            for I in ti.grouped(grid_v):
                offset = I * self.dx - ti.Vector(center)
                if offset.norm_sqr() < radius * radius:
                    if ti.static(surface == self.surface_sticky):
                        grid_v[I] = ti.Vector.zero(ti.f32, self.dim)
                    else:
                        v = grid_v[I]
                        normal = offset.normalized(1e-5)
                        normal_component = normal.dot(v)

                        if ti.static(surface == self.surface_slip):
                            # Project out all normal component
                            v = v - normal * normal_component
                        else:
                            # Project out only inward normal component
                            v = v - normal * min(normal_component, 0)

                        grid_v[I] = v

        self.grid_postprocess.append(collide)

    def clear_grid_postprocess(self):
        self.grid_postprocess.clear()

    def add_surface_collider(self,
                             point,
                             normal,
                             surface=surface_sticky,
                             friction=0.0):
        point = list(point)
        # Normalize normal
        normal_scale = 1.0 / math.sqrt(sum(x**2 for x in normal))
        normal = list(normal_scale * x for x in normal)

        if surface == self.surface_sticky and friction != 0:
            raise ValueError('friction must be 0 on sticky surfaces.')

        @ti.kernel
        def collide(t: ti.f32, dt: ti.f32, grid_v: ti.template()):
            for I in ti.grouped(grid_v):
                offset = I * self.dx - ti.Vector(point)
                n = ti.Vector(normal)
                if offset.dot(n) < 0:
                    if ti.static(surface == self.surface_sticky):
                        grid_v[I] = ti.Vector.zero(ti.f32, self.dim)
                    else:
                        v = grid_v[I]
                        normal_component = n.dot(v)

                        if ti.static(surface == self.surface_slip):
                            # Project out all normal component
                            v = v - n * normal_component
                        else:
                            # Project out only inward normal component
                            v = v - n * min(normal_component, 0)

                        if normal_component < 0 and v.norm() > 1e-30:
                            # Apply friction here
                            v = v.normalized() * max(
                                0,
                                v.norm() + normal_component * friction)

                        grid_v[I] = v

        self.grid_postprocess.append(collide)

    def add_bounding_box(self, unbounded):
        self.grid_postprocess.append(
            lambda t, dt, grid_v: self.grid_bounding_box(
                t, dt, unbounded, grid_v))

    @ti.kernel
    def g2p(self, dt: ti.f32):
        ti.block_dim(256)
        if ti.static(self.use_bls):
            for d in ti.static(range(self.dim)):
                ti.block_local(self.grid_v.get_scalar_field(d))
        ti.no_activate(self.particle)
        for I in ti.grouped(self.pid):
            p = self.pid[I]
            base = ti.floor(self.x[p] * self.inv_dx - 0.5).cast(int)
            Im = ti.rescale_index(self.pid, self.grid_m, I)
            for D in ti.static(range(self.dim)):
                base[D] = ti.assume_in_range(base[D], Im[D], 0, 1)
            fx = self.x[p] * self.inv_dx - base.cast(float)
            w = [
                0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2
            ]
            new_v = ti.Vector.zero(ti.f32, self.dim)
            new_C = ti.Matrix.zero(ti.f32, self.dim, self.dim)
            # Loop over 3x3 grid node neighborhood
            for offset in ti.static(ti.grouped(self.stencil_range())):
                dpos = offset.cast(float) - fx
                g_v = self.grid_v[base + offset]
                weight = 1.0
                for d in ti.static(range(self.dim)):
                    weight *= w[offset[d]][d]
                new_v += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)
            if self.material[p] != self.material_stationary:
                self.v[p], self.C[p] = new_v, new_C
                self.x[p] += dt * self.v[p]  # advection

    @ti.kernel
    def compute_max_velocity(self) -> ti.f32:
        max_velocity = 0.0
        for p in self.v:
            v = self.v[p]
            v_max = 0.0
            for i in ti.static(range(self.dim)):
                v_max = max(v_max, abs(v[i]))
            ti.atomic_max(max_velocity, v_max)
        return max_velocity

    @ti.kernel
    def compute_max_grid_velocity(self, grid_v: ti.template()) -> ti.f32:
        max_velocity = 0.0
        for I in ti.grouped(grid_v):
            v = grid_v[I]
            v_max = 0.0
            for i in ti.static(range(self.dim)):
                v_max = max(v_max, abs(v[i]))
            ti.atomic_max(max_velocity, v_max)
        return max_velocity

    def step(self, frame_dt, print_stat=False, smry_writer=None):
        begin_t = time.time()
        begin_substep = self.total_substeps

        substeps = int(frame_dt / self.default_dt) + 1

        dt = frame_dt / substeps
        frame_time_left = frame_dt
        if print_stat:
            print(f'needed substeps: {substeps}')

        while frame_time_left > 0:
            print('.', end='', flush=True)
            self.total_substeps += 1
            if self.use_adaptive_dt:
                if self.use_g2p2g:
                    max_grid_v = self.compute_max_grid_velocity(
                        self.grid_v[self.input_grid])
                else:
                    max_grid_v = self.compute_max_grid_velocity(
                        self.grid_v)
                cfl_dt = self.g2p2g_allowed_cfl * self.dx / (max_grid_v + 1e-6)
                dt = min(dt, cfl_dt, frame_time_left)
            frame_time_left -= dt

            if self.use_g2p2g:
                output_grid = 1 - self.input_grid
                self.grid[output_grid].deactivate_all()
                self.build_pid(self.pid[self.input_grid],
                               self.grid_m[self.input_grid], 0.5)
                self.g2p2g(dt, self.pid[self.input_grid],
                           self.grid_v[self.input_grid],
                           self.grid_v[output_grid], self.grid_m[output_grid])
                self.grid_normalization_and_gravity(dt,
                                                    self.grid_v[output_grid],
                                                    self.grid_m[output_grid])
                for p in self.grid_postprocess:
                    p(self.t, dt, self.grid_v[output_grid])
                self.input_grid = output_grid
                self.t += dt
            else:
                self.grid.deactivate_all()
                self.build_pid(self.pid, self.grid_m, 0.5)
                self.p2g(dt)
                self.grid_normalization_and_gravity(dt, self.grid_v,
                                                    self.grid_m)
                for p in self.grid_postprocess:
                    p(self.t, dt, self.grid_v)
                self.t += dt
                self.g2p(dt)

            cur_frame_velocity = self.compute_max_velocity()
            if smry_writer is not None:
                smry_writer.add_scalar("substep_max_CFL",
                                       cur_frame_velocity * dt / self.dx,
                                       self.total_substeps)
            self.all_time_max_velocity = max(self.all_time_max_velocity,
                                             cur_frame_velocity)

        print()

        if print_stat:
            ti.print_kernel_profile_info()
            try:
                ti.print_memory_profile_info()
            except:
                pass
            cur_frame_velocity = self.compute_max_velocity()
            print(f'CFL: {cur_frame_velocity * dt / self.dx}')
            print(f'num particles={self.n_particles[None]}')
            print(f'  frame time {time.time() - begin_t:.3f} s')
            print(
                f'  substep time {1000 * (time.time() - begin_t) / (self.total_substeps - begin_substep):.3f} ms'
            )

    @ti.func
    def seed_particle(self, i, x, material, color, velocity, emmiter_id):
        self.x[i] = x
        self.v[i] = velocity
        self.F[i] = ti.Matrix.identity(ti.f32, self.dim)
        self.color[i] = color
        self.material[i] = material

        if ti.static(self.support_plasticity):
            if material == self.material_sand:
                self.Jp[i] = 0
            else:
                self.Jp[i] = 1

        if ti.static(self.use_emitter_id):
            self.emitter_ids[i] = emmiter_id

    @ti.kernel
    def seed(self, new_particles: ti.i32, new_material: ti.i32, color: ti.i32):
        for i in range(self.n_particles[None],
                       self.n_particles[None] + new_particles):
            self.material[i] = new_material
            x = ti.Vector.zero(ti.f32, self.dim)
            for k in ti.static(range(self.dim)):
                x[k] = self.source_bound[0][k] + ti.random(
                ) * self.source_bound[1][k]
            self.seed_particle(i, x, new_material, color,
                               self.source_velocity[None], None)

    def set_source_velocity(self, velocity):
        if velocity is not None:
            velocity = list(velocity)
            assert len(velocity) == self.dim
            self.source_velocity[None] = velocity
        else:
            for i in range(self.dim):
                self.source_velocity[None][i] = 0

    def add_cube(self,
                 lower_corner,
                 cube_size,
                 material,
                 color=0xFFFFFF,
                 sample_density=None,
                 velocity=None):
        if sample_density is None:
            sample_density = 2**self.dim
        vol = 1
        for i in range(self.dim):
            vol = vol * cube_size[i]
        num_new_particles = int(sample_density * vol / self.dx**self.dim + 1)
        assert self.n_particles[
            None] + num_new_particles <= self.max_num_particles

        for i in range(self.dim):
            self.source_bound[0][i] = lower_corner[i]
            self.source_bound[1][i] = cube_size[i]

        self.set_source_velocity(velocity=velocity)

        self.seed(num_new_particles, material, color)
        self.n_particles[None] += num_new_particles

    def add_ngon(
        self,
        sides,
        center,
        radius,
        angle,
        material,
        color=0xFFFFFF,
        sample_density=None,
        velocity=None,
    ):
        if self.dim != 2:
            raise ValueError("Add Ngon only works for 2D simulations")

        if sample_density is None:
            sample_density = 2**self.dim

        num_particles = 0.5 * (radius * self.inv_dx)**2 * math.sin(
            2 * math.pi / sides) * sides

        num_particles = int(math.ceil(num_particles * sample_density))

        self.source_bound[0] = center
        self.source_bound[1] = [radius, radius]

        self.set_source_velocity(velocity=velocity)

        assert self.n_particles[None] + num_particles <= self.max_num_particles

        self.seed_polygon(num_particles, sides, angle, material, color)
        self.n_particles[None] += num_particles

    @ti.func
    def random_point_in_unit_polygon(self, sides, angle):
        point = ti.Vector.zero(ti.f32, 2)
        central_angle = 2 * math.pi / sides
        while True:
            point = ti.Vector([ti.random(), ti.random()]) * 2 - 1
            point_angle = ti.atan2(point.y, point.x)
            theta = (point_angle -
                     angle) % central_angle  # polygon angle is from +X axis
            phi = central_angle / 2
            dist = ti.sqrt((point**2).sum())
            if dist < ti.cos(phi) / ti.cos(phi - theta):
                break
        return point

    @ti.kernel
    def seed_polygon(self, new_particles: ti.i32, sides: ti.i32, angle: ti.f32,
                     new_material: ti.i32, color: ti.i32):
        for i in range(self.n_particles[None],
                       self.n_particles[None] + new_particles):
            x = self.random_point_in_unit_polygon(sides, angle)
            x = self.source_bound[0] + x * self.source_bound[1]
            self.seed_particle(i, x, new_material, color,
                               self.source_velocity[None], None)

    @ti.kernel
    def add_texture_2d(
            self,
            offset_x: ti.f32,
            offset_y: ti.f32,
            texture: ti.ext_arr(),
            new_material: ti.i32,
            color: ti.i32,
    ):
        for i, j in ti.ndrange(texture.shape[0], texture.shape[1]):
            if texture[i, j] > 0.1:
                pid = ti.atomic_add(self.n_particles[None], 1)
                x = ti.Vector([offset_x + i * self.dx, offset_y + j * self.dx])
                self.seed_particle(pid, x, new_material, color,
                                   self.source_velocity[None], None)

    @ti.func
    def random_point_in_unit_sphere(self):
        ret = ti.Vector.zero(ti.f32, n=self.dim)
        while True:
            for i in ti.static(range(self.dim)):
                ret[i] = ti.random(ti.f32) * 2 - 1
            if ret.norm_sqr() <= 1:
                break
        return ret

    @ti.kernel
    def seed_ellipsoid(self, new_particles: ti.i32, new_material: ti.i32,
                       color: ti.i32):

        for i in range(self.n_particles[None],
                       self.n_particles[None] + new_particles):
            x = self.source_bound[0] + self.random_point_in_unit_sphere(
            ) * self.source_bound[1]
            self.seed_particle(i, x, new_material, color,
                               self.source_velocity[None], None)

    def add_ellipsoid(self,
                      center,
                      radius,
                      material,
                      color=0xFFFFFF,
                      sample_density=None,
                      velocity=None):
        if sample_density is None:
            sample_density = 2**self.dim

        if isinstance(radius, numbers.Number):
            radius = [
                radius,
            ] * self.dim

        radius = list(radius)

        if self.dim == 2:
            num_particles = math.pi
        else:
            num_particles = 4 / 3 * math.pi

        for i in range(self.dim):
            num_particles *= radius[i] * self.inv_dx

        num_particles = int(math.ceil(num_particles * sample_density))

        self.source_bound[0] = center
        self.source_bound[1] = radius

        self.set_source_velocity(velocity=velocity)

        assert self.n_particles[None] + num_particles <= self.max_num_particles

        self.seed_ellipsoid(num_particles, material, color)
        self.n_particles[None] += num_particles

    @ti.kernel
    def seed_from_voxels(
            self,
            material: ti.i32,
            color: ti.i32,
            sample_density: ti.i32,
            emmiter_id: ti.u16
        ):
        for i, j, k in self.voxelizer.voxels:
            inside = 1
            for d in ti.static(range(3)):
                inside = inside and -self.grid_size // 2 + self.padding <= i and i < self.grid_size // 2 - self.padding
            if inside and self.voxelizer.voxels[i, j, k] > 0:
                s = sample_density / self.voxelizer_super_sample**self.dim
                for l in range(sample_density + 1):
                    if ti.random() + l < s:
                        x = ti.Vector([
                            ti.random() + i,
                            ti.random() + j,
                            ti.random() + k
                        ]) * (self.dx / self.voxelizer_super_sample
                              ) + self.source_bound[0]
                        p = ti.atomic_add(self.n_particles[None], 1)
                        self.seed_particle(
                            p,
                            x,
                            material,
                            color,
                            self.source_velocity[None],
                            emmiter_id
                        )

    def add_mesh(self,
                 triangles,
                 material,
                 color=0xFFFFFF,
                 sample_density=None,
                 velocity=None,
                 translation=None,
                 emmiter_id=0
        ):
        assert self.dim == 3
        if sample_density is None:
            sample_density = 2**self.dim

        self.set_source_velocity(velocity=velocity)

        for i in range(self.dim):
            if translation:
                self.source_bound[0][i] = translation[i]
            else:
                self.source_bound[0][i] = 0

        self.voxelizer.voxelize(triangles)
        t = time.time()
        self.seed_from_voxels(
            material,
            color,
            sample_density,
            emmiter_id
        )
        ti.sync()
        # print('Voxelization time:', (time.time() - t) * 1000, 'ms')

    @ti.kernel
    def seed_from_external_array(self, num_particles: ti.i32,
                                 pos: ti.ext_arr(), new_material: ti.i32,
                                 color: ti.i32):

        for i in range(num_particles):
            x = ti.Vector.zero(ti.f32, n=self.dim)
            if ti.static(self.dim == 3):
                x = ti.Vector([pos[i, 0], pos[i, 1], pos[i, 2]])
            else:
                x = ti.Vector([pos[i, 0], pos[i, 1]])
            self.seed_particle(self.n_particles[None] + i, x, new_material,
                               color, self.source_velocity[None], None)

        self.n_particles[None] += num_particles

    def add_particles(self,
                      particles,
                      material,
                      color=0xFFFFFF,
                      velocity=None):
        self.set_source_velocity(velocity=velocity)
        self.seed_from_external_array(len(particles), particles, material,
                                      color)

    @ti.kernel
    def recover_from_external_array(
            self,
            num_particles: ti.i32,
            pos: ti.ext_arr(),
            vel: ti.ext_arr(),
            material: ti.ext_arr(),
            color: ti.ext_arr(),
    ):
        for i in range(num_particles):
            x = ti.Vector.zero(ti.f32, n=self.dim)
            v = ti.Vector.zero(ti.f32, n=self.dim)
            if ti.static(self.dim == 3):
                x = ti.Vector([pos[i, 0], pos[i, 1], pos[i, 2]])
                v = ti.Vector([vel[i, 0], vel[i, 1], vel[i, 2]])
            else:
                x = ti.Vector([pos[i, 0], pos[i, 1]])
                v = ti.Vector([vel[i, 0], vel[i, 1]])
            self.seed_particle(self.n_particles[None] + i, x, material[i],
                               color[i], v, None)
        self.n_particles[None] += num_particles

    def read_restart(
        self,
        num_particles,
        pos,
        vel,
        material,
        color,
    ):
        slice_size = 50000
        num_slices = (num_particles + slice_size - 1) // slice_size
        for s in range(num_slices):
            begin = slice_size * s
            end = min(slice_size * (s + 1), num_particles)
            self.recover_from_external_array(end - begin, pos[begin:end],
                                             vel[begin:end],
                                             material[begin:end],
                                             color[begin:end])

    @ti.kernel
    def copy_dynamic_nd(self, np_x: ti.ext_arr(), input_x: ti.template()):
        for i in self.x:
            for j in ti.static(range(self.dim)):
                np_x[i, j] = input_x[i][j]

    @ti.kernel
    def copy_dynamic(self, np_x: ti.ext_arr(), input_x: ti.template()):
        for i in self.x:
            np_x[i] = input_x[i]

    @ti.kernel
    def copy_ranged(self, np_x: ti.ext_arr(), input_x: ti.template(),
                    begin: ti.i32, end: ti.i32):
        ti.no_activate(self.particle)
        for i in range(begin, end):
            np_x[i - begin] = input_x[i]

    @ti.kernel
    def copy_ranged_nd(self, np_x: ti.ext_arr(), input_x: ti.template(),
                       begin: ti.i32, end: ti.i32):
        ti.no_activate(self.particle)
        for i in range(begin, end):
            for j in ti.static(range(self.dim)):
                np_x[i - begin, j] = input_x[i, j]

    def particle_info(self):
        np_x = np.ndarray((self.n_particles[None], self.dim), dtype=np.float32)
        self.copy_dynamic_nd(np_x, self.x)
        np_v = np.ndarray((self.n_particles[None], self.dim), dtype=np.float32)
        self.copy_dynamic_nd(np_v, self.v)
        np_material = np.ndarray((self.n_particles[None], ), dtype=np.int32)
        self.copy_dynamic(np_material, self.material)
        np_color = np.ndarray((self.n_particles[None], ), dtype=np.int32)
        self.copy_dynamic(np_color, self.color)
        particles_data = {
            'position': np_x,
            'velocity': np_v,
            'material': np_material,
            'color': np_color
        }
        if self.use_emitter_id:
            np_emitters = np.ndarray((self.n_particles[None], ), dtype=np.int32)
            self.copy_dynamic(np_emitters, self.emitter_ids)
            particles_data['emitter_ids'] = np_emitters
        return particles_data

    @ti.kernel
    def clear_particles(self):
        self.n_particles[None] = 0
        ti.deactivate(self.x.loop_range().parent().snode(), [])

    def write_particles(self, fn, slice_size=1000000):
        from .particle_io import ParticleIO
        ParticleIO.write_particles(self, fn, slice_size)

    def write_particles_ply(self, fn):
        np_x = np.ndarray((self.n_particles[None], self.dim), dtype=np.float32)
        self.copy_dynamic_nd(np_x, self.x)
        np_color = np.ndarray((self.n_particles[None]), dtype=np.uint32)
        self.copy_dynamic(np_color, self.color)
        data = np.hstack([np_x, (np_color[:, None]).view(np.float32)])
        from mesh_io import write_point_cloud
        write_point_cloud(fn, data)
