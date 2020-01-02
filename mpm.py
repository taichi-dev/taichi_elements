import taichi as ti
import random
import numpy as np

quality = 1  # Use a larger value for higher-res simulations
n_particles, n_grid = 9000 * quality**2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5)**2, 1
p_mass = p_vol * p_rho
E, nu = 0.1e4, 0.2  # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / (
    (1 + nu) * (1 - 2 * nu))  # Lame parameters
ti.cfg.arch = ti.cuda  # Try to run on GPU


class MPMSolver:

  def __init__(self):
    self.x = ti.Vector(2, dt=ti.f32, shape=n_particles)  # position
    self.v = ti.Vector(2, dt=ti.f32, shape=n_particles)  # velocity
    self.C = ti.Matrix(
        2, 2, dt=ti.f32, shape=n_particles)  # affine velocity field
    self.F = ti.Matrix(
        2, 2, dt=ti.f32, shape=n_particles)  # deformation gradient
    self.material = ti.var(dt=ti.i32, shape=n_particles)  # material id
    self.Jp = ti.var(dt=ti.f32, shape=n_particles)  # plastic deformation
    self.grid_v = ti.Vector(
        2, dt=ti.f32, shape=(n_grid, n_grid))  # grid node momemtum/velocity
    self.grid_m = ti.var(dt=ti.f32, shape=(n_grid, n_grid))  # grid node mass

  @ti.classkernel
  def substep(self):
    for i, j in ti.ndrange(n_grid, n_grid):
      self.grid_v[i, j] = [0, 0]
      self.grid_m[i, j] = 0

    # Particle state update and scatter to grid (P2G)
    for p in range(n_particles):
      base = (self.x[p] * inv_dx - 0.5).cast(int)
      fx = self.x[p] * inv_dx - base.cast(float)
      # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
      w = [
          0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1), 0.5 * ti.sqr(fx - 0.5)
      ]
      self.F[p] = (ti.Matrix.identity(ti.f32, 2) +
                   dt * self.C[p]) @ self.F[p]  # deformation gradient update
      # Hardening coefficient: snow gets harder when compressed
      h = ti.exp(10 * (1.0 - self.Jp[p]))
      if self.material[p] == 1:  # jelly, make it softer
        h = 0.3
      mu, la = mu_0 * h, lambda_0 * h
      if self.material[p] == 0:  # liquid
        mu = 0.0
      U, sig, V = ti.svd(self.F[p])
      J = 1.0
      for d in ti.static(range(2)):
        new_sig = sig[d, d]
        if self.material[p] == 2:  # Snow
          new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
        self.Jp[p] *= sig[d, d] / new_sig
        sig[d, d] = new_sig
        J *= new_sig
      if self.material[p] == 0:
        # Reset deformation gradient to avoid numerical instability
        self.F[p] = ti.Matrix.identity(ti.f32, 2) * ti.sqrt(J)
      elif self.material[p] == 2:
        # Reconstruct elastic deformation gradient after plasticity
        self.F[p] = U @ sig @ V.T()

      stress = 2 * mu * (self.F[p] - U @ V.T()) @ self.F[p].T() + ti.Matrix.identity(
          ti.f32, 2) * la * J * (
              J - 1)
      stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
      affine = stress + p_mass * self.C[p]

      # Loop over 3x3 grid node neighborhood
      for i, j in ti.static(ti.ndrange(3, 3)):
        offset = ti.Vector([i, j])
        dpos = (offset.cast(float) - fx) * dx
        weight = w[i][0] * w[j][1]
        self.grid_v[base + offset] += weight * (p_mass * self.v[p] + affine @ dpos)
        self.grid_m[base + offset] += weight * p_mass

    for i, j in ti.ndrange(n_grid, n_grid):
      if self.grid_m[i, j] > 0:  # No need for epsilon here
        self.grid_v[i, j] = (1 / self.grid_m[i, j]) * self.grid_v[i, j]  # Momentum to velocity
        self.grid_v[i, j][1] -= dt * 50  # gravity
        if i < 3 and self.grid_v[i, j][0] < 0:
          self.grid_v[i, j][0] = 0  # Boundary conditions
        if i > n_grid - 3 and self.grid_v[i, j][0] > 0:
          self.grid_v[i, j][0] = 0
        if j < 3 and self.grid_v[i, j][1] < 0:
          self.grid_v[i, j][1] = 0
        if j > n_grid - 3 and self.grid_v[i, j][1] > 0:
          self.grid_v[i, j][1] = 0
    for p in range(n_particles):  # grid to particle (G2P)
      base = (self.x[p] * inv_dx - 0.5).cast(int)
      fx = self.x[p] * inv_dx - base.cast(float)
      w = [
          0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1.0),
          0.5 * ti.sqr(fx - 0.5)
      ]
      new_v = ti.Vector.zero(ti.f32, 2)
      new_C = ti.Matrix.zero(ti.f32, 2, 2)
      for i, j in ti.static(ti.ndrange(
          3, 3)):  # loop over 3x3 grid node neighborhood
        dpos = ti.Vector([i, j]).cast(float) - fx
        g_v = self.grid_v[base + ti.Vector([i, j])]
        weight = w[i][0] * w[j][1]
        new_v += weight * g_v
        new_C += 4 * inv_dx * weight * ti.outer_product(g_v, dpos)
      self.v[p], self.C[p] = new_v, new_C
      self.x[p] += dt * self.v[p]  # advection

  def init(self):
    group_size = n_particles // 3
    for i in range(n_particles):
      self.x[i] = [
          random.random() * 0.2 + 0.3 + 0.10 * (i // group_size),
          random.random() * 0.2 + 0.05 + 0.32 * (i // group_size)
      ]
      self.material[i] = i // group_size  # 0: fluid 1: jelly 2: snow
      self.v[i] = [0, 0]
      self.F[i] = [[1, 0], [0, 1]]
      self.Jp[i] = 1


gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)
mpm = MPMSolver()
mpm.init()
for frame in range(20000):
  for s in range(int(2e-3 // dt)):
    mpm.substep()
  colors = np.array([0x068587, 0xED553B, 0xEEEEF0], dtype=np.uint32)
  gui.circles(mpm.x.to_numpy(), radius=1.5, color=colors[mpm.material.to_numpy()])
  gui.show()  # Change to gui.show(f'{frame:06d}.png') to write images to disk
