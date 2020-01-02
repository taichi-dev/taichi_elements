import taichi as ti
import random
import numpy as np

# TODO: make attributes per particle
# Young's modulus and Poisson's ratio
E, nu = 0.1e4, 0.2
# Lame parameters
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))
# Try to run on GPU
ti.cfg.arch = ti.cuda


class MPMSolver:

  def __init__(self, res):
    self.dim = len(res)
    self.res = res
    self.n_particles = 9000
    self.dx = 1 / res[0]
    self.dt = 1e-4
    self.inv_dx = float(res[0])
    self.p_vol = self.dx**self.dim
    self.p_rho = 1
    self.p_mass = self.p_vol * self.p_rho
    self.gravity = -50
    # position
    self.x = ti.Vector(self.dim, dt=ti.f32, shape=self.n_particles)
    # velocity
    self.v = ti.Vector(self.dim, dt=ti.f32, shape=self.n_particles)
    # affine velocity field
    self.C = ti.Matrix(self.dim, self.dim, dt=ti.f32, shape=self.n_particles)
    # deformation gradient
    self.F = ti.Matrix(self.dim, self.dim, dt=ti.f32, shape=self.n_particles)
    # material id
    self.material = ti.var(dt=ti.i32, shape=self.n_particles)
    # plastic deformation
    self.Jp = ti.var(dt=ti.f32, shape=self.n_particles)
    # grid node momemtum/velocity
    self.grid_v = ti.Vector(self.dim, dt=ti.f32, shape=self.res)
    # grid node mass
    self.grid_m = ti.var(dt=ti.f32, shape=self.res)

  @ti.classkernel
  def p2g(self):
    for p in self.x:
      base = (self.x[p] * self.inv_dx - 0.5).cast(int)
      fx = self.x[p] * self.inv_dx - base.cast(float)
      # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
      w = [
          0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1), 0.5 * ti.sqr(fx - 0.5)
      ]
      # deformation gradient update
      self.F[p] = (ti.Matrix.identity(ti.f32, self.dim) +
                   self.dt * self.C[p]) @ self.F[p]
      # Hardening coefficient: snow gets harder when compressed
      h = ti.exp(10 * (1.0 - self.Jp[p]))
      if self.material[p] == 1:  # jelly, make it softer
        h = 0.3
      mu, la = mu_0 * h, lambda_0 * h
      if self.material[p] == 0:  # liquid
        mu = 0.0
      U, sig, V = ti.svd(self.F[p])
      J = 1.0
      for d in ti.static(range(self.dim)):
        new_sig = sig[d, d]
        if self.material[p] == 2:  # Snow
          new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
        self.Jp[p] *= sig[d, d] / new_sig
        sig[d, d] = new_sig
        J *= new_sig
      if self.material[p] == 0:
        # Reset deformation gradient to avoid numerical instability
        self.F[p] = ti.Matrix.identity(ti.f32, self.dim) * ti.sqrt(J)
      elif self.material[p] == 2:
        # Reconstruct elastic deformation gradient after plasticity
        self.F[p] = U @ sig @ V.T()

      stress = 2 * mu * (self.F[p] - U @ V.T()) @ self.F[p].T(
      ) + ti.Matrix.identity(ti.f32, 2) * la * J * (
          J - 1)
      stress = (-self.dt * self.p_vol * 4 * self.inv_dx**2) * stress
      affine = stress + self.p_mass * self.C[p]

      # Loop over 3x3 grid node neighborhood
      for i, j in ti.static(ti.ndrange(3, 3)):
        offset = ti.Vector([i, j])
        dpos = (offset.cast(float) - fx) * self.dx
        weight = w[i][0] * w[j][1]
        self.grid_v[base + offset] += weight * (
            self.p_mass * self.v[p] + affine @ dpos)
        self.grid_m[base + offset] += weight * self.p_mass

  @ti.classkernel
  def grid_op(self):
    for i, j in self.grid_m:
      if self.grid_m[i, j] > 0:  # No need for epsilon here
        self.grid_v[i, j] = (
            1 / self.grid_m[i, j]) * self.grid_v[i, j]  # Momentum to velocity
        self.grid_v[i, j][1] += self.dt * self.gravity
        if i < 3 and self.grid_v[i, j][0] < 0:
          self.grid_v[i, j][0] = 0  # Boundary conditions
        if i > self.res[0] - 3 and self.grid_v[i, j][0] > 0:
          self.grid_v[i, j][0] = 0
        if j < 3 and self.grid_v[i, j][1] < 0:
          self.grid_v[i, j][1] = 0
        if j > self.res[1] - 3 and self.grid_v[i, j][1] > 0:
          self.grid_v[i, j][1] = 0

  @ti.classkernel
  def g2p(self):
    for p in self.x:
      base = (self.x[p] * self.inv_dx - 0.5).cast(int)
      fx = self.x[p] * self.inv_dx - base.cast(float)
      w = [
          0.5 * ti.sqr(1.5 - fx), 0.75 - ti.sqr(fx - 1.0),
          0.5 * ti.sqr(fx - 0.5)
      ]
      new_v = ti.Vector.zero(ti.f32, self.dim)
      new_C = ti.Matrix.zero(ti.f32, self.dim, self.dim)
      # loop over 3x3 grid node neighborhood
      for i, j in ti.static(ti.ndrange(3, 3)):
        dpos = ti.Vector([i, j]).cast(float) - fx
        g_v = self.grid_v[base + ti.Vector([i, j])]
        weight = w[i][0] * w[j][1]
        new_v += weight * g_v
        new_C += 4 * self.inv_dx * weight * ti.outer_product(g_v, dpos)
      self.v[p], self.C[p] = new_v, new_C
      self.x[p] += self.dt * self.v[p]  # advection

  def substep(self):
    self.grid_v.fill(0)
    self.grid_m.fill(0)
    self.p2g()
    self.grid_op()
    self.g2p()

  def init(self):
    group_size = self.n_particles // 3
    for i in range(self.n_particles):
      self.x[i] = [
          random.random() * 0.2 + 0.3 + 0.10 * (i // group_size),
          random.random() * 0.2 + 0.05 + 0.32 * (i // group_size)
      ]
      self.material[i] = i // group_size  # 0: fluid 1: jelly 2: snow
      self.v[i] = [0, 0]
      self.F[i] = [[1, 0], [0, 1]]
      self.Jp[i] = 1


gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)
mpm = MPMSolver(res=(128, 128))
mpm.init()

for frame in range(20000):
  for s in range(int(2e-3 // mpm.dt)):
    mpm.substep()
  colors = np.array([0x068587, 0xED553B, 0xEEEEF0], dtype=np.uint32)
  gui.circles(
      mpm.x.to_numpy(), radius=1.5, color=colors[mpm.material.to_numpy()])
  gui.show()  # Change to gui.show(f'{frame:06d}.png') to write images to disk
