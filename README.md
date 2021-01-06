# Taichi Elements [![Build Status](https://travis-ci.com/taichi-dev/taichi_elements.svg?branch=master)](https://travis-ci.com/taichi-dev/taichi_elements) [![Code Coverage](https://codecov.io/gh/taichi-dev/taichi_elements/branch/master/graph/badge.svg)](https://codecov.io/gh/taichi-dev/taichi_elements)
High-Performance Multi-Material Continuum Physics Engine (work in progress). 

The solver is being developed using Taichi, therefore it is cross-platform and supports multithreaded CPUs and massively parallel GPUs. 

The plan is
 - To build a reusable MLS-MPM multimaterial (water/elastic/snow/sand/mud) simulator
 - To integrate the simulator into Blender

# Using `taichi_elements` in Python
 - Install [taichi](https://github.com/taichi-dev/taichi) with `pip`: `python3 -m pip install taichi`
 - `python3 demo/demo_2d.py` and you will see
 <img src="https://github.com/yuanming-hu/public_files/raw/master/graphics/elements/demo_2d.gif">
 
 - `python3 demo_3d.py` and you will see a 3D simulation visualized in 2D.

## To simulate and render an example 3D scene with Python
 - Make sure you have a modern NVIDIA GPU (e.g. GTX 1080 Ti)
 - Download [`taichi.ply`](https://github.com/taichi-dev/taichi_elements_blender_examples/releases/download/ply/taichi.ply) and run `python3 demo_3d_letters.py` (wait for at least 10 frames)
   - A binary particle folder with a timestamp in its time (e.g., `sim_2020-07-27_20-55-48`) will be created under the current folder.
 - `python3 engine/render_particles [particle_input_folder] [begin] [end] [step] [render_output_folder]`
   - E.g., `python3 engine/render_particles.py sim_2020-07-27_20-55-48/ 0 100 1 frames`
 - Images are in the `rendered` folder. For example, 100 million MPM particles simulated in 8 hours on a V100 GPU:

[[Watch on YouTube]](https://www.youtube.com/watch?v=oiuSax_iPto)
<img src="https://raw.githubusercontent.com/taichi-dev/public_files/master/taichi_elements/100Mparticles.jpg" height="600px">

# Using `taichi_elements` in Blender

## Installing Taichi for Blender
(Not sure if it is the standard approach, but it works for now.)
 - Find the Python3 executable bundled in Blender. Open a console in Blender and type in
 ```
 import sys
 print(sys.exec_prefix)
 ```
  The output looks like `/XXX/blender-2.81a-linux-glibc217-x86_64/2.81`, which means python3 is located at `/XXX/blender-2.81a-linux-glibc217-x86_64/2.81/python/bin/python3.7`
 - Install [pip](https://pip.pypa.io/en/stable/installing/) using that Python executable
 - Install Taichi: `./python3.7m -m pip install --upgrade taichi` (Note: as of July 28 2020, Taichi version is `v0.6.22`. Please use the latest version.)

## Installing taichi_elements (experimental) for Blender
 - Set the environment variable `BLENDER_USER_ADDON_PATH`, e.g. `/home/XXX/.config/blender/2.81/scripts/addons`
 - Go to `utils` folder
 - Execute `python3 install_blender_addon.py` to install the addon
   - If you are doing development and wish to **k**eep refreshing the installed addon, add argument `-k`.
 - Restart Blender to reload the addon
